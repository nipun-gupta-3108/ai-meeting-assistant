from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.llm_client import create_llm
import json

# Maximum number of items kept per section, enforced both in the prompt
# and again here in Python — the prompt limit alone can't be trusted since
# the model may not always comply.
MAX_ITEMS_PER_SECTION = 5


def build_extraction_chain(system_prompt: str):
    llm = create_llm(temperature=0.2)
    return (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | ChatPromptTemplate.from_messages(
            [
                ("system", system_prompt),
                ("human", "{text}"),
            ]
        )
        | llm
        | StrOutputParser()
    )


INSIGHTS_SYSTEM_PROMPT = """You are an expert meeting analyst. Read the meeting transcript and extract structured insights.

Return ONLY a single valid JSON object. Do not include markdown code fences \
(no ```), explanations, headings, or any text outside the JSON object.

The JSON object must have exactly these three keys:

"action_items": an array of at most 5 objects, each with keys "task", "owner", "deadline".
- Only include items that are explicitly assigned to someone or explicitly agreed \
upon in the transcript. Do not convert a suggestion, idea, or proposal into an \
action item unless the transcript shows it was explicitly assigned or agreed to.
- If multiple items refer to the same underlying task, merge them into a single \
entry rather than listing them separately.
- "task": a short, objective description of the action item.
- "owner": the person responsible, exactly as stated in the transcript. Use \
"Not specified" if no owner is explicitly named — never infer an owner from role, \
tone, or context.
- "deadline": the due date, exactly as stated in the transcript. Use "Not specified" \
if no deadline is explicitly given — never infer a deadline.
- Order items by priority: items with an explicit deadline or explicitly stated \
urgency first, followed by other agreed items.
- If there are no action items, return an empty array.

"key_decisions": an array of at most 5 short strings, each describing one decision \
that the transcript explicitly frames as agreed or finalized.
- Do not include proposals, suggestions, or options that were only discussed or \
left open.
- Do not list the same decision more than once even if it is phrased differently \
in different parts of the transcript — merge these into a single entry.
- If there are no decisions, return an empty array.

"open_questions": an array of at most 5 short strings, each describing one question, \
blocker, or follow-up that remains unresolved by the end of the transcript.
- Exclude questions that are answered later in the transcript and exclude rhetorical \
questions that aren't genuine open items.
- If there are none, return an empty array.

Use objective, neutral language throughout; do not add sentiment or editorializing \
that isn't explicitly present in the transcript.

Do not invent facts that are not present in the transcript. Do not repeat the same \
item twice; merge duplicate or overlapping items into one entry.

Return only the JSON object, nothing else."""

# Built lazily on first use and cached, instead of at import time —
# building at import time would require GROQ_API_KEY before
# load_dotenv() runs in streamlit_app.py, causing a startup crash.
_INSIGHTS_CHAIN = None


def get_insights_chain():
    global _INSIGHTS_CHAIN
    if _INSIGHTS_CHAIN is None:
        _INSIGHTS_CHAIN = build_extraction_chain(INSIGHTS_SYSTEM_PROMPT)
    return _INSIGHTS_CHAIN


# Returned when the model's output is empty or cannot be parsed as JSON at
# all. Always all three keys, always list-shaped, so callers never need to
# special-case a parse failure.
_EMPTY_RESULT = {
    "action_items": [],
    "key_decisions": [],
    "open_questions": [],
}


def _extract_json_object(raw_text: str) -> str:
    """Best-effort extraction of a JSON object from the model's raw response.

    The prompt asks for JSON only, but models sometimes wrap the object in
    markdown code fences anyway. This strips fences and slices out the
    outermost {...} span so json.loads gets the best possible input.
    """
    text = raw_text.strip()

    if text.startswith("```"):
        text = text.strip("`")
        if text[:4].lower() == "json":
            text = text[4:]
        text = text.strip()

    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text

    return text[start : end + 1]


def _normalize_action_item(item) -> dict | None:
    """Coerce one raw action-item entry into {task, owner, deadline}, or
    None if it has no usable task text."""
    if not isinstance(item, dict):
        return None

    task = str(item.get("task", "")).strip()
    if not task:
        return None

    owner = str(item.get("owner") or "").strip() or "Not specified"
    deadline = str(item.get("deadline") or "").strip() or "Not specified"

    return {"task": task, "owner": owner, "deadline": deadline}


def _normalize_string_list(values) -> list:
    """Coerce a raw list into a list of non-empty, stripped strings."""
    if not isinstance(values, list):
        return []
    return [str(v).strip() for v in values if str(v).strip()]


def parse_insight_sections(raw_text: str) -> dict:
    """Parse the model's JSON output into action_items / key_decisions / open_questions.

    Defensive by design: always returns all three keys as lists, capped at
    MAX_ITEMS_PER_SECTION, and never raises — even if the model's output is
    empty, not valid JSON, or only partially matches the requested shape.
    """
    result = {key: list(value) for key, value in _EMPTY_RESULT.items()}

    if not raw_text:
        return result

    try:
        data = json.loads(_extract_json_object(raw_text))
    except Exception:
        return result

    if not isinstance(data, dict):
        return result

    normalized_items = [
        _normalize_action_item(item) for item in data.get("action_items", []) or []
    ]
    result["action_items"] = [item for item in normalized_items if item is not None][
        :MAX_ITEMS_PER_SECTION
    ]

    result["key_decisions"] = _normalize_string_list(data.get("key_decisions"))[
        :MAX_ITEMS_PER_SECTION
    ]

    result["open_questions"] = _normalize_string_list(data.get("open_questions"))[
        :MAX_ITEMS_PER_SECTION
    ]

    return result


def extract_meeting_insights_from_transcript(transcript: str) -> dict:
    """Single Groq call returning action items, key decisions, and open questions
    as structured lists (see parse_insight_sections for the exact shape)."""
    raw_output = get_insights_chain().invoke(transcript)
    return parse_insight_sections(raw_output)
