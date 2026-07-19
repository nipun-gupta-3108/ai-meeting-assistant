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
- "task": a short description of the action item.
- "owner": the person responsible, or "Not specified" if unknown.
- "deadline": the due date, or "Not specified" if unknown.
- If there are no action items, return an empty array.

"key_decisions": an array of at most 5 short strings, each describing one decision made.
- If there are no decisions, return an empty array.

"open_questions": an array of at most 5 short strings, each describing one unresolved \
question, blocker, or follow-up.
- If there are none, return an empty array.

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
