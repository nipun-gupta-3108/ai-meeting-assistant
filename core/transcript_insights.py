from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.llm_client import create_llm
import re


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


INSIGHTS_SYSTEM_PROMPT = """Review the meeting transcript and extract three things.
Return your answer as markdown using EXACTLY these three section headers,
in this order, with nothing before the first header:

## Action Items
For each item include Task, Owner (or "Not specified"), and Deadline (or "Not specified").
Numbered list. If none, write "No action items found."

## Key Decisions
Numbered list of decisions made. If none, write "No key decisions found."

## Open Questions
Numbered list of unanswered questions, blockers, or follow-ups. If none, write "No open questions found."
"""

# Built once at import time and reused for every meeting, instead of
# rebuilding the prompt/chain on every call.
INSIGHTS_CHAIN = build_extraction_chain(INSIGHTS_SYSTEM_PROMPT)

# Fallback text used if a section header is missing or unparsable from the model's output
_DEFAULTS = {
    "action_items": "No action items found.",
    "key_decisions": "No key decisions found.",
    "open_questions": "No open questions found.",
}

# Matches "## Action Items", "## Key Decisions", "## Open Questions" (case-insensitive)
_SECTION_PATTERN = re.compile(
    r"##\s*(Action Items|Key Decisions|Open Questions)\s*\n(.*?)(?=\n##\s*(?:Action Items|Key Decisions|Open Questions)|\Z)",
    re.IGNORECASE | re.DOTALL,
)

_HEADER_TO_KEY = {
    "action items": "action_items",
    "key decisions": "key_decisions",
    "open questions": "open_questions",
}


def parse_insight_sections(raw_text: str) -> dict:
    """Split the model's markdown output into action_items / key_decisions / open_questions.

    Defensive by design: always returns all three keys, and never raises,
    even if the model's output is empty, malformed, or only partially
    follows the requested format.
    """
    result = dict(_DEFAULTS)

    if not raw_text:
        return result

    try:
        matches = _SECTION_PATTERN.findall(raw_text)
    except Exception:
        # Regex should not fail on a string input, but guard anyway —
        # falling back to defaults is always safer than raising.
        return result

    for header, body in matches:
        key = _HEADER_TO_KEY.get(header.strip().lower())
        if key is None:
            continue
        content = body.strip()
        if content:
            result[key] = content

    return result


def extract_meeting_insights_from_transcript(transcript: str) -> dict:
    """Single Groq call returning action items, key decisions, and open questions."""
    raw_output = INSIGHTS_CHAIN.invoke(transcript)
    return parse_insight_sections(raw_output)
