from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from google.api_core.exceptions import GoogleAPICallError
from groq import APIStatusError
from core.llm_client import LLMServiceError, create_summary_llm
import json
import re

import logging

logger = logging.getLogger(__name__)

MAX_SUMMARY_BULLETS = 5
MAX_BULLET_WORDS = 25

COMBINE_SYSTEM_PROMPT = """You are an expert meeting summarizer.

You will be given several partial summaries from different portions of the same meeting.

Your job is to merge them into ONE concise meeting summary.

Rules:
- Maximum 5 bullets.
- Each bullet must contain at most 25 words.
- Merge duplicate or overlapping points.
- Keep important names, dates, deadlines and numbers.
- Never invent facts.
- Use neutral language.

Priority:
1. Decisions
2. Action items
3. Important facts / metrics
4. Risks / blockers
5. General discussion

Return ONLY valid JSON.

The output MUST EXACTLY follow this schema:

{{
  "summary": [
    "Bullet 1",
    "Bullet 2",
    "Bullet 3"
  ]
}}

Rules:
- summary must be a JSON array of strings.
- Every bullet must be enclosed in double quotes.
- Do NOT use markdown bullets (* or -).
- Do NOT number the bullets.
- Do NOT output markdown.
- Do NOT output explanations.
- Do NOT output anything outside the JSON object.
"""


def _extract_json_object(raw_text: str) -> str:
    text = raw_text.strip()

    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?", "", text).strip()
        text = re.sub(r"```$", "", text).strip()

    start = text.find("{")
    end = text.rfind("}")

    if start != -1 and end != -1:
        return text[start : end + 1]

    return text


def _truncate_words(text: str, limit: int) -> str:
    words = text.split()
    return " ".join(words[:limit])


def parse_summary_bullets(raw_text: str) -> list[str]:

    if not raw_text:
        return []

    try:
        data = json.loads(_extract_json_object(raw_text))

        bullets = data.get("summary", [])

        cleaned = []
        seen = set()

        for bullet in bullets:

            bullet = str(bullet).strip()

            if not bullet:
                continue

            bullet = _truncate_words(
                bullet,
                MAX_BULLET_WORDS,
            )

            if bullet.lower() in seen:
                continue

            seen.add(bullet.lower())
            cleaned.append(bullet)

        return cleaned[:MAX_SUMMARY_BULLETS]

    except Exception:

        bullets = re.findall(
            r"[*-]\s*(.+)",
            raw_text,
        )

        if bullets:
            cleaned = []

            for b in bullets:
                b = _truncate_words(
                    b.strip(),
                    MAX_BULLET_WORDS,
                )

                if b not in cleaned:
                    cleaned.append(b)

            return cleaned[:MAX_SUMMARY_BULLETS]

        stripped = raw_text.strip()

        if stripped:
            return [
                _truncate_words(
                    stripped,
                    MAX_BULLET_WORDS,
                )
            ]

        return []


def split_transcript_for_summary(transcript: str):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=3000,
        chunk_overlap=200,
    )

    return splitter.split_text(transcript)


def _invoke_llm_chain(chain, chain_input, step_name: str) -> str:
    """Invoke a summarization chain, converting provider rate-limit/service
    errors into a clean LLMServiceError instead of letting a raw SDK
    exception (and its stack trace) reach the Streamlit UI.

    Mirrors the error-handling style already used in
    core/transcript_qa.py's ask_transcript_question(): catch specific
    provider exceptions, log the full detail, and raise/return a
    short, user-facing message.
    """
    try:
        return chain.invoke(chain_input)
    except APIStatusError as exc:
        logger.exception("Groq API error during %s.", step_name)
        raise LLMServiceError(
            "The AI service is temporarily busy. Please wait a minute and " "try again."
        ) from exc
    except GoogleAPICallError as exc:
        logger.exception("Gemini API error during %s.", step_name)
        raise LLMServiceError(
            "The AI service is temporarily busy. Please wait a minute and " "try again."
        ) from exc


def summarize_transcript(transcript: str) -> list[str]:

    llm = create_summary_llm()

    map_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Summarize this portion of the meeting.

Keep:
- decisions
- action items
- deadlines
- names
- numbers
- metrics

Be concise.
Do not invent facts.
""",
            ),
            ("human", "{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript_for_summary(transcript)

    combined_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", COMBINE_SYSTEM_PROMPT),
            ("human", "{text}"),
        ]
    )

    combined_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | combined_prompt
        | llm
        | StrOutputParser()
    )

    if len(chunks) == 1:
        # Single-chunk meetings skip the map step and feed the transcript
        # chunk straight into the combine chain instead. The map prompt
        # only ever produces plain prose ("Summarize this portion..."),
        # while combine is the step that actually emits the JSON shape
        # parse_summary_bullets() expects — so skipping combine (as is done
        # on the insights side) would change the returned format. Skipping
        # map instead saves one LLM call while keeping the output identical
        # to the multi-chunk path, since COMBINE_SYSTEM_PROMPT already
        # tolerates merging just one "partial summary".
        raw_output = _invoke_llm_chain(combined_chain, chunks[0], "summary combine")
    else:
        chunk_summaries = [
            _invoke_llm_chain(map_chain, {"text": chunk}, "summary map")
            for chunk in chunks
        ]

        combined = "\n\n".join(chunk_summaries)

        raw_output = _invoke_llm_chain(combined_chain, combined, "summary combine")

    logger.debug(
        "\n%s\nRAW SUMMARY OUTPUT\n%s\n%s\n%s",
        "=" * 80,
        "=" * 80,
        raw_output,
        "=" * 80,
    )

    return parse_summary_bullets(raw_output)


def generate_meeting_title(summary_bullets: list[str]) -> str:

    llm = create_summary_llm()

    if isinstance(summary_bullets, list):
        summary_text = "\n".join(f"- {bullet}" for bullet in summary_bullets)
    else:
        summary_text = str(summary_bullets)

    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Generate a concise meeting title.

Rules:
- Maximum 8 words.
- Mention the project/topic if possible.
- Do not invent information.
- Return ONLY the title.
""",
            ),
            ("human", "{text}"),
        ]
    )

    title_chain = (
        RunnablePassthrough()
        | RunnableLambda(lambda x: {"text": x})
        | title_prompt
        | llm
        | StrOutputParser()
    )

    title = _invoke_llm_chain(title_chain, summary_text, "meeting title").strip()

    return title if title else "Untitled Meeting"
