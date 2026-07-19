from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.llm_client import create_llm
import json

# Enforced both in the combine prompt and again here in Python, since the
# prompt limit alone can't be trusted to hold on every response.
MAX_SUMMARY_BULLETS = 5
MAX_BULLET_WORDS = 25

COMBINE_SYSTEM_PROMPT = """You are an expert meeting summarizer. You will be given \
several partial summaries of segments of one meeting.

Combine them into a single meeting summary of at most 5 bullet points.

Rules:
- At most 5 bullets total.
- Each bullet must be under 25 words.
- Do not repeat similar points; merge overlapping observations into a single bullet.
- Do not invent facts that are not present in the partial summaries.

Return ONLY a single valid JSON object with exactly one key, "summary", whose value \
is an array of bullet strings. Do not include markdown code fences (no ```), \
explanations, or any text outside the JSON object."""


def _extract_json_object(raw_text: str) -> str:
    """Best-effort extraction of a JSON object from the model's raw response.

    The prompt asks for JSON only, but models sometimes wrap the object in
    markdown code fences anyway. This strips fences and slices out the
    outermost {...} span so json.loads gets the best possible input.

    Kept local to this module rather than shared with core/transcript_insights.py,
    since both modules parse a different JSON shape and are meant to stay
    independent.
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


def _truncate_words(text: str, max_words: int) -> str:
    words = text.split()
    if len(words) <= max_words:
        return text
    return " ".join(words[:max_words])


def parse_summary_bullets(raw_text: str) -> list:
    """Parse the model's JSON output into a list of summary bullets.

    Defensive by design: never raises. If the output isn't valid JSON, the
    raw text is kept as a single bullet rather than losing the model's
    output entirely. Always capped at MAX_SUMMARY_BULLETS bullets of at
    most MAX_BULLET_WORDS words each.
    """
    if not raw_text:
        return []

    try:
        data = json.loads(_extract_json_object(raw_text))
    except Exception:
        stripped = raw_text.strip()
        return [_truncate_words(stripped, MAX_BULLET_WORDS)] if stripped else []

    if not isinstance(data, dict):
        return []

    bullets = data.get("summary")
    if not isinstance(bullets, list):
        return []

    cleaned = []
    for bullet in bullets:
        text = str(bullet).strip()
        if text:
            cleaned.append(_truncate_words(text, MAX_BULLET_WORDS))

    return cleaned[:MAX_SUMMARY_BULLETS]


def split_transcript_for_summary(transcript: str) -> list:
    splitter = RecursiveCharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
    return splitter.split_text(transcript)


def summarize_transcript(transcript: str) -> list:
    llm = create_llm()
    map_prompt = ChatPromptTemplate.from_messages(
        [
            ("system", "Summarize this portion of a meeting transcript concisely."),
            ("human", "{text}"),
        ]
    )

    map_chain = map_prompt | llm | StrOutputParser()

    chunks = split_transcript_for_summary(transcript)

    chunk_summaries = [map_chain.invoke({"text": chunk}) for chunk in chunks]

    combined = "\n\n".join(chunk_summaries)

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

    raw_output = combined_chain.invoke(combined)
    return parse_summary_bullets(raw_output)


def generate_meeting_title(transcript: str) -> str:
    llm = create_llm()

    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                "Based on the meeting transcript, generate a short professional meeting title "
                "(max 8 words). Only return the title, nothing else.",
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

    title = title_chain.invoke(transcript[:2000]).strip()
    return title if title else "Untitled Meeting"
