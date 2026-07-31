from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from core.llm_client import create_llm
import json
import re

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

{
  "summary": [
    "Bullet 1",
    "Bullet 2",
    "Bullet 3"
  ]
}

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


def _truncate_words(text: str, limit: int):
    words = text.split()
    return " ".join(words[:limit])


def parse_summary_bullets(raw_text: str):

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


def summarize_transcript(transcript: str):

    llm = create_llm()

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

    print("\n")
    print("=" * 80)
    print("RAW SUMMARY OUTPUT")
    print("=" * 80)
    print(raw_output)
    print("=" * 80)
    print()

    return parse_summary_bullets(raw_output)


def generate_meeting_title(transcript: str):

    llm = create_llm()

    title_prompt = ChatPromptTemplate.from_messages(
        [
            (
                "system",
                """Generate a meeting title.

Rules:
- Maximum 8 words.
- Be specific.
- Mention project/product if possible.
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

    title = title_chain.invoke(transcript[:2000]).strip()

    return title if title else "Untitled Meeting"
