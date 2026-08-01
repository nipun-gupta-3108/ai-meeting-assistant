from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_text_splitters import RecursiveCharacterTextSplitter
from core.llm_client import create_gemini
import json

# Maximum number of items kept per section, enforced both in the prompts
# and again here in Python — the prompt limit alone can't be trusted since
# the model may not always comply.
MAX_ITEMS_PER_SECTION = 5

# Matches the chunk_size/overlap used by transcript_summary.py's map-reduce
# summarization step (see split_transcript_for_summary there). Using the
# same window size means both pipeline stages see the same-shaped chunks
# for a given transcript, which keeps their behavior easy to reason about
# together — there's no functional requirement that the two match, but
# there's no reason for them to drift either.
INSIGHTS_CHUNK_SIZE = 3000
INSIGHTS_CHUNK_OVERLAP = 200


def build_extraction_chain(system_prompt: str):
    """Build a simple (prompt | llm | parser) chain around one system
    prompt. Shared by both the per-chunk "map" chain and the "reduce"
    combine chain below — they differ only in which prompt is used.
    """
    llm = create_gemini(temperature=0)
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


# ---------------------------------------------------------------------------
# MAP step: extract structured insights from a single transcript chunk.
#
# Deliberately scoped to "this excerpt only" — a chunk has no visibility
# into the rest of the transcript, so it must not attempt cross-chunk
# concerns like global deduplication or final prioritization. Those are
# the REDUCE step's job, once every chunk's partial output is available
# together. Each chunk is still capped at MAX_ITEMS_PER_SECTION so the
# reduce step's input stays bounded even for very long meetings with many
# chunks.
# ---------------------------------------------------------------------------
MAP_INSIGHTS_SYSTEM_PROMPT = """You are an expert meeting analyst. You will be given \
ONE EXCERPT of a longer meeting transcript (not the full meeting). Extract structured \
insights that are explicitly present in this excerpt only.

Return ONLY a single valid JSON object. Do not include markdown code fences \
(no ```), explanations, headings, or any text outside the JSON object.

The JSON object must have exactly these three keys:

"action_items": an array of at most 5 objects, each with keys "task", "owner", "deadline".
- Only include items that are explicitly assigned to someone or explicitly agreed \
upon in this excerpt. Do not convert a suggestion, idea, or proposal into an action \
item unless the excerpt shows it was explicitly assigned or agreed to.
- "task": a short, objective description of the action item.
- "owner": the person responsible, exactly as stated in the excerpt. Use \
"Not specified" if no owner is explicitly named — never infer an owner from role, \
tone, or context.
- "deadline": the due date, exactly as stated in the excerpt. Use "Not specified" \
if no deadline is explicitly given — never infer a deadline.
- If there are no action items in this excerpt, return an empty array.

"key_decisions": an array of at most 5 short strings, each describing one decision \
that this excerpt explicitly frames as agreed or finalized.
- Do not include proposals, suggestions, or options that were only discussed or \
left open.
- If there are no decisions in this excerpt, return an empty array.

"open_questions": an array of at most 5 short strings, each describing one question, \
blocker, or follow-up that is raised in this excerpt and not resolved within it.
- Exclude rhetorical questions that aren't genuine open items. A question resolved \
later in the full meeting (outside this excerpt) may still be listed here — merging \
across excerpts and dropping questions resolved elsewhere is handled in a later step.
- If there are none, return an empty array.

Use objective, neutral language throughout; do not add sentiment or editorializing \
that isn't explicitly present in the excerpt.

Do not invent facts that are not present in this excerpt. Return only the JSON \
object, nothing else."""


# ---------------------------------------------------------------------------
# REDUCE step: merge every chunk's partial JSON output into one final result.
#
# This is where cross-chunk concerns that MAP intentionally skips actually
# get handled: collapsing the same action item / decision / question when
# different chunks phrased it differently, dropping open questions that a
# later chunk shows were answered, and applying the final "at most 5,
# highest priority first" cap across the whole meeting rather than per
# chunk.
# ---------------------------------------------------------------------------
REDUCE_INSIGHTS_SYSTEM_PROMPT = """You are an expert meeting analyst. You will be \
given several partial JSON extraction results, each produced independently from a \
different, non-overlapping excerpt of the SAME meeting transcript, in chronological \
order.

Your job is to merge them into ONE final structured result for the whole meeting.

Rules:
- Merge duplicate or overlapping items across excerpts into a single entry. Two \
items describing the same underlying task, decision, or question — even if worded \
differently in different excerpts — must be combined into one.
- When merging an action item that appears with different owner/deadline detail in \
different excerpts, keep the most complete version (prefer a named owner or explicit \
deadline over "Not specified") rather than dropping that detail. Never invent an \
owner or deadline that isn't present in any of the partial results.
- For open questions: if a later excerpt's content shows a question from an earlier \
excerpt was answered, exclude that question from the final result rather than \
carrying it forward as still open.
- Never invent facts that are not present in the partial results you were given.
- Use neutral, objective language.

Priority order when deciding which items make the final cut (at most 5 each):
1. Decisions
2. Action items (explicit deadline or urgency first)
3. Open questions/blockers

Return ONLY valid JSON. The output MUST EXACTLY follow this schema:

{{
  "action_items": [
    {{"task": "...", "owner": "...", "deadline": "..."}}
  ],
  "key_decisions": ["..."],
  "open_questions": ["..."]
}}

Rules:
- Each array holds at most 5 items.
- "owner" and "deadline" must be "Not specified" when not explicitly stated.
- Do NOT output markdown, explanations, or anything outside the JSON object.
- If a section has no items, return an empty array for it — never omit the key."""


# Built lazily on first use and cached, instead of at import time —
# building at import time would require GOOGLE_API_KEY before
# load_dotenv() runs in streamlit_app.py, causing a startup crash.
_MAP_CHAIN = None
_REDUCE_CHAIN = None

# Kept for backward compatibility with the pre-map-reduce single-shot
# chain used by extract_meeting_insights_from_summary() below.
_LEGACY_SUMMARY_CHAIN = None


def get_map_chain():
    global _MAP_CHAIN
    if _MAP_CHAIN is None:
        _MAP_CHAIN = build_extraction_chain(MAP_INSIGHTS_SYSTEM_PROMPT)
    return _MAP_CHAIN


def get_reduce_chain():
    global _REDUCE_CHAIN
    if _REDUCE_CHAIN is None:
        _REDUCE_CHAIN = build_extraction_chain(REDUCE_INSIGHTS_SYSTEM_PROMPT)
    return _REDUCE_CHAIN


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

    The prompts ask for JSON only, but models sometimes wrap the object in
    markdown code fences anyway. This strips fences and slices out the
    outermost {...} span so json.loads gets the best possible input.

    Unchanged from the previous single-shot implementation — reused as-is
    by both the map-reduce path and the legacy path below, per the
    "keep the existing JSON parser" requirement.
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

    Unchanged from the previous single-shot implementation. Used as the
    final parse step for both the new map-reduce path (on the reduce
    chain's output) and the legacy single-shot path below, so both paths
    are guaranteed to produce an identically-shaped result.
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


def split_transcript_for_insights(transcript: str) -> list:
    """Split a full transcript into chunks for the map step.

    Same splitter shape as transcript_summary.py's
    split_transcript_for_summary() (see INSIGHTS_CHUNK_SIZE/OVERLAP above),
    kept as its own function rather than importing that one directly —
    insights and summary are independent pipeline stages that happen to
    use the same chunking parameters today; they are not required to stay
    in lockstep, and importing across those modules would create a
    dependency between two stages that should be free to evolve
    independently.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=INSIGHTS_CHUNK_SIZE,
        chunk_overlap=INSIGHTS_CHUNK_OVERLAP,
    )
    return splitter.split_text(transcript)


def _format_partial_insights_for_reduce(partial_outputs: list) -> str:
    """Join every chunk's raw map-step output into one labeled block of
    text for the reduce chain.

    Each chunk's output is passed through as the raw text the map chain
    returned (not re-parsed/re-serialized here) — the reduce prompt is
    explicitly instructed to treat its input as a sequence of partial JSON
    extraction results, so it can do its own merging directly from that
    text. Numbering the excerpts also preserves chronological order,
    which the reduce prompt relies on when deciding whether a later
    excerpt resolves an earlier excerpt's open question.
    """
    sections = [
        f"--- Excerpt {i} extraction result ---\n{output}"
        for i, output in enumerate(partial_outputs, start=1)
    ]
    return "\n\n".join(sections)


def extract_meeting_insights_from_transcript(transcript: str) -> dict:
    """Extract action items, key decisions, and open questions directly
    from a full meeting transcript, using the same map-reduce shape as
    transcript_summary.summarize_transcript():

      1. Split the transcript into chunks (split_transcript_for_insights).
      2. MAP: run the extraction chain independently on each chunk.
      3. REDUCE: merge every chunk's partial JSON output into one final,
         deduplicated result via a second LLM call.
      4. Parse the reduce step's output with the existing, unchanged
         parse_insight_sections() so the returned shape is identical to
         what the previous single-shot implementation produced.

    This replaces sending an entire transcript to the model in one request
    (which runs into request-size/context limits on long meetings) with a
    bounded number of bounded-size requests, matching transcript_summary.py's
    approach to the same problem.

    The reduce step always runs, even for a single-chunk transcript — this
    mirrors transcript_summary.py's combine step, which also always runs
    regardless of chunk count, so behavior doesn't change shape at the
    chunk-count boundary.
    """
    if not transcript or not transcript.strip():
        return {key: list(value) for key, value in _EMPTY_RESULT.items()}

    chunks = split_transcript_for_insights(transcript)

    map_chain = get_map_chain()
    partial_outputs = [map_chain.invoke(chunk) for chunk in chunks]

    reduce_chain = get_reduce_chain()
    combined_input = _format_partial_insights_for_reduce(partial_outputs)
    raw_output = reduce_chain.invoke(combined_input)

    return parse_insight_sections(raw_output)


# ---------------------------------------------------------------------------
# Legacy single-shot path — DEPRECATED.
#
# Kept unchanged and fully functional for backward compatibility: any
# existing caller (e.g. core/pipeline.py, before it is updated to call
# extract_meeting_insights_from_transcript() instead) continues to work
# exactly as before. New code should call
# extract_meeting_insights_from_transcript() with the full transcript
# instead of this function with the summary.
# ---------------------------------------------------------------------------
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


def get_insights_chain():
    """Deprecated: backs only extract_meeting_insights_from_summary() below.
    New code should use get_map_chain()/get_reduce_chain() via
    extract_meeting_insights_from_transcript()."""
    global _LEGACY_SUMMARY_CHAIN
    if _LEGACY_SUMMARY_CHAIN is None:
        _LEGACY_SUMMARY_CHAIN = build_extraction_chain(INSIGHTS_SYSTEM_PROMPT)
    return _LEGACY_SUMMARY_CHAIN


def extract_meeting_insights_from_summary(summary) -> dict:
    """DEPRECATED — retained only for backward compatibility.

    Extracts action items, key decisions, and open questions from the
    already-condensed meeting summary in a single request. This is the
    pre-map-reduce behavior, unchanged.

    New code should call extract_meeting_insights_from_transcript(transcript)
    instead, which extracts directly from the full transcript via map-reduce
    and does not depend on transcript_summary.py's output. Note that
    core/pipeline.py currently calls this function — switching it over to
    the transcript-based function is a separate, explicit change to that
    call site, not made as part of this file.
    """
    if isinstance(summary, list):
        text = "\n".join(f"- {item}" for item in summary)
    else:
        text = str(summary)

    raw_output = get_insights_chain().invoke(text)

    return parse_insight_sections(raw_output)
