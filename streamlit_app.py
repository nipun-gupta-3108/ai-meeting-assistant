import html
import logging
import uuid
from datetime import datetime, timezone
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.logging_config import configure_logging

configure_logging()

from core.meeting_repository import (
    get_meeting,
    initialize_database,
    list_meetings,
    save_meeting,
)
from core.pipeline import run_meeting_assistant_pipeline
from core.transcript_qa import (
    ask_transcript_question,
    ensure_rag_chain,
    format_sources_line,
)
from core.transcript_vector_store import delete_collection, cleanup_stale_collections
from utils.audio_preparation import (
    DOWNLOAD_DIR,
    cleanup_stale_temp_files,
)

logger = logging.getLogger(__name__)

APP_NAME = "AI Meeting Assistant"

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

STYLE_PATH = Path(__file__).parent / "assets" / "style.css"

# Empty-state copy shown when a section's list comes back empty (either the
# meeting genuinely had nothing to report, or JSON parsing fell back to an
# empty list). Display-only — matches the wording the LLM used to produce
# directly, just no longer relies on exact string matching against it.
EMPTY_STATE_MESSAGES = {
    "action_items": "No action items found.",
    "key_decisions": "No key decisions found.",
    "open_questions": "No open questions found.",
}

SECTION_GAP = '<div style="height:2.25rem"></div>'

# Maps the internal language code stored in the meetings table (set in
# render_landing()'s pending_language) to the same user-facing labels
# already used elsewhere in this file (e.g. st.session_state.language_label).
LANGUAGE_DISPLAY_LABELS = {
    "english": "English",
    "hinglish": "Hinglish / Hindi",
}

# meeting_repository stores created_at via SQLite's datetime('now'), which
# is UTC, formatted as "YYYY-MM-DD HH:MM:SS". This is the matching parse
# format used to convert it for display — see format_history_timestamp().
_CREATED_AT_STORAGE_FORMAT = "%Y-%m-%d %H:%M:%S"
_CREATED_AT_DISPLAY_FORMAT = "%b %d, %Y %I:%M %p"


def format_summary_bullets(bullets: list) -> str:
    """Render summary bullets (a list of strings) as markdown, matching the
    bullet-point format the LLM used to return directly."""
    return "\n".join(f"- {bullet}" for bullet in bullets)


def format_insight_items(items: list) -> str:
    """Render one insights section's items as numbered markdown.

    Action items arrive as dicts ({task, owner, deadline}); key decisions
    and open questions arrive as plain strings. Output matches the numbered
    list format the LLM used to return directly.
    """
    lines = []
    for i, item in enumerate(items, start=1):
        if isinstance(item, dict):
            task = item.get("task", "")
            owner = item.get("owner", "Not specified")
            deadline = item.get("deadline", "Not specified")
            lines.append(f"{i}. **{task}** — Owner: {owner}, Deadline: {deadline}")
        else:
            lines.append(f"{i}. {item}")
    return "\n".join(lines)


def format_action_items_for_export(items: list) -> str:
    if not items:
        return EMPTY_STATE_MESSAGES["action_items"]
    lines = [
        f"{i}. {item.get('task', '')} "
        f"(Owner: {item.get('owner', 'Not specified')}, "
        f"Deadline: {item.get('deadline', 'Not specified')})"
        for i, item in enumerate(items, start=1)
    ]
    return "\n".join(lines)


def format_string_list_for_export(items: list, empty_key: str) -> str:
    if not items:
        return EMPTY_STATE_MESSAGES[empty_key]
    return "\n".join(f"{i}. {item}" for i, item in enumerate(items, start=1))


def format_history_timestamp(raw_created_at: str) -> str:
    """Convert a stored UTC created_at string to a local-time display string.

    meeting_repository writes created_at using SQLite's datetime('now'),
    which is naive UTC text — never local time and never timezone-aware.
    This parses it as UTC, converts to the machine's local timezone via
    stdlib datetime.astimezone(), and formats it for display. The database
    value itself is never modified.

    Falls back to returning the raw string as-is if it doesn't match the
    expected format, rather than raising and breaking the whole history list
    over one malformed row.
    """
    if not raw_created_at:
        return "Unknown time"

    try:
        naive_utc = datetime.strptime(raw_created_at, _CREATED_AT_STORAGE_FORMAT)
        aware_utc = naive_utc.replace(tzinfo=timezone.utc)
        local_dt = aware_utc.astimezone()  # no tz arg = convert to system local tz
        return local_dt.strftime(_CREATED_AT_DISPLAY_FORMAT)
    except (ValueError, TypeError):
        return raw_created_at


def save_uploaded_file(uploaded_file) -> str:
    extension = Path(uploaded_file.name).suffix
    filename = f"{uuid.uuid4().hex}{extension}"
    file_path = UPLOAD_DIR / filename
    file_path.write_bytes(uploaded_file.getbuffer())
    return str(file_path)


def build_text_export(result: dict) -> str:
    summary_text = (
        format_summary_bullets(result["summary"])
        if result["summary"]
        else "No summary available."
    )
    return "\n\n".join(
        [
            f"Meeting Title\n{result['title']}",
            f"Summary\n{summary_text}",
            f"Action Items\n{format_action_items_for_export(result['action_items'])}",
            f"Key Decisions\n{format_string_list_for_export(result['key_decisions'], 'key_decisions')}",
            f"Open Questions\n{format_string_list_for_export(result['open_questions'], 'open_questions')}",
            f"Transcript\n{result['transcript']}",
        ]
    )


def initialize_state():
    defaults = {
        "result": None,
        "chat_history": [],
        "language_label": "English",
        "processing": False,
        "pending_source": None,
        "pending_language": "english",
        "error_message": None,
        "input_mode": "YouTube URL",
        "uploaded_temp_path": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_styles():
    css = STYLE_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_insight_section(title: str, items: list, empty_key: str):
    """Render one insights section, showing a quiet empty state when there
    are no items to display."""
    st.markdown(f'<p class="section-heading">{title}</p>', unsafe_allow_html=True)
    if not items:
        message = EMPTY_STATE_MESSAGES[empty_key]
        st.markdown(
            f'<p class="empty-state">{html.escape(message)}</p>', unsafe_allow_html=True
        )
    else:
        st.markdown(format_insight_items(items))


def _open_historical_meeting(meeting_id: int) -> None:
    """Load one saved meeting from history into the workspace.

    Only ever populates st.session_state.result with what get_meeting()
    returns from SQLite — this never calls run_meeting_assistant_pipeline()
    and never touches Chroma, so no vector store is built just from
    opening a meeting. The loaded dict has no "rag_chain" or
    "collection_name" key at this point (get_meeting() never returns
    those — see meeting_repository's module docstring), and none is
    fabricated here.

    Both keys are attached lazily, on demand, the first time the user asks
    a question — see core.transcript_qa.ensure_rag_chain(), called from
    render_chat(). "is_historical" is a UI-only marker added at this
    layer; it is never written back to SQLite.
    """
    try:
        meeting = get_meeting(meeting_id)
    except Exception:
        logger.exception("Failed to load historical meeting %s.", meeting_id)
        st.session_state.error_message = (
            "Could not load that meeting right now. Please try again."
        )
        return

    if meeting is None:
        st.session_state.error_message = "That meeting could not be found."
        return

    meeting["is_historical"] = True

    st.session_state.result = meeting
    st.session_state.language_label = LANGUAGE_DISPLAY_LABELS.get(
        meeting.get("language"), meeting.get("language") or "Unknown language"
    )
    st.session_state.chat_history = []
    st.session_state.error_message = None


def render_history_item(meeting: dict):
    """Render one row of lightweight meeting metadata.

    Only reads the fields list_meetings() actually returns (id, title,
    language, word_count, created_at) — never transcript or insight
    fields, which list_meetings() intentionally omits.

    Title text is untrusted (it's LLM-generated, then user-persisted), so
    it's passed to st.markdown/st.caption without unsafe_allow_html: both
    render markdown text with HTML escaped by default, so a title
    containing "<script>" or similar is displayed as literal text rather
    than interpreted.
    """
    title = meeting.get("title") or "Untitled meeting"
    language_label = LANGUAGE_DISPLAY_LABELS.get(
        meeting.get("language"), meeting.get("language") or "Unknown language"
    )
    word_count = meeting.get("word_count", 0)
    created_display = format_history_timestamp(meeting.get("created_at"))

    meeting_id = meeting.get("id")

    with st.container(border=True):
        st.markdown(f"**{title}**")
        st.caption(f"{language_label} • {word_count} words • {created_display}")
        if st.button(
            "Open", key=f"open_meeting_{meeting_id}", use_container_width=True
        ):
            _open_historical_meeting(meeting_id)
            st.rerun()


def render_meeting_history():
    """Render the 'Meeting history' section on the landing screen.

    Uses core.meeting_repository.list_meetings() only — never queries
    sqlite3 directly — and never loads a full transcript just to show this
    list. A repository read failure is logged in full and degrades to a
    short inline message so the "Analyze a meeting" workflow above it stays
    fully usable either way.
    """
    st.markdown(SECTION_GAP, unsafe_allow_html=True)
    st.markdown(
        '<p class="section-heading">Meeting history</p>', unsafe_allow_html=True
    )

    try:
        meetings = list_meetings()
    except Exception:
        logger.exception("Failed to load meeting history.")
        st.markdown(
            '<p class="empty-state">Could not load meeting history right now.</p>',
            unsafe_allow_html=True,
        )
        return

    if not meetings:
        st.markdown(
            '<p class="empty-state">No meetings analyzed yet.</p>',
            unsafe_allow_html=True,
        )
        return

    # list_meetings() already orders newest-first (created_at DESC, id
    # DESC); rendered in that same order without any re-sorting here.
    for meeting in meetings:
        render_history_item(meeting)


def render_landing():
    st.markdown(
        '<div class="brand-row">'
        '<svg width="20" height="20" viewBox="0 0 24 24" fill="none" stroke="#f8fafc" '
        'stroke-width="1.6" stroke-linecap="round" stroke-linejoin="round">'
        '<rect x="9" y="2" width="6" height="11" rx="3"></rect>'
        '<path d="M5 10v1a7 7 0 0 0 14 0v-1"></path>'
        '<line x1="12" y1="18" x2="12" y2="22"></line>'
        '<line x1="8" y1="22" x2="16" y2="22"></line>'
        "</svg>"
        f'<span class="brand-name">{APP_NAME}</span>'
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="landing-headline">What meeting should we go through?</p>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<p class="landing-sub">Turn recordings into transcripts, summaries, '
        "action items, and searchable Q&amp;A.</p>",
        unsafe_allow_html=True,
    )

    if st.session_state.error_message:
        st.error(st.session_state.error_message)

    with st.container(border=True):
        toggle_col_a, toggle_col_b = st.columns(2)
        with toggle_col_a:
            if st.button(
                "YouTube URL",
                type=(
                    "primary"
                    if st.session_state.input_mode == "YouTube URL"
                    else "secondary"
                ),
                use_container_width=True,
                key="select_mode_url",
            ):
                st.session_state.input_mode = "YouTube URL"
        with toggle_col_b:
            if st.button(
                "Upload file",
                type=(
                    "primary"
                    if st.session_state.input_mode == "Upload file"
                    else "secondary"
                ),
                use_container_width=True,
                key="select_mode_upload",
            ):
                st.session_state.input_mode = "Upload file"

        source = ""
        uploaded_file = None
        if st.session_state.input_mode == "YouTube URL":
            source = st.text_input(
                "YouTube URL",
                placeholder="https://www.youtube.com/watch?v=...",
                label_visibility="collapsed",
            )
        else:
            uploaded_file = st.file_uploader(
                "Upload audio or video",
                type=["mp3", "mp4", "wav", "m4a", "webm", "mov", "aac"],
                label_visibility="collapsed",
            )

        lang_col, _spacer_col = st.columns(2)
        with lang_col:
            language_label = st.selectbox(
                "Language",
                ["English", "Hinglish / Hindi"],
                label_visibility="collapsed",
            )

        run_clicked = st.button(
            "Analyze meeting", type="primary", use_container_width=True
        )

    st.markdown(
        '<p class="landing-footnote">Supports YouTube links, MP3, MP4, WAV and M4A</p>',
        unsafe_allow_html=True,
    )

    render_meeting_history()

    if run_clicked:
        st.session_state.error_message = None
        input_mode = st.session_state.input_mode

        if input_mode == "Upload file":
            if uploaded_file is None:
                st.warning("Upload an audio or video file before running analysis.")
                return
            resolved_source = save_uploaded_file(uploaded_file)
            # Track this as our own temp artifact so it can be cleaned up
            # once the pipeline is done with it — see _cleanup_uploaded_temp_file.
            st.session_state.uploaded_temp_path = resolved_source
        elif not source.strip():
            st.warning("Enter a YouTube URL before running analysis.")
            return
        else:
            resolved_source = source.strip()
            st.session_state.uploaded_temp_path = None

        st.session_state.pending_source = resolved_source
        st.session_state.pending_language = (
            "hinglish" if language_label == "Hinglish / Hindi" else "english"
        )
        st.session_state.language_label = language_label
        st.session_state.chat_history = []
        st.session_state.processing = True
        st.rerun()


def _cleanup_uploaded_temp_file():
    """Remove the app's own temp copy of an uploaded file once the
    pipeline is done with it (success or failure). Only ever targets a
    path this app created via save_uploaded_file — never a YouTube URL,
    and never a path the user typed directly."""
    temp_path = st.session_state.get("uploaded_temp_path")
    if not temp_path:
        return
    try:
        Path(temp_path).unlink(missing_ok=True)
        logger.debug("Removed uploaded temp file: %s", temp_path)
    except OSError as exc:
        logger.warning("Could not remove uploaded temp file %s: %s", temp_path, exc)
    st.session_state.uploaded_temp_path = None


def render_processing():
    _, center, _ = st.columns([1, 3, 1])
    with center:
        with st.spinner(
            "Processing media, transcribing audio, and building meeting intelligence..."
        ):
            try:
                result = run_meeting_assistant_pipeline(
                    st.session_state.pending_source,
                    st.session_state.pending_language,
                )
            except Exception as exc:
                # Full stack trace goes to the log; the user only sees a
                # short, actionable message in the UI.
                logger.exception(
                    "Meeting analysis pipeline failed for source=%s language=%s",
                    st.session_state.pending_source,
                    st.session_state.pending_language,
                )
                st.session_state.processing = False
                st.session_state.error_message = (
                    f"Analysis failed: {exc}. Please check your input or "
                    "API configuration and try again."
                )
                _cleanup_uploaded_temp_file()
                st.rerun()
                return

        _cleanup_uploaded_temp_file()

        # Persist immediately after a successful pipeline run, and only
        # here. render_processing() runs exactly once per analysis: it is
        # only entered while st.session_state.processing is True, and that
        # flag is flipped to False a few lines below, before st.rerun().
        # The next rerun sees st.session_state.result already set and takes
        # the render_workspace() branch in main() instead — it never
        # re-enters render_processing() (and therefore never re-runs this
        # save) for the same pipeline result. A persistence failure here
        # must not discard an otherwise successful analysis, so it's
        # caught, logged, and surfaced as a null meeting_id rather than
        # raised.
        try:
            meeting_id = save_meeting(result, st.session_state.pending_language)
            result["meeting_id"] = meeting_id
        except Exception:
            logger.exception("Failed to save meeting to history database.")
            result["meeting_id"] = None

        st.session_state.result = result
        st.session_state.processing = False
        st.rerun()


def render_chat(result: dict):
    """Render the Q&A panel for a meeting.

    Works identically for freshly processed (live) meetings and meetings
    reopened from history: neither this function nor its caller branches
    on result.get("is_historical"). The only difference between the two
    cases is invisible from here — whether result["rag_chain"] was
    already set by core/pipeline.py, or gets attached lazily by
    ensure_rag_chain() below the first time a question is asked.
    """
    st.markdown(
        '<p class="section-heading">Ask about this meeting</p>', unsafe_allow_html=True
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
            # Old chat-history entries (from before sources were tracked)
            # simply have no "sources" key — .get() renders them exactly
            # as before, with no citation line.
            sources_line = format_sources_line(message.get("sources"))
            if sources_line:
                st.caption(sources_line)

    question = st.chat_input("Ask anything about this meeting...")
    if question:
        # Snapshot the turns that came BEFORE this question. This is what
        # gets sent to the RAG chain as conversational context — appending
        # the new user turn below must not leak into its own
        # contextualization input (a question can't be its own history).
        history_for_chain = list(st.session_state.chat_history)

        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching transcript..."):
                try:
                    # No-op for live meetings (rag_chain already set by
                    # core/pipeline.py). For a historical meeting, this is
                    # where the vector store / retriever / chain actually
                    # get built — on this, the first question, and never
                    # again for this meeting. If building fails here,
                    # result["rag_chain"] stays unset, so the next
                    # question will simply retry rather than being stuck
                    # in a broken state.
                    ensure_rag_chain(result)

                    qa_result = ask_transcript_question(
                        result["rag_chain"],
                        question,
                        chat_history=history_for_chain,
                    )
                    answer = qa_result["answer"]
                    sources = qa_result["sources"]
                except Exception as exc:
                    logger.exception("Q&A failed for question: %s", question)
                    answer = f"Sorry, I couldn't answer that: {exc}"
                    sources = []
            st.markdown(answer)
            sources_line = format_sources_line(sources)
            if sources_line:
                st.caption(sources_line)

        # Sources are stored alongside the answer so that on the next
        # Streamlit rerun (e.g. from a later chat_input submission),
        # re-rendering this message from chat_history above does not need
        # to invoke retrieval again — it just re-displays the same list.
        st.session_state.chat_history.append(
            {"role": "assistant", "content": answer, "sources": sources}
        )


def render_export(result: dict):
    st.markdown('<p class="section-heading">Export</p>', unsafe_allow_html=True)
    col1, col2 = st.columns(2)
    with col1:
        st.download_button(
            "Download report",
            data=build_text_export(result),
            file_name="meeting_analysis.txt",
            mime="text/plain",
            use_container_width=True,
        )
    with col2:
        st.download_button(
            "Download transcript",
            data=result["transcript"],
            file_name="transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )


def render_workspace(result: dict):
    is_historical = result.get("is_historical", False)
    back_label = "← Back to history" if is_historical else "← New meeting"

    if st.button(back_label):
        # The RAG chain for this meeting is about to go out of scope for
        # good — clean up its Chroma collection now rather than waiting
        # for the next process's startup sweep. A historical meeting only
        # has a "collection_name" if a question was actually asked during
        # this session (see ensure_rag_chain in core/transcript_qa.py); if
        # none was asked, this is simply a no-op rather than a call on a
        # nonexistent collection.
        collection_name = result.get("collection_name")
        if collection_name:
            delete_collection(collection_name)
        st.session_state.result = None
        st.session_state.chat_history = []
        st.session_state.error_message = None
        st.rerun()

    word_count = len(result["transcript"].split())

    st.markdown(
        f'<h1 class="workspace-h1">{html.escape(result["title"])}</h1>',
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p class="workspace-meta">{html.escape(st.session_state.language_label)} • {word_count} words</p>',
        unsafe_allow_html=True,
    )

    st.markdown('<p class="section-heading">Summary</p>', unsafe_allow_html=True)
    if result["summary"]:
        st.markdown(format_summary_bullets(result["summary"]))
    else:
        st.markdown(
            '<p class="empty-state">No summary available.</p>', unsafe_allow_html=True
        )
    st.markdown(SECTION_GAP, unsafe_allow_html=True)

    render_insight_section("Action items", result["action_items"], "action_items")
    st.markdown(SECTION_GAP, unsafe_allow_html=True)
    render_insight_section("Key decisions", result["key_decisions"], "key_decisions")
    st.markdown(SECTION_GAP, unsafe_allow_html=True)
    render_insight_section("Open questions", result["open_questions"], "open_questions")
    st.markdown(SECTION_GAP, unsafe_allow_html=True)

    with st.expander("Full transcript"):
        st.markdown(
            f'<div class="transcript-box">{html.escape(result["transcript"])}</div>',
            unsafe_allow_html=True,
        )

    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
    # Live and historical meetings now share the exact same Q&A code path.
    # A historical meeting simply starts this call with no "rag_chain" yet
    # — render_chat/ensure_rag_chain handle that transparently.
    render_chat(result)

    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
    render_export(result)


@st.cache_resource
def _initialize_database_once():
    """Create the meetings table once per Streamlit process.

    initialize_database() itself is idempotent (CREATE TABLE IF NOT
    EXISTS), but wrapping it in st.cache_resource avoids opening a
    throwaway sqlite connection on every rerun.
    """
    initialize_database()
    return True


@st.cache_resource
def _cleanup_stale_artifacts_once():
    """Run stale-artifact cleanup once per Streamlit process.

    Chroma collections and filesystem temp artifacts use conservative
    age-based cleanup so active sessions are not affected.

    Only app-owned directories are swept:
    - downloads/ contains files created by the YouTube processing path.
    - uploads/ contains UUID-named copies created by save_uploaded_file().

    Never point cleanup_stale_temp_files() at arbitrary user directories.
    """

    cleanup_stale_collections()

    cleanup_stale_temp_files(DOWNLOAD_DIR)
    cleanup_stale_temp_files(str(UPLOAD_DIR))

    return True


def main():
    st.set_page_config(page_title=APP_NAME, page_icon="🎙️", layout="centered")
    _initialize_database_once()
    _cleanup_stale_artifacts_once()
    initialize_state()
    render_styles()

    if st.session_state.result:
        render_workspace(st.session_state.result)
    elif st.session_state.processing:
        render_processing()
    else:
        render_landing()


if __name__ == "__main__":
    main()
