import html
import traceback
import uuid
from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.pipeline import run_meeting_assistant_pipeline
from core.transcript_qa import ask_transcript_question

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

    # Centering is done via CSS (max-width + margin:auto on .landing-headline,
    # .landing-sub, .landing-footnote, and the bordered panel) rather than
    # st.columns([1, 3, 1]). The old column split capped the panel at
    # roughly 3/5 of the 760px block-container (~420-450px) no matter what
    # padding/border CSS was applied to the panel itself — CSS centering
    # removes that ceiling so the panel can reach its intended 640px width.
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
        # Source mode lives in session_state and is switched by two real
        # buttons (not st.radio + CSS), so it survives reruns correctly
        # and never depends on Streamlit's internal DOM structure.
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

        # Only the field for the CURRENT mode is rendered/read this run,
        # so a leftover value from the other mode can never be submitted.
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

    if run_clicked:
        st.session_state.error_message = None
        input_mode = st.session_state.input_mode

        if input_mode == "Upload file":
            if uploaded_file is None:
                st.warning("Upload an audio or video file before running analysis.")
                return
            resolved_source = save_uploaded_file(uploaded_file)
        elif not source.strip():
            st.warning("Enter a YouTube URL before running analysis.")
            return
        else:
            resolved_source = source.strip()

        st.session_state.pending_source = resolved_source
        st.session_state.pending_language = (
            "hinglish" if language_label == "Hinglish / Hindi" else "english"
        )
        st.session_state.language_label = language_label
        st.session_state.chat_history = []
        st.session_state.processing = True
        st.rerun()


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
                traceback.print_exc()
                st.session_state.processing = False
                st.session_state.error_message = (
                    f"Analysis failed: {exc}. Please check your input or "
                    "API configuration and try again."
                )
                st.rerun()
                return

        st.session_state.result = result
        st.session_state.processing = False
        st.rerun()


def render_chat(result: dict):
    st.markdown(
        '<p class="section-heading">Ask about this meeting</p>', unsafe_allow_html=True
    )

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    question = st.chat_input("Ask anything about this meeting...")
    if question:
        st.session_state.chat_history.append({"role": "user", "content": question})
        with st.chat_message("user"):
            st.markdown(question)

        with st.chat_message("assistant"):
            with st.spinner("Searching transcript..."):
                answer = ask_transcript_question(result["rag_chain"], question)
            st.markdown(answer)
        st.session_state.chat_history.append({"role": "assistant", "content": answer})


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
    if st.button("← New meeting"):
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
    render_chat(result)

    st.markdown('<hr class="section-divider" />', unsafe_allow_html=True)
    render_export(result)


def main():
    st.set_page_config(page_title=APP_NAME, page_icon="🎙️", layout="centered")
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
