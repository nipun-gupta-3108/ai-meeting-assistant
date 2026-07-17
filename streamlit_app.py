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

# Exact fallback strings produced by core.transcript_insights when a section
# has nothing to report. Only used here to render a quiet empty state instead
# of plain body text — this is a display choice, not a re-implementation of
# any backend logic.
EMPTY_STATE_TEXT = {
    "action_items": "No action items found.",
    "key_decisions": "No key decisions found.",
    "open_questions": "No open questions found.",
}

SECTION_GAP = '<div style="height:2.25rem"></div>'


def save_uploaded_file(uploaded_file) -> str:
    extension = Path(uploaded_file.name).suffix
    filename = f"{uuid.uuid4().hex}{extension}"
    file_path = UPLOAD_DIR / filename
    file_path.write_bytes(uploaded_file.getbuffer())
    return str(file_path)


def build_text_export(result: dict) -> str:
    return "\n\n".join(
        [
            f"Meeting Title\n{result['title']}",
            f"Summary\n{result['summary']}",
            f"Action Items\n{result['action_items']}",
            f"Key Decisions\n{result['key_decisions']}",
            f"Open Questions\n{result['open_questions']}",
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
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_styles():
    css = STYLE_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_insight_section(title: str, content: str, empty_key: str):
    """Render one insights section, showing a quiet empty state when the
    section content matches the parser's known "nothing found" fallback."""
    st.markdown(f'<p class="section-heading">{title}</p>', unsafe_allow_html=True)
    if content.strip() == EMPTY_STATE_TEXT[empty_key]:
        st.markdown(
            f'<p class="empty-state">{html.escape(content)}</p>', unsafe_allow_html=True
        )
    else:
        st.markdown(content)


def render_landing():
    st.markdown(f'<div class="masthead">{APP_NAME}</div>', unsafe_allow_html=True)

    _, center, _ = st.columns([1, 3, 1])
    with center:
        st.markdown(
            '<p class="landing-title">What meeting should we go through</p>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<p class="landing-sub">Paste a YouTube link or upload a recording</p>',
            unsafe_allow_html=True,
        )

        if st.session_state.error_message:
            st.error(st.session_state.error_message)

        input_mode = st.radio(
            "Source",
            ["YouTube URL", "Upload file"],
            horizontal=True,
            label_visibility="collapsed",
        )

        source = ""
        uploaded_file = None
        if input_mode == "YouTube URL":
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

        lang_col, button_col = st.columns([2, 1])
        with lang_col:
            language_label = st.selectbox(
                "Language",
                ["English", "Hinglish / Hindi"],
                label_visibility="collapsed",
            )
        with button_col:
            run_clicked = st.button(
                "Run analysis", type="primary", use_container_width=True
            )

        if run_clicked:
            st.session_state.error_message = None

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
    st.markdown(result["summary"])
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
