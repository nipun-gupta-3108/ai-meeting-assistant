import sys

# Streamlit Community Cloud ships an old system SQLite (<3.35), which
# ChromaDB requires at a minimum. This swaps in pysqlite3-binary's modern
# build. Must run before chromadb (or anything importing it) is loaded.
try:
    __import__("pysqlite3")
    sys.modules["sqlite3"] = sys.modules["pysqlite3"]
except ImportError:
    pass


from pathlib import Path

import streamlit as st
from dotenv import load_dotenv

load_dotenv()

from core.transcript_qa import ask_transcript_question
from core.pipeline import run_meeting_assistant_pipeline

APP_NAME = "AI Meeting Assistant"

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)

STYLE_PATH = Path(__file__).parent / "assets" / "style.css"


def save_uploaded_file(uploaded_file) -> str:
    file_path = UPLOAD_DIR / uploaded_file.name
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
        "last_source": "",
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def render_styles():
    """Load the external stylesheet and inject it into the page."""
    css = STYLE_PATH.read_text(encoding="utf-8")
    st.markdown(f"<style>{css}</style>", unsafe_allow_html=True)


def render_metric_row(items: list[tuple[str, str]]):
    """Render a row of metric cards, e.g. [("Title", "Standup"), ...]."""
    cards = "".join(
        f'<div class="metric-card"><span>{label}</span><strong>{value}</strong></div>'
        for label, value in items
    )
    st.markdown(f'<div class="metric-row">{cards}</div>', unsafe_allow_html=True)


def render_section_card(title: str, body: str):
    """Render a single titled info card used on the empty-state screen."""
    st.markdown(
        f'<div class="section-card"><h3>{title}</h3><p>{body}</p></div>',
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.title(APP_NAME)
    st.sidebar.caption("Transcribe, summarize, extract decisions, and chat with a meeting.")

    input_mode = st.sidebar.radio("Input type", ["YouTube URL", "Upload file"])
    language_label = st.sidebar.selectbox("Audio language", ["English", "Hinglish / Hindi"])
    language = "hinglish" if language_label == "Hinglish / Hindi" else "english"

    source = ""
    uploaded_file = None

    if input_mode == "YouTube URL":
        source = st.sidebar.text_input("YouTube URL", placeholder="https://www.youtube.com/watch?v=...")
    else:
        uploaded_file = st.sidebar.file_uploader(
            "Upload audio or video",
            type=["mp3", "mp4", "wav", "m4a", "webm", "mov", "aac"],
        )

    run_clicked = st.sidebar.button("Run analysis", type="primary", use_container_width=True)

    st.sidebar.divider()
    st.sidebar.caption("Required: FFmpeg, Groq API key, and Whisper dependencies.")

    return input_mode, source, uploaded_file, language, run_clicked


def render_hero():
    st.markdown(
        f"""
        <div class="hero">
            <div class="eyebrow">Meeting Intelligence Workspace</div>
            <h1>{APP_NAME}</h1>
            <p>Turn YouTube videos or meeting recordings into transcripts, summaries, action items, decisions, and searchable Q&A.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state():
    render_metric_row(
        [
            ("Input", "URL or file"),
            ("Transcription", "Whisper"),
            ("Analysis", "Groq"),
            ("Search", "ChromaDB"),
        ]
    )

    col1, col2 = st.columns(2)
    with col1:
        render_section_card(
            "What it does",
            "Processes a video or audio source, creates a transcript, and extracts "
            "the meeting details people usually lose after the call.",
        )
    with col2:
        render_section_card(
            "How to start",
            "Choose an input type in the sidebar, select the language, then run "
            "the analysis. Results will appear here.",
        )


def render_result_tabs(result: dict):
    export_text = build_text_export(result)

    render_metric_row(
        [
            ("Title", result["title"]),
            ("Transcript", f"{len(result['transcript'].split())} words"),
            ("Insights", "3 sections"),
            ("Q&A", "Ready"),
        ]
    )

    summary_tab, insights_tab, transcript_tab, chat_tab, export_tab = st.tabs(
        ["Summary", "Insights", "Transcript", "Chat", "Export"]
    )

    with summary_tab:
        with st.container(border=True):
            st.subheader(result["title"])
            st.markdown(result["summary"])

    with insights_tab:
        col1, col2 = st.columns(2)
        with col1:
            with st.container(border=True):
                st.markdown("#### Action Items")
                st.markdown(result["action_items"])
        with col2:
            with st.container(border=True):
                st.markdown("#### Key Decisions")
                st.markdown(result["key_decisions"])
        with st.container(border=True):
            st.markdown("#### Open Questions")
            st.markdown(result["open_questions"])

    with transcript_tab:
        st.text_area("Full transcript", result["transcript"], height=420)

    with chat_tab:
        for message in st.session_state.chat_history:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

        question = st.chat_input("Ask a question about the transcript")
        if question:
            st.session_state.chat_history.append({"role": "user", "content": question})
            with st.chat_message("user"):
                st.markdown(question)

            with st.chat_message("assistant"):
                with st.spinner("Searching transcript..."):
                    answer = ask_transcript_question(result["rag_chain"], question)
                st.markdown(answer)
            st.session_state.chat_history.append({"role": "assistant", "content": answer})

    with export_tab:
        st.download_button(
            "Download TXT report",
            data=export_text,
            file_name="meeting_analysis.txt",
            mime="text/plain",
            use_container_width=True,
        )
        st.download_button(
            "Download transcript",
            data=result["transcript"],
            file_name="transcript.txt",
            mime="text/plain",
            use_container_width=True,
        )


def run_analysis(input_mode: str, source: str, uploaded_file, language: str):
    if input_mode == "Upload file":
        if uploaded_file is None:
            st.warning("Upload an audio or video file before running analysis.")
            return
        source = save_uploaded_file(uploaded_file)
    elif not source.strip():
        st.warning("Enter a YouTube URL before running analysis.")
        return

    st.session_state.chat_history = []
    try:
        with st.spinner("Processing media, transcribing audio, and building meeting intelligence..."):
            result = run_meeting_assistant_pipeline(source.strip(), language)
    except Exception as exc:
        st.error(
            f"Analysis failed: {exc}\n\nPlease check your input or API configuration and try again."
        )
        return

    st.session_state.result = result
    st.session_state.last_source = source

    st.success("Analysis complete.")


def main():
    st.set_page_config(
        page_title=APP_NAME,
        page_icon="🎙️",
        layout="wide",
    )
    initialize_state()
    render_styles()

    input_mode, source, uploaded_file, language, run_clicked = render_sidebar()
    render_hero()

    if run_clicked:
        run_analysis(input_mode, source, uploaded_file, language)

    if st.session_state.result:
        render_result_tabs(st.session_state.result)
    else:
        render_empty_state()


if __name__ == "__main__":
    main()
