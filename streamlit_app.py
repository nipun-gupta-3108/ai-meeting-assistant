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

from core.transcript_qa import ask_transcript_question
from meeting_assistant_cli import run_meeting_assistant_pipeline


load_dotenv()

UPLOAD_DIR = Path("uploads")
UPLOAD_DIR.mkdir(exist_ok=True)


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
    st.markdown(
        """
        <style>
        .stApp {
            background: #0f172a;
            color: #e5e7eb;
        }

        [data-testid="stSidebar"] {
            background: #111827;
            border-right: 1px solid rgba(148, 163, 184, 0.18);
        }

        .main .block-container {
            max-width: 1180px;
            padding-top: 2rem;
            padding-bottom: 3rem;
        }

        .hero {
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(30, 41, 59, 0.94));
            padding: 28px;
            border-radius: 8px;
            margin-bottom: 22px;
        }

        .eyebrow {
            color: #67e8f9;
            font-size: 0.78rem;
            font-weight: 700;
            letter-spacing: 0.16em;
            text-transform: uppercase;
            margin-bottom: 10px;
        }

        .hero h1 {
            color: #f8fafc;
            font-size: 3rem;
            line-height: 1.08;
            margin: 0 0 12px;
        }

        .hero p {
            color: #cbd5e1;
            font-size: 1.04rem;
            margin: 0;
            max-width: 760px;
        }

        .metric-row {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin-bottom: 20px;
        }

        .metric-card {
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: #1e293b;
            border-radius: 8px;
            padding: 16px;
        }

        .metric-card span {
            color: #94a3b8;
            display: block;
            font-size: 0.8rem;
            margin-bottom: 6px;
        }

        .metric-card strong {
            color: #f8fafc;
            font-size: 1.45rem;
        }

        .section-card {
            border: 1px solid rgba(148, 163, 184, 0.22);
            background: #1e293b;
            border-radius: 8px;
            padding: 18px;
            min-height: 180px;
        }

        .section-card h3 {
            color: #f8fafc;
            margin-top: 0;
        }

        div[data-testid="stMarkdownContainer"] p,
        div[data-testid="stMarkdownContainer"] li {
            color: #dbe4f0;
        }

        .stTabs [data-baseweb="tab-list"] {
            gap: 8px;
        }

        .stTabs [data-baseweb="tab"] {
            background: #1e293b;
            border-radius: 8px;
            color: #cbd5e1;
            border: 1px solid rgba(148, 163, 184, 0.18);
        }

        .stTabs [aria-selected="true"] {
            background: #0e7490;
            color: #ffffff;
        }

        @media (max-width: 760px) {
            .hero h1 {
                font-size: 2.15rem;
            }

            .metric-row {
                grid-template-columns: repeat(2, minmax(0, 1fr));
            }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def render_sidebar():
    st.sidebar.title("AI Video Assistant")
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
        """
        <div class="hero">
            <div class="eyebrow">Meeting Intelligence Workspace</div>
            <h1>AI Video Assistant</h1>
            <p>Turn YouTube videos or meeting recordings into transcripts, summaries, action items, decisions, and searchable Q&A.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_empty_state():
    st.markdown(
        """
        <div class="metric-row">
            <div class="metric-card"><span>Input</span><strong>URL or file</strong></div>
            <div class="metric-card"><span>Transcription</span><strong>Whisper</strong></div>
            <div class="metric-card"><span>Analysis</span><strong>Groq</strong></div>
            <div class="metric-card"><span>Search</span><strong>ChromaDB</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    col1, col2 = st.columns(2)
    with col1:
        st.markdown(
            """
            <div class="section-card">
                <h3>What it does</h3>
                <p>Processes a video or audio source, creates a transcript, and extracts the meeting details people usually lose after the call.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with col2:
        st.markdown(
            """
            <div class="section-card">
                <h3>How to start</h3>
                <p>Choose an input type in the sidebar, select the language, then run the analysis. Results will appear here.</p>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_result_tabs(result: dict):
    export_text = build_text_export(result)

    st.markdown(
        f"""
        <div class="metric-row">
            <div class="metric-card"><span>Title</span><strong>{result['title']}</strong></div>
            <div class="metric-card"><span>Transcript</span><strong>{len(result['transcript'].split())} words</strong></div>
            <div class="metric-card"><span>Insights</span><strong>3 sections</strong></div>
            <div class="metric-card"><span>Q&A</span><strong>Ready</strong></div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    summary_tab, insights_tab, transcript_tab, chat_tab, export_tab = st.tabs(
        ["Summary", "Insights", "Transcript", "Chat", "Export"]
    )

    with summary_tab:
        st.subheader(result["title"])
        st.markdown(result["summary"])

    with insights_tab:
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("#### Action Items")
            st.markdown(result["action_items"])
        with col2:
            st.markdown("#### Key Decisions")
            st.markdown(result["key_decisions"])
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
    with st.spinner("Processing media, transcribing audio, and building meeting intelligence..."):
        st.session_state.result = run_meeting_assistant_pipeline(source.strip(), language)
        st.session_state.last_source = source

    st.success("Analysis complete.")


def main():
    st.set_page_config(
        page_title="AI Video Assistant",
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
