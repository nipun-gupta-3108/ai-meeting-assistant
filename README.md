# 🎙️ AI Meeting Assistant

An AI-powered meeting assistant that transcribes meeting recordings, generates concise summaries, extracts meeting insights, and enables transcript-based question answering using Retrieval-Augmented Generation (RAG).

Available as both a **CLI tool** and a **Streamlit web app**.

---

## ✨ Features

- **Flexible input** — process a YouTube URL or an uploaded local audio/video file.
- **Automatic chunking** — audio is converted to mono 16kHz WAV and split into fixed-length chunks before transcription.
- **Dual transcription engines**
  - 🇬🇧 **English** → local [OpenAI Whisper](https://github.com/openai/whisper) model.
  - 🇮🇳 **Hinglish / Hindi** → [Sarvam AI](https://www.sarvam.ai/) speech-to-text-translate API (audio is further split into ≤30s pieces to satisfy the API's request limit).
- **Meeting title generation** — a short, professional title generated from the transcript.
- **Map-reduce summarization** — the transcript is chunked, summarized in parts, then combined into one final bullet-point summary.
- **Structured insight extraction**
  - ✅ Action items (task, owner, deadline)
  - 📌 Key decisions
  - ❓ Open questions / follow-ups
- **Retrieval-Augmented Q&A** — chat with the transcript using a Chroma vector store and sentence-embedding retrieval, answers grounded strictly in the transcript.
- **Streamlit UI** — sidebar-driven workflow with Summary, Insights, Transcript, Chat, and Export tabs.
- **Text export** — download the full analysis or raw transcript as a `.txt` file.

---

## 🏗️ Architecture

```
                 ┌──────────────────────┐
                 │  YouTube URL /       │
                 │  Local audio-video   │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │  Audio Preparation   │
                 │  (yt-dlp / pydub)    │
                 │  → mono 16kHz WAV    │
                 │  → chunked segments  │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Transcription      │
                 │ Whisper  |  Sarvam AI│
                 │ (English)|(Hinglish) │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Full Transcript    │
                 └──────────┬───────────┘
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
      ┌───────────┐ ┌───────────────┐ ┌────────────────┐
      │  Summary  │ │    Insights   │ │  Vector Store  │
      │ (Groq LLM,│ │ (Action items,│ │ (Chroma +      │
      │ map-reduce)│ │ decisions,   │ │ HF embeddings) │
      └───────────┘ │ questions)    │ └────────┬───────┘
                     └───────────────┘          │
                                                 ▼
                                        ┌──────────────────┐
                                        │  RAG Q&A Chain   │
                                        │  (Groq LLM)      │
                                        └──────────────────┘
```

---

## 📁 Project Structure

```
.
├── core/
│   ├── audio_transcription.py     # Whisper + Sarvam transcription logic
│   ├── llm_client.py               # Groq LLM client factory
│   ├── transcript_summary.py       # Map-reduce summarization + title generation
│   ├── transcript_insights.py      # Action items / decisions / open questions
│   ├── transcript_qa.py            # RAG chain construction and Q&A
│   └── transcript_vector_store.py  # Chroma vector store creation & retrieval
├── utils/
│   └── audio_preparation.py        # Download, convert, and chunk audio
├── meeting_assistant_cli.py        # CLI entry point / pipeline orchestration
├── streamlit_app.py                # Streamlit web interface
├── requirements.txt
└── .gitignore
```

---

## 🛠️ Tech Stack

| Layer            | Technology                                      |
|-------------------|-------------------------------------------------|
| Audio acquisition | `yt-dlp`, `pydub`, `ffmpeg`                      |
| Transcription     | `openai-whisper` (local), Sarvam AI API          |
| LLM               | Groq (`langchain-groq`, `llama-3.3-70b-versatile`) |
| Orchestration     | LangChain (LCEL chains)                          |
| Vector search     | ChromaDB + `langchain-huggingface` embeddings (`BAAI/bge-small-en-v1.5`) |
| Web UI            | Streamlit                                        |

---

## 🚀 Installation

```bash
# 1. Clone the repository
git clone https://github.com/nipun-gupta-3108/ai-meeting-assistant.git
cd ai-meeting-assistant

# 2. Create and activate a virtual environment
python -m venv .venv
source .venv/bin/activate   # Windows: .venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Make sure FFmpeg is installed and on your PATH
ffmpeg -version
```

---

## 🔑 Environment Variables

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key
SARVAM_API_KEY=your_sarvam_api_key      # only required for Hinglish/Hindi transcription

# Optional overrides
WHISPER_MODEL=small                     # tiny / base / small / medium / large
LLM_MODEL=llama-3.3-70b-versatile
SARVAM_STT_MODEL=saaras:v2.5
```

---

## ▶️ Running Locally

**CLI:**

```bash
python meeting_assistant_cli.py
```
You'll be prompted for a YouTube URL (or local file path) and a language (`english` / `hinglish`). After processing, an interactive Q&A prompt lets you ask questions about the transcript.

**Streamlit web app:**

```bash
streamlit run streamlit_app.py
```
Use the sidebar to choose an input source and language, run the analysis, then explore the Summary, Insights, Transcript, Chat, and Export tabs.

---

## 🖼️ Screenshots

> _Add screenshots or a short demo GIF of the Streamlit app here._

| Home / Empty State | Summary & Insights | Chat |
|---|---|---|
| `assets/screenshot-home.png` | `assets/screenshot-summary.png` | `assets/screenshot-chat.png` |

---

## 🔭 Future Improvements

- Persist and reload past sessions instead of recomputing per run.
- Speaker-level timestamps for easier navigation through the transcript.
- PDF export of the full meeting report.
- Batch processing for multiple files/URLs in one run.
- Automated tests for the transcription and RAG pipelines.

---

## 📄 License

This project is licensed under the [MIT License](LICENSE).

---

## 👤 Author

**Nipun Gupta**
[GitHub](https://github.com/nipun-gupta-3108) · [LinkedIn](https://linkedin.com/in/link-nipun-gupta)
