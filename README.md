# 🎙️ AI Meeting Assistant

![Python](https://img.shields.io/badge/Python-3.11+-3776AB?logo=python&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Web%20App-FF4B4B?logo=streamlit&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)
![LangChain](https://img.shields.io/badge/LangChain-LCEL-blue)
![ChromaDB](https://img.shields.io/badge/Vector%20DB-ChromaDB-purple)

An AI-powered meeting assistant that transcribes meeting recordings, generates concise summaries, extracts structured meeting insights, and enables transcript-based question answering using Retrieval-Augmented Generation (RAG).

Built as a **Streamlit web application** using **LangChain**, **Groq**, **Faster-Whisper**, **Sarvam AI**, and **ChromaDB**.

---

## 🎥 Demo

> Add a short demo GIF or video here.

Example:

```
assets/demo.gif
```

or

```
https://github.com/user-attachments/...
```

---

## ✨ Features

### 🎧 Flexible Input

- Process a **YouTube URL**
- Upload a local **audio/video file**

### 📝 Automatic Audio Processing

- Converts audio to **mono 16kHz WAV**
- Automatically splits long recordings into manageable chunks

### 🎤 Dual Transcription Engines

#### 🇬🇧 English

- Local Faster-Whisper model

#### 🇮🇳 Hinglish / Hindi

- Sarvam AI Speech-to-Text API
- Automatically splits audio into ≤30 second segments to satisfy API limits

### 📰 AI Meeting Summary

- Automatic meeting title generation
- Map-reduce summarization pipeline
- Clean bullet-point summaries

### 📌 Structured Meeting Insights

Extracts:

- ✅ Action Items
- 📌 Key Decisions
- ❓ Open Questions

### 💬 Multi-turn RAG Chat

- Ask questions about the transcript
- Context-aware follow-up conversations
- Transcript chunk citations
- Chroma vector search

### 🗂️ Meeting History

- SQLite persistence
- User-specific history
- Reopen previous analyses
- Lazy RAG reconstruction

### 🔐 Authentication

- Email/password signup
- Secure bcrypt password hashing
- Private meeting history

### 📄 Export

Download:

- Full meeting analysis
- Raw transcript

---

# 🔄 Pipeline Flow

```
YouTube URL / Audio File
          │
          ▼
 Audio Preparation
          │
          ▼
   Audio Chunking
          │
          ▼
  Speech Transcription
          │
          ▼
  Full Transcript
          │
 ┌────────┼─────────┐
 ▼        ▼         ▼
Title   Summary  Insights
                  │
                  ▼
          Vector Store
                  │
                  ▼
          Transcript Chat
                  │
                  ▼
       SQLite Persistence
```

---

# 🏗️ Architecture

```text
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
                 │ Whisper  | Sarvam AI │
                 └──────────┬───────────┘
                            │
                            ▼
                 ┌──────────────────────┐
                 │   Full Transcript    │
                 └──────────┬───────────┘
              ┌─────────────┼─────────────┐
              ▼             ▼             ▼
      ┌───────────┐ ┌───────────────┐ ┌────────────────┐
      │ Summary   │ │ Insights      │ │ Vector Store   │
      └───────────┘ └───────────────┘ └────────┬───────┘
                                               │
                                               ▼
                                      ┌────────────────┐
                                      │ RAG Q&A Chain  │
                                      └────────────────┘
```

---

# 📁 Project Structure

```text
.
├── core/
│   ├── audio_transcription.py
│   ├── auth.py
│   ├── llm_client.py
│   ├── logging_config.py
│   ├── meeting_repository.py
│   ├── pipeline.py
│   ├── transcript_summary.py
│   ├── transcript_insights.py
│   ├── transcript_qa.py
│   ├── transcript_vector_store.py
│   └── user_repository.py
│
├── utils/
│   └── audio_preparation.py
│
├── assets/
│   └── style.css
│
├── streamlit_app.py
├── requirements.txt
└── .gitignore
```

---

# 🛠️ Tech Stack

| Layer              | Technology                |
| ------------------ | ------------------------- |
| Frontend           | Streamlit                 |
| Audio Processing   | yt-dlp, FFmpeg, pydub     |
| Speech Recognition | Faster-Whisper, Sarvam AI |
| LLM                | Groq (Llama 3.3 70B)      |
| AI Framework       | LangChain (LCEL)          |
| Vector Database    | ChromaDB                  |
| Embeddings         | BAAI/bge-base-en-v1.5    |
| Authentication     | bcrypt                    |
| Database           | SQLite                    |

---

# 🚀 Installation

> **Python 3.11 or newer is recommended.**

```bash
git clone https://github.com/nipun-gupta-3108/ai-meeting-assistant.git

cd ai-meeting-assistant

python -m venv .venv

# Linux / macOS
source .venv/bin/activate

# Windows
.venv\Scripts\activate

pip install -r requirements.txt

ffmpeg -version
```

---

# 🔑 Environment Variables

Create a `.env` file:

```env
# Required

GROQ_API_KEY=

# Required only for Hinglish / Hindi transcription

SARVAM_API_KEY=

# Optional

WHISPER_MODEL=small

LLM_MODEL=llama-3.3-70b-versatile

SARVAM_STT_MODEL=saaras:v2.5
```

---

# ▶️ Running

```bash
streamlit run streamlit_app.py
```

Workflow:

1. Sign up or log in
2. Select YouTube URL or File Upload
3. Choose language
4. Analyze the meeting
5. Review summary and insights
6. Chat with the transcript
7. Export the report
8. Reopen previous meetings anytime

---

# 🖼️ Screenshots

| Home           | Summary        | Chat           |
| -------------- | -------------- | -------------- |
| Add Screenshot | Add Screenshot | Add Screenshot |

---

# 🔮 Future Improvements

- Speaker diarization
- Speaker-level timestamps
- PDF report export
- Batch processing
- Automated tests

---

# 📄 License

This project is licensed under the **MIT License**.

---

# 👤 Author

**Nipun Gupta**

GitHub: https://github.com/nipun-gupta-3108

LinkedIn: https://linkedin.com/in/link-nipun-gupta
