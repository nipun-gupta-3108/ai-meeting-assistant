from faster_whisper import WhisperModel
import os
import torch
import requests
from pydub import AudioSegment
import streamlit as st

# Sarvam's sync STT-translate API rejects audio longer than 30s.
# We slice each chunk into 25s pieces (with a 5s safety margin) before sending.
SARVAM_PIECE_SECONDS = 25


WHISPER_MODEL = os.getenv("WHISPER_MODEL", "base")


SARVAM_API_KEY = os.getenv("SARVAM_API_KEY")
SARVAM_STT_TRANSLATE_URL = "https://api.sarvam.ai/speech-to-text-translate"
SARVAM_MODEL = os.getenv("SARVAM_STT_MODEL", "saaras:v2.5")

device = "cuda" if torch.cuda.is_available() else "cpu"


@st.cache_resource
def load_whisper_model():
    compute_type = "float16" if device == "cuda" else "int8"

    print(f"Loading Faster-Whisper ({WHISPER_MODEL}) on {device} ({compute_type})...")

    model = WhisperModel(
        WHISPER_MODEL,
        device=device,
        compute_type=compute_type,
    )

    print("Faster-Whisper model loaded.")

    return model


def transcribe_audio_chunk_with_whisper(chunk_path: str) -> str:
    model = load_whisper_model()

    segments, info = model.transcribe(
        chunk_path,
        beam_size=1,
        vad_filter=True,
    )

    text = " ".join(segment.text for segment in segments)

    return text


def _send_audio_piece_to_sarvam(piece_path: str) -> str:
    headers = {"api-subscription-key": SARVAM_API_KEY}

    with open(piece_path, "rb") as f:
        files = {"file": (os.path.basename(piece_path), f, "audio/wav")}
        data = {
            "model": SARVAM_MODEL,
            "with_diarization": "false",
        }

        response = requests.post(
            SARVAM_STT_TRANSLATE_URL,
            headers=headers,
            files=files,
            data=data,
            timeout=30,
        )

    response.raise_for_status()

    data = response.json()

    text = data.get("transcript")

    if text is None:
        print("Warning: Sarvam response did not contain a transcript.")
        return ""

    return text


def transcribe_audio_chunk_with_sarvam(chunk_path: str) -> str:
    """
    Sarvam sync API only accepts ≤30s audio. We split this chunk into
    25-second pieces, send each separately, and join the transcripts.
    """
    if not SARVAM_API_KEY:
        raise RuntimeError("SARVAM_API_KEY is not set in environment / .env")

    audio = AudioSegment.from_wav(chunk_path)
    piece_ms = SARVAM_PIECE_SECONDS * 1000

    full_text = ""

    for i, start in enumerate(range(0, len(audio), piece_ms)):
        piece = audio[start : start + piece_ms]
        piece_path = f"{chunk_path}_sv_{i}.wav"
        piece.export(piece_path, format="wav")

        try:
            piece_text = _send_audio_piece_to_sarvam(piece_path)

            if piece_text:
                full_text += piece_text + " "
        finally:
            if os.path.exists(piece_path):
                os.remove(piece_path)

    return full_text.strip()


def transcribe_audio_chunk(chunk_path: str, language: str = "english") -> str:
    """
    Route one chunk to Whisper or Sarvam depending on language choice.
    - english  → Whisper (local model)
    - hinglish → Sarvam (translates to English while transcribing)
    """
    if language.lower() == "hinglish":
        return transcribe_audio_chunk_with_sarvam(chunk_path)
    return transcribe_audio_chunk_with_whisper(chunk_path)


def transcribe_audio_chunks(chunks: list, language: str = "english") -> str:

    full_transcript = ""

    engine = "Sarvam AI" if language.lower() == "hinglish" else "Whisper"
    print(f"Using {engine} for transcription.")

    for i, chunk in enumerate(chunks):

        print(f"Transcribing chunk {i + 1}/{len(chunks)}...")

        text = transcribe_audio_chunk(chunk, language=language)

        full_transcript += text + " "

    print("Transcription complete.")

    return full_transcript.strip()
