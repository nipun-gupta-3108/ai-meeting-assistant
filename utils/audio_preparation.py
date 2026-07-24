import logging
import os
import shutil
import subprocess

import yt_dlp
from pydub import AudioSegment

logger = logging.getLogger(__name__)

ffmpeg_path = shutil.which("ffmpeg")

if ffmpeg_path is None:
    raise RuntimeError(
        "FFmpeg not found on PATH. Please install FFmpeg and add it to PATH."
    )

FFMPEG_DIR = os.path.dirname(ffmpeg_path)

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

logger.debug("Using FFmpeg from: %s", FFMPEG_DIR)


def download_audio_from_youtube(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s_%(title).80s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
        "noplaylist": True,
        "quiet": True,
        "no_warnings": False,
        "extractor_args": {"youtube": {"player_client": ["android"]}},
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


def convert_media_to_wav(input_path: str) -> str:
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    command = [
        ffmpeg_path,
        "-y",
        "-i",
        input_path,
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        output_path,
    ]

    result = subprocess.run(
        command,
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        logger.error("FFmpeg conversion failed for %s: %s", input_path, result.stderr)
        raise RuntimeError("FFmpeg conversion failed.")

    return output_path


def split_audio_into_chunks(wav_path: str, chunk_minutes: int = 10) -> list:
    audio = AudioSegment.from_wav(wav_path)
    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    for i, start in enumerate(range(0, len(audio), chunk_ms)):
        chunk = audio[start : start + chunk_ms]
        chunk_path = f"{wav_path}_chunk_{i}.wav"
        chunk.export(chunk_path, format="wav")

        chunks.append(chunk_path)

    return chunks


def _remove_file_if_exists(path: str) -> None:
    """Best-effort delete of a single temp file.

    Never raises: a failed cleanup should never crash the pipeline or mask
    the real result/error. Logged at WARNING so it's visible without being
    treated as fatal.
    """
    try:
        if path and os.path.exists(path):
            os.remove(path)
            logger.debug("Removed temp file: %s", path)
    except OSError as exc:
        logger.warning("Could not remove temp file %s: %s", path, exc)


def cleanup_chunk_files(chunks: list) -> None:
    """Delete audio chunk files once transcription no longer needs them."""
    for chunk_path in chunks:
        _remove_file_if_exists(chunk_path)


def prepare_audio_chunks(source: str) -> list:
    downloaded_path = None

    try:
        if source.startswith("http://") or source.startswith("https://"):
            logger.info("Detected YouTube URL. Downloading audio...")
            downloaded_path = download_audio_from_youtube(source)
            wav_path = convert_media_to_wav(downloaded_path)
        else:
            logger.info("Detected local file. Converting to WAV...")
            wav_path = convert_media_to_wav(source)
    finally:
        # The raw YouTube download is only an intermediate for conversion;
        # once convert_media_to_wav has run (or failed), it's no longer
        # needed. The original local-file `source` path is NOT touched
        # here — it may not be a file this app created (e.g. a CLI user's
        # own file), so we never delete it. (See streamlit_app.py for
        # cleanup of the app's own uploaded-file copy.)
        if downloaded_path:
            _remove_file_if_exists(downloaded_path)

    logger.info("Chunking audio...")
    try:
        chunks = split_audio_into_chunks(wav_path)
    finally:
        # The converted WAV is only an intermediate for chunking; once
        # split_audio_into_chunks has run (or failed), it's no longer
        # needed — the chunks themselves are separate files.
        _remove_file_if_exists(wav_path)

    logger.info("Audio ready — %d chunk(s) created.", len(chunks))
    return chunks
