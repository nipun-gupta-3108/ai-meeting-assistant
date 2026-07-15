import os
import yt_dlp
from pydub import AudioSegment
import shutil
import subprocess

ffmpeg_path = shutil.which("ffmpeg")

if ffmpeg_path is None:
    raise RuntimeError(
        "FFmpeg not found on PATH. Please install FFmpeg and add it to PATH."
    )

FFMPEG_DIR = os.path.dirname(ffmpeg_path)

DOWNLOAD_DIR = "downloads"
os.makedirs(DOWNLOAD_DIR, exist_ok=True)

print("Using FFmpeg from:", FFMPEG_DIR)


def download_audio_from_youtube(url: str) -> str:
    output_path = os.path.join(DOWNLOAD_DIR, "%(id)s_%(title).80s.%(ext)s")

    ydl_opts = {
        "format": "bestaudio/best",
        "outtmpl": output_path,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        return ydl.prepare_filename(info)


def convert_media_to_wav(input_path: str) -> str:
    output_path = os.path.splitext(input_path)[0] + "_converted.wav"

    command = [
        "ffmpeg",
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
        print(result.stderr)
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


def prepare_audio_chunks(source: str) -> list:
    if source.startswith("http://") or source.startswith("https://"):
        print("Detected YouTube URL. Downloading audio...")
        downloaded = download_audio_from_youtube(source)
        wav_path = convert_media_to_wav(downloaded)
    else:
        print("Detected local file. Converting to WAV...")
        wav_path = convert_media_to_wav(source)

    print("Chunking audio...")
    chunks = split_audio_into_chunks(wav_path)
    print(f"Audio ready — {len(chunks)} chunk(s) created.")
    return chunks
