import logging
import os
import shutil
import subprocess
import time

from pydub import AudioSegment

logger = logging.getLogger(__name__)

ffmpeg_path = shutil.which("ffmpeg")

if ffmpeg_path is None:
    raise RuntimeError(
        "FFmpeg not found on PATH. Please install FFmpeg and add it to PATH."
    )

FFMPEG_DIR = os.path.dirname(ffmpeg_path)

logger.debug("Using FFmpeg from: %s", FFMPEG_DIR)

# How old a leftover file in a managed temp directory must be before the
# startup sweep will remove it. Same mechanism and same rationale as
# core/transcript_vector_store.py's DEFAULT_STALE_COLLECTION_MAX_AGE_HOURS:
# a threshold well above any realistic single-run duration so an in-progress
# run (in this process or a concurrent one) is never at risk.
DEFAULT_STALE_TEMP_FILE_MAX_AGE_HOURS = float(
    os.getenv("STALE_TEMP_FILE_MAX_AGE_HOURS", "24")
)


# User-facing error messages
INVALID_MEDIA_MESSAGE = (
    "The uploaded file could not be read as valid audio/video. "
    "Please upload a valid recording."
)

EMPTY_TRANSCRIPT_MESSAGE = (
    "No speech was detected in the recording. "
    "Please upload a recording containing spoken audio."
)


class InvalidMediaError(Exception):
    """Raised when an uploaded media file cannot be read or decoded as valid audio/video."""

    def __init__(self, message: str = INVALID_MEDIA_MESSAGE):
        super().__init__(message)


class EmptyTranscriptError(Exception):
    """Raised when no speech is detected in the audio recording."""

    def __init__(self, message: str = EMPTY_TRANSCRIPT_MESSAGE):
        super().__init__(message)


def validate_media_file(file_path: str) -> None:
    """Validate that the given path is a readable media file containing audio.

    Raises InvalidMediaError if the file does not exist, is empty, is corrupt,
    or does not contain a decodable audio stream.
    """
    if not file_path or not os.path.isfile(file_path):
        raise InvalidMediaError(INVALID_MEDIA_MESSAGE)

    try:
        if os.path.getsize(file_path) <= 0:
            raise InvalidMediaError(INVALID_MEDIA_MESSAGE)
    except OSError as exc:
        logger.warning("Could not check size of %s: %s", file_path, exc)
        raise InvalidMediaError(INVALID_MEDIA_MESSAGE) from exc

    # Check for audio stream using ffprobe (preferred) or ffmpeg
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        probe_cmd = [
            ffprobe_bin,
            "-v",
            "error",
            "-select_streams",
            "a",
            "-show_entries",
            "stream=codec_type",
            "-of",
            "csv=p=0",
            file_path,
        ]
        try:
            probe_res = subprocess.run(
                probe_cmd,
                capture_output=True,
                text=True,
                check=False,
            )
            if probe_res.returncode != 0 or "audio" not in probe_res.stdout.lower():
                logger.warning(
                    "FFprobe media validation failed for %s: returncode=%d stderr=%s stdout=%s",
                    file_path,
                    probe_res.returncode,
                    probe_res.stderr.strip(),
                    probe_res.stdout.strip(),
                )
                raise InvalidMediaError(INVALID_MEDIA_MESSAGE)
            return
        except InvalidMediaError:
            raise
        except Exception as exc:
            logger.warning(
                "FFprobe probe encountered error for %s: %s; falling back to FFmpeg",
                file_path,
                exc,
            )

    # Fallback to FFmpeg decode test
    cmd = [
        ffmpeg_path,
        "-v",
        "error",
        "-i",
        file_path,
        "-vn",
        "-t",
        "1",
        "-f",
        "null",
        "-",
    ]
    try:
        res = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            check=False,
        )
        if res.returncode != 0:
            logger.warning(
                "FFmpeg media check failed for %s: returncode=%d stderr=%s",
                file_path,
                res.returncode,
                res.stderr.strip(),
            )
            raise InvalidMediaError(INVALID_MEDIA_MESSAGE)
    except InvalidMediaError:
        raise
    except Exception as exc:
        logger.error("FFmpeg execution error for %s: %s", file_path, exc)
        raise InvalidMediaError(INVALID_MEDIA_MESSAGE) from exc


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
        logger.error(
            "FFmpeg conversion failed for %s: %s",
            input_path,
            result.stderr,
        )
        raise InvalidMediaError(INVALID_MEDIA_MESSAGE)

    if not os.path.isfile(output_path) or os.path.getsize(output_path) <= 44:
        logger.error(
            "FFmpeg conversion produced empty or missing WAV file for %s",
            input_path,
        )
        _remove_file_if_exists(output_path)
        raise InvalidMediaError(INVALID_MEDIA_MESSAGE)

    return output_path


def split_audio_into_chunks(wav_path: str, chunk_minutes: int = 10) -> list:
    try:
        audio = AudioSegment.from_wav(wav_path)
    except Exception as exc:
        logger.error("Failed to parse WAV file %s: %s", wav_path, exc)
        raise InvalidMediaError(INVALID_MEDIA_MESSAGE) from exc

    if len(audio) == 0:
        logger.warning("Converted audio from %s has zero length", wav_path)
        raise InvalidMediaError(INVALID_MEDIA_MESSAGE)

    chunk_ms = chunk_minutes * 60 * 1000

    chunks = []

    # Tracks the path currently being written so it can be cleaned up too if
    # the export call itself is what fails (it may have written a partial
    # file before raising).
    in_progress_path = None

    try:
        for i, start in enumerate(range(0, len(audio), chunk_ms)):
            in_progress_path = f"{wav_path}_chunk_{i}.wav"

            chunk = audio[start : start + chunk_ms]
            chunk.export(in_progress_path, format="wav")

            chunks.append(in_progress_path)

    except Exception:
        # This invocation's own chunk files
        # (`{wav_path}_chunk_{i}.wav`) are the only files touched here.
        # wav_path itself and any user-owned input are never in this list.
        #
        # Cleanup is best-effort: _remove_file_if_exists() never raises,
        # so a failed cleanup cannot mask the original exception.
        logger.warning(
            "Chunking failed after creating %d chunk(s) for %s; cleaning up "
            "partial output.",
            len(chunks),
            wav_path,
        )

        for chunk_path in chunks + [in_progress_path]:
            _remove_file_if_exists(chunk_path)

        raise

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
        logger.warning(
            "Could not remove temp file %s: %s",
            path,
            exc,
        )


def cleanup_chunk_files(chunks: list) -> None:
    """Delete audio chunk files once transcription no longer needs them."""
    for chunk_path in chunks:
        _remove_file_if_exists(chunk_path)


def cleanup_stale_temp_files(
    directory: str,
    max_age_hours: float = DEFAULT_STALE_TEMP_FILE_MAX_AGE_HOURS,
) -> None:
    """Remove old leftover files from a directory this app exclusively owns.

    SAFETY SCOPE — read before pointing this at a new directory:

    This is only safe to call on the Streamlit app's own upload directory
    ("uploads/").

    Nothing else in this codebase ever writes into that directory, and
    every file that can exist there is one this app generated itself:
    the UUID-named copy Streamlit saves on upload, plus its own
    `_converted.wav` and `_chunk_N.wav` pieces.

    Because directory scoping guarantees that everything here belongs to
    the application, no filename pattern matching is required.

    This function must NEVER be pointed at a directory that can also contain
    user-owned files.

    For example, if a caller passes an arbitrary local-file input path,
    its converted WAV/chunks may be written next to the user's own file,
    so that directory cannot safely be swept.

    Only regular files are removed. Subdirectories are never recursively
    traversed or deleted.

    Best-effort throughout: cleanup failures are logged but never raised,
    so a failed sweep cannot prevent the application from starting.
    """
    if not os.path.isdir(directory):
        return

    try:
        entries = os.listdir(directory)

    except OSError as exc:
        logger.warning(
            "Could not list %s for stale-file sweep: %s",
            directory,
            exc,
        )
        return

    cutoff = time.time() - (max_age_hours * 3600)
    removed = 0

    for name in entries:
        path = os.path.join(directory, name)

        try:
            # Never recurse into directories.
            if not os.path.isfile(path):
                continue

            # Keep files that are still recent.
            if os.path.getmtime(path) >= cutoff:
                continue

            os.remove(path)
            removed += 1

            logger.info(
                "Removed stale temp file: %s",
                path,
            )

        except OSError as exc:
            logger.warning(
                "Could not remove stale temp file %s: %s",
                path,
                exc,
            )

    if removed:
        logger.info(
            "Stale-file sweep removed %d file(s) from %s.",
            removed,
            directory,
        )
    else:
        logger.debug(
            "Stale-file sweep found nothing to remove in %s.",
            directory,
        )


def prepare_audio_chunks(source: str) -> list:
    # The original local-file `source` path is NOT touched here because
    # it may be a user-owned file.
    logger.info("Validating media file: %s", source)
    validate_media_file(source)

    logger.info("Converting local file to WAV...")

    wav_path = convert_media_to_wav(source)

    logger.info("Chunking audio...")

    try:
        chunks = split_audio_into_chunks(wav_path)

    finally:
        # The converted WAV is only an intermediate for chunking.
        # Once split_audio_into_chunks has run (or failed), it is no longer
        # needed because the chunks themselves are separate files.
        _remove_file_if_exists(wav_path)

    if not chunks:
        logger.warning("No audio chunks produced for %s", source)
        raise InvalidMediaError(INVALID_MEDIA_MESSAGE)

    logger.info(
        "Audio ready — %d chunk(s) created.",
        len(chunks),
    )

    return chunks
