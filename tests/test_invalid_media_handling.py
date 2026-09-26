import math
import os
import struct
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import MagicMock, patch

from core.pipeline import (
    EMPTY_TRANSCRIPT_MESSAGE,
    INVALID_MEDIA_MESSAGE,
    EmptyTranscriptError,
    InvalidMediaError,
    run_meeting_assistant_pipeline,
)
from streamlit_app import (
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    SUPPORTED_EXTENSIONS,
    save_uploaded_file,
    validate_uploaded_file,
)
from utils.audio_preparation import (
    cleanup_chunk_files,
    convert_media_to_wav,
    prepare_audio_chunks,
    validate_media_file,
)


class DummyUploadedFile:
    """Mock Streamlit UploadedFile for testing."""

    def __init__(self, name: str, size: int, content: bytes | None = None):
        self.name = name
        self.size = size
        if content is not None:
            self._content = content
        else:
            self._content = b"x" * min(size, 1024)

    def getbuffer(self):
        return memoryview(self._content)


def create_synthetic_wav(path: str, duration_sec: float = 1.0, sample_rate: int = 16000) -> None:
    """Create a real, fully valid PCM WAV file with a tone using python's wave module."""
    n_samples = int(duration_sec * sample_rate)
    with wave.open(path, "wb") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(sample_rate)
        samples = [
            int(16000 * math.sin(2 * math.pi * 440 * i / sample_rate))
            for i in range(n_samples)
        ]
        raw_data = struct.pack("<" + "h" * n_samples, *samples)
        wf.writeframes(raw_data)


class TestInvalidMediaHandling(unittest.TestCase):
    def setUp(self):
        self.temp_files = []

    def tearDown(self):
        for path in self.temp_files:
            if os.path.exists(path):
                try:
                    os.remove(path)
                except OSError:
                    pass

    def _track_temp_file(self, path: str) -> str:
        self.temp_files.append(path)
        return path

    # -------------------------------------------------------------------------
    # 1. Corrupt/invalid media rejected before AI calls
    # -------------------------------------------------------------------------
    def test_corrupt_dummy_file_rejected_by_validate_media_file(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"not a valid audio file content at all" * 50)
            dummy_path = self._track_temp_file(f.name)

        with self.assertRaises(InvalidMediaError) as ctx:
            validate_media_file(dummy_path)

        self.assertEqual(str(ctx.exception), INVALID_MEDIA_MESSAGE)

    def test_corrupt_dummy_file_rejected_before_ai_calls_in_pipeline(self):
        with tempfile.NamedTemporaryFile(suffix=".mp3", delete=False) as f:
            f.write(b"fake audio data" * 100)
            dummy_path = self._track_temp_file(f.name)

        with patch("core.pipeline.transcribe_audio_chunks") as mock_transcribe, \
             patch("core.pipeline.summarize_transcript") as mock_summarize, \
             patch("core.pipeline.generate_meeting_title") as mock_title, \
             patch("core.pipeline.extract_meeting_insights_from_transcript") as mock_insights, \
             patch("core.pipeline.build_transcript_rag_chain") as mock_rag:

            with self.assertRaises(InvalidMediaError) as ctx:
                run_meeting_assistant_pipeline(dummy_path)

            self.assertEqual(str(ctx.exception), INVALID_MEDIA_MESSAGE)
            mock_transcribe.assert_not_called()
            mock_summarize.assert_not_called()
            mock_title.assert_not_called()
            mock_insights.assert_not_called()
            mock_rag.assert_not_called()

    def test_video_without_audio_rejected_before_ai_calls(self):
        # Simulate video file where ffprobe / ffmpeg finds no audio stream
        with tempfile.NamedTemporaryFile(suffix=".mp4", delete=False) as f:
            f.write(b"video_without_audio_stream")
            video_path = self._track_temp_file(f.name)

        with patch("core.pipeline.transcribe_audio_chunks") as mock_transcribe, \
             patch("core.pipeline.summarize_transcript") as mock_summarize:

            with self.assertRaises(InvalidMediaError) as ctx:
                run_meeting_assistant_pipeline(video_path)

            self.assertEqual(str(ctx.exception), INVALID_MEDIA_MESSAGE)
            mock_transcribe.assert_not_called()
            mock_summarize.assert_not_called()

    def test_nonexistent_file_rejected(self):
        with self.assertRaises(InvalidMediaError) as ctx:
            validate_media_file("nonexistent_recording_file.mp3")
        self.assertEqual(str(ctx.exception), INVALID_MEDIA_MESSAGE)

    def test_zero_byte_media_rejected(self):
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            empty_path = self._track_temp_file(f.name)

        with self.assertRaises(InvalidMediaError) as ctx:
            validate_media_file(empty_path)
        self.assertEqual(str(ctx.exception), INVALID_MEDIA_MESSAGE)

    # -------------------------------------------------------------------------
    # 2. Empty transcript stops the pipeline
    # -------------------------------------------------------------------------
    def test_empty_transcript_stops_pipeline_before_llm_calls(self):
        with patch("core.pipeline.prepare_audio_chunks", return_value=["dummy_chunk.wav"]), \
             patch("core.pipeline.cleanup_chunk_files") as mock_cleanup, \
             patch("core.pipeline.transcribe_audio_chunks", return_value=""), \
             patch("core.pipeline.summarize_transcript") as mock_summarize, \
             patch("core.pipeline.generate_meeting_title") as mock_title, \
             patch("core.pipeline.extract_meeting_insights_from_transcript") as mock_insights, \
             patch("core.pipeline.build_transcript_rag_chain") as mock_rag:

            with self.assertRaises(EmptyTranscriptError) as ctx:
                run_meeting_assistant_pipeline("some_valid_audio.wav")

            self.assertEqual(str(ctx.exception), EMPTY_TRANSCRIPT_MESSAGE)
            # Chunks must be cleaned up in finally block
            mock_cleanup.assert_called_once_with(["dummy_chunk.wav"])
            # No LLM or RAG calls must occur
            mock_summarize.assert_not_called()
            mock_title.assert_not_called()
            mock_insights.assert_not_called()
            mock_rag.assert_not_called()

    def test_whitespace_only_transcript_stops_pipeline_before_llm_calls(self):
        whitespace_transcripts = ["   ", "\n\t  \n", "   \r\n   "]
        for ws in whitespace_transcripts:
            with patch("core.pipeline.prepare_audio_chunks", return_value=["chunk.wav"]), \
                 patch("core.pipeline.cleanup_chunk_files") as mock_cleanup, \
                 patch("core.pipeline.transcribe_audio_chunks", return_value=ws), \
                 patch("core.pipeline.summarize_transcript") as mock_summarize, \
                 patch("core.pipeline.build_transcript_rag_chain") as mock_rag:

                with self.assertRaises(EmptyTranscriptError) as ctx:
                    run_meeting_assistant_pipeline("audio.wav")

                self.assertEqual(str(ctx.exception), EMPTY_TRANSCRIPT_MESSAGE)
                mock_cleanup.assert_called_once()
                mock_summarize.assert_not_called()
                mock_rag.assert_not_called()

    # -------------------------------------------------------------------------
    # 3. Valid meeting_test-style audio still works
    # -------------------------------------------------------------------------
    def test_valid_meeting_style_audio_pipeline(self):
        temp_dir = tempfile.gettempdir()
        valid_wav_path = self._track_temp_file(
            os.path.join(temp_dir, "meeting_test.wav")
        )
        create_synthetic_wav(valid_wav_path, duration_sec=1.0)

        # 1. Media validation passes
        validate_media_file(valid_wav_path)

        # 2. Audio chunks are prepared properly
        chunks = prepare_audio_chunks(valid_wav_path)
        self.assertGreater(len(chunks), 0)
        for c in chunks:
            self._track_temp_file(c)
        cleanup_chunk_files(chunks)

        # 3. Full pipeline runs with valid transcript and mocked LLM/RAG responses
        expected_summary = "Meeting summary points"
        expected_title = "Quarterly Sync"
        expected_insights = {
            "action_items": ["Action 1"],
            "key_decisions": ["Decision 1"],
            "open_questions": ["Question 1"],
        }
        mock_chain = MagicMock()

        with patch("core.pipeline.transcribe_audio_chunks", return_value="Welcome to the quarterly sync meeting."), \
             patch("core.pipeline.summarize_transcript", return_value=expected_summary) as mock_sum, \
             patch("core.pipeline.generate_meeting_title", return_value=expected_title) as mock_ttl, \
             patch("core.pipeline.extract_meeting_insights_from_transcript", return_value=expected_insights) as mock_ins, \
             patch("core.pipeline.build_transcript_rag_chain", return_value=mock_chain) as mock_rag:

            result = run_meeting_assistant_pipeline(valid_wav_path)

            self.assertEqual(result["title"], expected_title)
            self.assertEqual(result["transcript"], "Welcome to the quarterly sync meeting.")
            self.assertEqual(result["summary"], expected_summary)
            self.assertEqual(result["action_items"], ["Action 1"])
            self.assertEqual(result["key_decisions"], ["Decision 1"])
            self.assertEqual(result["open_questions"], ["Question 1"])
            self.assertEqual(result["rag_chain"], mock_chain)
            self.assertTrue(result["collection_name"].startswith("meeting_"))

            mock_sum.assert_called_once()
            mock_ttl.assert_called_once()
            mock_ins.assert_called_once()
            mock_rag.assert_called_once()

    # -------------------------------------------------------------------------
    # 4. Oversized file still rejected
    # -------------------------------------------------------------------------
    def test_oversized_file_still_rejected(self):
        oversized = DummyUploadedFile("huge.mp4", size=MAX_UPLOAD_SIZE_BYTES + 1)
        is_valid, msg = validate_uploaded_file(oversized)
        self.assertFalse(is_valid)
        self.assertIn(f"exceeds the {MAX_UPLOAD_SIZE_MB} MB limit", msg)
        self.assertIn("audio-only version", msg)

        size_250mb = 250 * 1024 * 1024
        file_250mb = DummyUploadedFile("video.mp4", size=size_250mb)
        is_valid_250, msg_250 = validate_uploaded_file(file_250mb)
        self.assertFalse(is_valid_250)
        self.assertIn("250.0 MB", msg_250)
        self.assertIn(f"exceeds the {MAX_UPLOAD_SIZE_MB} MB limit", msg_250)

    # -------------------------------------------------------------------------
    # 5. Supported extensions still accepted
    # -------------------------------------------------------------------------
    def test_supported_extensions_still_accepted(self):
        expected_extensions = ["mp3", "mp4", "wav", "m4a", "webm", "mov", "aac"]
        self.assertEqual(set(SUPPORTED_EXTENSIONS), set(expected_extensions))

        for ext in SUPPORTED_EXTENSIONS:
            file = DummyUploadedFile(f"meeting_recording.{ext}", size=10 * 1024 * 1024)
            is_valid, msg = validate_uploaded_file(file)
            self.assertTrue(is_valid, f"Extension {ext} should be accepted")
            self.assertEqual(msg, "")


if __name__ == "__main__":
    unittest.main()
