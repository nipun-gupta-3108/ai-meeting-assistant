import os
import unittest
from pathlib import Path

from streamlit_app import (
    MAX_UPLOAD_SIZE_BYTES,
    MAX_UPLOAD_SIZE_MB,
    SUPPORTED_EXTENSIONS,
    save_uploaded_file,
    validate_uploaded_file,
)
from utils.audio_preparation import (
    cleanup_chunk_files,
    cleanup_stale_temp_files,
    convert_media_to_wav,
    prepare_audio_chunks,
    split_audio_into_chunks,
    validate_media_file,
)


class DummyUploadedFile:
    """Mock Streamlit UploadedFile for testing validation and saving."""

    def __init__(self, name: str, size: int, content: bytes | None = None):
        self.name = name
        self.size = size
        if content is not None:
            self._content = content
        else:
            # Provide sample bytes matching the length up to 1KB to keep memory low
            self._content = b"x" * min(size, 1024)

    def getbuffer(self):
        return memoryview(self._content)


class TestUploadValidation(unittest.TestCase):
    def test_upload_none_rejected(self):
        is_valid, msg = validate_uploaded_file(None)
        self.assertFalse(is_valid)
        self.assertIn("Upload an audio or video file", msg)

    def test_valid_small_audio_upload_accepted(self):
        file = DummyUploadedFile("meeting.mp3", size=5 * 1024 * 1024)
        is_valid, msg = validate_uploaded_file(file)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")

    def test_valid_small_video_upload_accepted(self):
        file = DummyUploadedFile("presentation.mp4", size=50 * 1024 * 1024)
        is_valid, msg = validate_uploaded_file(file)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")

    def test_all_supported_extensions_accepted(self):
        for ext in SUPPORTED_EXTENSIONS:
            file = DummyUploadedFile(f"recording.{ext}", size=1024 * 1024)
            is_valid, msg = validate_uploaded_file(file)
            self.assertTrue(is_valid, f"Extension {ext} should be accepted")
            self.assertEqual(msg, "")

    def test_unsupported_extension_rejected(self):
        for ext in ["exe", "txt", "pdf", "zip", "py"]:
            file = DummyUploadedFile(f"bad_file.{ext}", size=1024)
            is_valid, msg = validate_uploaded_file(file)
            self.assertFalse(is_valid, f"Extension {ext} should be rejected")
            self.assertIn("Unsupported file format", msg)

    def test_empty_file_rejected(self):
        file = DummyUploadedFile("empty.wav", size=0, content=b"")
        is_valid, msg = validate_uploaded_file(file)
        self.assertFalse(is_valid)
        self.assertIn("empty", msg)

    def test_exact_limit_accepted(self):
        file = DummyUploadedFile("boundary.mp4", size=MAX_UPLOAD_SIZE_BYTES)
        is_valid, msg = validate_uploaded_file(file)
        self.assertTrue(is_valid)
        self.assertEqual(msg, "")

    def test_oversized_upload_rejected_cleanly(self):
        # 1 byte over 150 MB
        oversized = DummyUploadedFile("over.mp4", size=MAX_UPLOAD_SIZE_BYTES + 1)
        is_valid, msg = validate_uploaded_file(oversized)
        self.assertFalse(is_valid)
        self.assertIn(f"exceeds the {MAX_UPLOAD_SIZE_MB} MB limit", msg)
        self.assertIn("audio-only version", msg)

    def test_large_251mb_video_upload_rejected_cleanly(self):
        # The specific user scenario: 251.2 MB 1080p MP4
        size_251_2_mb = int(251.2 * 1024 * 1024)
        file = DummyUploadedFile("large_meeting_1080p.mp4", size=size_251_2_mb)
        is_valid, msg = validate_uploaded_file(file)
        self.assertFalse(is_valid)
        self.assertIn("251.2 MB", msg)
        self.assertIn(f"exceeds the {MAX_UPLOAD_SIZE_MB} MB limit", msg)

    def test_save_uploaded_file_and_cleanup(self):
        content = b"dummy audio content 12345"
        file = DummyUploadedFile("test_audio.wav", size=len(content), content=content)
        saved_path = save_uploaded_file(file)

        self.assertTrue(os.path.exists(saved_path))
        self.assertTrue(saved_path.endswith(".wav"))
        with open(saved_path, "rb") as f:
            self.assertEqual(f.read(), content)

        # Cleanup
        Path(saved_path).unlink()
        self.assertFalse(os.path.exists(saved_path))

    def test_oversized_file_is_not_saved_to_disk(self):
        uploads_dir = Path("uploads")
        before_files = set(uploads_dir.iterdir()) if uploads_dir.exists() else set()

        oversized = DummyUploadedFile(
            "huge.mp4", size=300 * 1024 * 1024, content=b"fake video"
        )
        is_valid, _ = validate_uploaded_file(oversized)
        self.assertFalse(is_valid)

        # Application logic only calls save_uploaded_file if is_valid is True
        if is_valid:
            save_uploaded_file(oversized)

        after_files = set(uploads_dir.iterdir()) if uploads_dir.exists() else set()
        self.assertEqual(
            before_files, after_files, "No files should be written to uploads/ for invalid files"
        )

    def test_audio_preparation_pipeline_functions_intact(self):
        """Verify that existing audio preparation functions remain intact and callable."""
        self.assertTrue(callable(convert_media_to_wav))
        self.assertTrue(callable(split_audio_into_chunks))
        self.assertTrue(callable(cleanup_chunk_files))
        self.assertTrue(callable(cleanup_stale_temp_files))
        self.assertTrue(callable(prepare_audio_chunks))
        self.assertTrue(callable(validate_media_file))


if __name__ == "__main__":
    unittest.main()
