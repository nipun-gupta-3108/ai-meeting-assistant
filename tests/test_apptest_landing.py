import unittest
from streamlit.testing.v1 import AppTest


class TestLandingAppTest(unittest.TestCase):
    def test_authenticated_landing_renders_upload_ux_and_limits(self):
        # Allow sufficient timeout for cold-start module loading
        at = AppTest.from_file("streamlit_app.py", default_timeout=60)
        at.run()

        # Unauthenticated: should render login/signup
        self.assertEqual(len(at.exception), 0, "No exceptions on initial render")

        # Simulate authenticated user
        at.session_state["current_user"] = {
            "id": "test-user-id",
            "name": "Production Tester",
            "email": "tester@example.com",
        }
        at.run()

        # Check for clean render with no exceptions
        self.assertEqual(len(at.exception), 0, "No exceptions on authenticated landing")

        # Verify file uploader is present
        self.assertGreater(
            len(at.file_uploader), 0, "File uploader widget should be present"
        )

        uploader = at.file_uploader[0]
        # Verify 150 MB is in uploader label/help
        self.assertIn("150 MB", uploader.label)
        self.assertIn("150 MB", uploader.help)

        # Verify UX recommendation caption is present
        caption_texts = [c.value for c in at.caption]
        expected_ux_message = (
            "Audio files are recommended for faster uploads. Video files are supported, "
            "but large video files may take longer to upload and process."
        )
        self.assertTrue(
            any(expected_ux_message in text for text in caption_texts),
            f"Expected UX guidance message not found in captions: {caption_texts}",
        )

        # Verify footnote mentions 150 MB and audio recommendation
        markdown_texts = [m.value for m in at.markdown]
        self.assertTrue(
            any("up to 150 MB" in text and "Audio recommended" in text for text in markdown_texts),
            "Expected footnote with 150 MB and audio recommendation not found in markdown",
        )


if __name__ == "__main__":
    unittest.main()
