"""Community Cloud entrypoint in direct mode, with no database or paid calls."""

from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from media_rag.config import RagError, Settings


APP = str(Path(__file__).resolve().parents[1] / "apps" / "youtube_rag" / "app.py")
URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
ANSWER = {"status": "answered", "message": "", "claims": [
    {"text": "Supported benefit", "start_sec": 5, "end_sec": 20},
]}


def find(elements, label):
    return next(item for item in elements if item.label == label)


class DirectAppTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()

    def app(self, key="test-key"):
        self.settings = self.enterContext(patch("youtube_rag.direct_app.load_direct_settings",
                                               return_value=Settings(gemini_api_key=key)))
        self.ask = self.enterContext(patch("youtube_rag.direct_app.ask_youtube", return_value=ANSWER))
        self.infrastructure = self.enterContext(patch("media_rag.config.Settings.load", side_effect=AssertionError("DB config")))
        self.library = self.enterContext(patch("youtube_rag.service.YouTubeLibrary", side_effect=AssertionError("DB access")))
        self.tools = self.enterContext(patch("media_rag_app.check_tools", side_effect=AssertionError("FFmpeg")))
        app = AppTest.from_file(APP, default_timeout=15).run()
        self.assertFalse(app.exception)
        return app

    def submit(self, app, url=URL, question="What is the benefit?"):
        find(app.text_input, "YouTube URL").set_value(url)
        find(app.text_area, "What would you like to know?").set_value(question)
        find(app.button, "Ask Gemini").click().run()
        self.assertFalse(app.exception)

    def test_default_mode_works_with_key_only_and_never_initializes_infrastructure(self):
        app = self.app()
        self.assertEqual(app.radio[0].value, "Ask YouTube directly")
        self.assertFalse(find(app.button, "Ask Gemini").disabled)
        self.submit(app, "https://youtu.be/jNQXAC9IVRw?t=15&si=tracking")
        self.ask.assert_called_once_with(Settings(gemini_api_key="test-key"), URL, "What is the benefit?")
        self.infrastructure.assert_not_called()
        self.library.assert_not_called()
        self.tools.assert_not_called()
        self.assertTrue(any(item.proto.url == URL + "&t=5s" for item in app.get("link_button")))
        self.assertTrue(any("have not been independently verified" in item.value for item in app.caption))
        app.run()
        self.ask.assert_called_once()

    def test_no_key_disables_submission_and_explains_setup(self):
        app = self.app(key="")
        self.assertTrue(find(app.button, "Ask Gemini").disabled)
        self.assertTrue(any("GEMINI_API_KEY" in item.value for item in app.info))
        self.ask.assert_not_called()

    def test_invalid_url_or_empty_question_never_calls_gemini(self):
        app = self.app()
        self.submit(app, "https://example.com/private")
        self.assertTrue(any("HTTPS YouTube" in item.value for item in app.error))
        self.submit(app, question=" ")
        self.assertTrue(any("Enter a question" in item.value for item in app.error))
        self.ask.assert_not_called()

    def test_error_retry_is_explicit_and_preserves_the_submitted_question(self):
        app = self.app()
        self.ask.side_effect = [RagError("Gemini rate limit reached. Retry later."), ANSWER]
        self.submit(app)
        self.assertTrue(any("rate limit" in item.value for item in app.error))
        app.run()
        self.ask.assert_called_once()
        find(app.text_input, "YouTube URL").set_value("https://example.com/changed")
        find(app.text_area, "What would you like to know?").set_value("Unsubmitted edit")
        find(app.button, "Retry Gemini answer").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(self.ask.call_count, 2)
        self.assertEqual(self.ask.call_args.args[1:], (URL, "What is the benefit?"))
        self.assertFalse(app.error)
        self.assertTrue(any("Supported benefit" in item.value for item in app.markdown))

    def test_new_failure_clears_old_answer_and_unexpected_error_body_is_hidden(self):
        app = self.app()
        self.submit(app)
        self.ask.side_effect = RuntimeError("private-key private-url")
        self.submit(app, question="Another question?")
        self.assertFalse(any("Supported benefit" in item.value for item in app.markdown))
        self.assertTrue(app.error)
        self.assertNotIn("private-key", app.error[0].value)

    def test_output_is_escaped_and_links_always_use_the_submitted_video(self):
        app = self.app()
        self.ask.return_value = {"status": "answered", "message": "", "claims": [
            {"text": "![tracker](https://evil.example/pixel)<script>alert(1)</script>",
             "start_sec": 5, "end_sec": 20},
        ]}
        self.submit(app, question="[private](https://evil.example)")
        self.assertTrue(any("\\!\\[tracker\\]" in item.value for item in app.markdown))
        self.assertTrue(all(item.proto.url.startswith(URL) for item in app.get("link_button")))

    def test_insufficient_evidence_does_not_display_claims(self):
        app = self.app()
        self.ask.return_value = {"status": "insufficient_evidence", "message": "Not explained in this video.", "claims": []}
        self.submit(app)
        self.assertTrue(any("Not explained" in item.value for item in app.info))
        self.assertFalse(app.get("video"))

    def test_youtube_secret_table_is_passed_to_direct_configuration(self):
        with patch("streamlit.secrets", {"youtube_rag": {"GEMINI_API_KEY": "youtube-key"},
                                         "media_rag": {"GEMINI_API_KEY": "media-key"}}):
            self.app()
            self.settings.assert_called_once_with({"GEMINI_API_KEY": "youtube-key"})

    def test_switching_modes_preserves_direct_answer_without_another_generation(self):
        app = self.app()
        self.submit(app)
        self.infrastructure.side_effect = None
        self.infrastructure.return_value = Settings(database_url="postgresql://test", minio_access_key="test",
                                                   minio_secret_key="test", gemini_api_key="test-key")
        self.library.side_effect = None
        library = Mock()
        library.database.list_youtube_assets.return_value = []
        self.library.return_value = library
        self.tools.side_effect = None
        app.radio[0].set_value("Indexed library").run()
        self.assertFalse(app.exception)
        self.assertTrue(find(app.button, "Import & index"))
        app.radio[0].set_value("Ask YouTube directly").run()
        self.assertFalse(app.exception)
        self.assertTrue(any("Supported benefit" in item.value for item in app.markdown))
        self.ask.assert_called_once()


if __name__ == "__main__":
    unittest.main()
