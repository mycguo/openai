"""Streamlit UI regressions with local services and providers substituted."""

from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from media_rag.config import Settings
from media_rag.models import Evidence, Hit


APP = str(Path(__file__).resolve().parents[1] / "media_rag_app.py")
SETTINGS = Settings(database_url="postgresql://test", minio_access_key="test", minio_secret_key="test",
                    twelvelabs_api_key="test", gemini_api_key="test")
ASSET = {"id": "asset", "title": "Episode", "status": "ready", "duration": 30, "kind": "audio",
         "stage": "Ready to search", "embedding_count": 2}
HIT = Hit("asset", "Episode", "original.wav", "audio", 30, 5, 15, 0.8, ("audio",))


def find(elements, label):
    return next(item for item in elements if item.label == label)


class AppTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()

    def run_with_library(self, assets=None, settings=SETTINGS):
        library = Mock()
        library.database.list_assets.return_value = [ASSET] if assets is None else assets
        library.retrieve.return_value = [HIT]
        library.storage.playback_url.return_value = "https://example.test/clip"
        library.answer.return_value = (
            {"status": "answered", "message": "", "claims": [{"text": "Supported answer", "source_ids": [1]}]},
            [Evidence(1, "Episode", "audio", 5, 15, "clip.mp3")],
        )
        self.enterContext(patch("media_rag.config.Settings.load", return_value=settings))
        self.enterContext(patch("media_rag.service.MediaLibrary", return_value=library))
        self.enterContext(patch("media_rag.media.check_tools"))
        app = AppTest.from_file(APP, default_timeout=15).run()
        self.assertFalse(app.exception)
        return app, library

    def test_unconfigured_app_explains_setup_without_provider_calls(self):
        app, library = self.run_with_library(settings=Settings())
        self.assertTrue(any("Set up your local media library" in item.value for item in app.info))
        library.initialize.assert_not_called()
        library.retrieve.assert_not_called()

    def test_empty_library_disables_search(self):
        app, library = self.run_with_library(assets=[])
        self.assertTrue(find(app.button, "Ask & cite").disabled)
        self.assertTrue(find(app.button, "Search only").disabled)
        library.retrieve.assert_not_called()

    def test_search_only_does_not_call_gemini_and_reruns_do_not_repeat_api_calls(self):
        app, library = self.run_with_library()
        find(app.text_area, "What would you like to know?").set_value("What was explained?")
        find(app.button, "Search only").click().run()
        self.assertFalse(app.exception)
        library.retrieve.assert_called_once()
        library.answer.assert_not_called()
        find(app.button, "Refresh library").click().run()
        library.retrieve.assert_called_once()
        library.answer.assert_not_called()

    def test_ask_renders_answer_and_original_timeline_citations(self):
        app, library = self.run_with_library()
        find(app.text_area, "What would you like to know?").set_value("What was explained?")
        find(app.button, "Ask & cite").click().run()
        self.assertFalse(app.exception)
        library.answer.assert_called_once()
        self.assertTrue(any("Supported answer" in item.value for item in app.markdown))
        self.assertTrue(any("00:05–00:15 in the original recording" in item.value for item in app.caption))
        app.run()
        library.answer.assert_called_once()

    def test_zero_hits_do_not_send_media_to_gemini(self):
        app, library = self.run_with_library()
        library.retrieve.return_value = []
        find(app.text_area, "What would you like to know?").set_value("Missing question")
        find(app.button, "Ask & cite").click().run()
        self.assertFalse(app.exception)
        library.answer.assert_not_called()
        self.assertTrue(any("No relevant moments" in item.value for item in app.info))

    def test_model_text_and_source_titles_cannot_create_markdown_images(self):
        dangerous = "![leak](https://example.test/image)"
        app, library = self.run_with_library()
        library.answer.return_value = (
            {"status": "answered", "claims": [{"text": dangerous, "source_ids": [1]}]},
            [Evidence(1, dangerous, "audio", 5, 15, "clip.mp3")],
        )
        find(app.text_area, "What would you like to know?").set_value(dangerous)
        find(app.button, "Ask & cite").click().run()
        self.assertFalse(app.exception)
        for item in list(app.markdown) + list(app.caption):
            self.assertNotIn(dangerous, item.value)
        self.assertTrue(any(r"\!\[leak\]" in item.value for item in app.markdown))

    def test_missing_gemini_key_still_allows_search(self):
        settings = Settings(database_url="postgresql://test", minio_access_key="test", minio_secret_key="test",
                            twelvelabs_api_key="test")
        app, _ = self.run_with_library(settings=settings)
        self.assertTrue(find(app.button, "Ask & cite").disabled)
        self.assertFalse(find(app.button, "Search only").disabled)

    def test_failed_job_can_be_retried(self):
        failed = {**ASSET, "status": "failed", "stage": "Indexing failed", "error": "Provider timeout"}
        app, library = self.run_with_library(assets=[failed])
        find(app.button, "Retry indexing").click().run()
        self.assertFalse(app.exception)
        library.database.retry.assert_called_once_with("asset")


if __name__ == "__main__":
    unittest.main()
