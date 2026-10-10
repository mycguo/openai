"""Streamlit UI regressions with local services and providers substituted."""

from pathlib import Path
import io
import unittest
from unittest.mock import Mock, patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from media_rag.config import RagError, Settings
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

    def test_answer_failure_preserves_search_and_retry_does_not_repeat_retrieval(self):
        app, library = self.run_with_library()
        library.answer.side_effect = RagError("Gemini could not generate a cited answer (HTTP 429). Check its quota.")
        find(app.text_area, "What would you like to know?").set_value("What was explained?")
        find(app.button, "Ask & cite").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("HTTP 429" in item.value for item in app.error))
        self.assertTrue(any("Search found relevant clips" in item.value for item in app.info))
        self.assertTrue(any(item.label == "Episode · 00:05–00:15" for item in app.expander))
        find(app.button, "Refresh library").click().run()
        self.assertTrue(any("HTTP 429" in item.value for item in app.error))
        library.retrieve.assert_called_once()
        library.answer.assert_called_once()
        library.answer.side_effect = None
        find(app.button, "Retry cited answer").click().run()
        self.assertFalse(app.exception)
        self.assertFalse(app.error)
        self.assertTrue(any("Supported answer" in item.value for item in app.markdown))
        library.retrieve.assert_called_once()
        self.assertEqual(library.answer.call_count, 2)
        self.assertEqual(library.answer.call_args.args, ("What was explained?", [HIT]))
        app.run()
        self.assertEqual(library.answer.call_count, 2)

    def test_failed_retry_remains_visible_and_new_search_clears_old_answer_error(self):
        app, library = self.run_with_library()
        library.answer.side_effect = RuntimeError("private-key")
        find(app.text_area, "What would you like to know?").set_value("What was explained?")
        find(app.button, "Ask & cite").click().run()
        find(app.button, "Retry cited answer").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(app.error)
        self.assertFalse(any("private-key" in item.value for item in app.error))
        library.retrieve.assert_called_once()
        self.assertEqual(library.answer.call_count, 2)
        find(app.text_area, "What would you like to know?").set_value("A different question")
        find(app.button, "Search only").click().run()
        self.assertFalse(app.error)
        self.assertTrue(find(app.button, "Answer this search with Gemini"))
        self.assertEqual(library.retrieve.call_count, 2)
        self.assertEqual(library.answer.call_count, 2)

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
        library.index_recording.assert_called_once()
        self.assertEqual(library.index_recording.call_args.args, ("asset",))
        self.assertTrue(library.index_recording.call_args.kwargs["retry"])

    def test_queued_recording_indexes_in_streamlit_and_reruns_do_not_repeat_it(self):
        queued = {**ASSET, "status": "queued", "stage": "Waiting for the indexing worker"}
        app, library = self.run_with_library(assets=[queued])
        self.assertTrue(any(item.value == "Ready to index" for item in app.caption))
        library.index_recording.side_effect = lambda *args, **kwargs: setattr(
            library.database.list_assets, "return_value", [ASSET])
        find(app.button, "Index recording").click().run()
        self.assertFalse(app.exception)
        library.index_recording.assert_called_once()
        self.assertEqual(library.index_recording.call_args.args, ("asset",))
        self.assertFalse(library.index_recording.call_args.kwargs["retry"])
        self.assertTrue(any("ready to search" in item.value for item in app.success))
        find(app.button, "Refresh library").click().run()
        library.index_recording.assert_called_once()

    def test_interrupted_recording_can_be_resumed(self):
        interrupted = {**ASSET, "status": "indexing", "stage": "Creating native embeddings"}
        app, library = self.run_with_library(assets=[interrupted])
        find(app.button, "Resume indexing").click().run()
        self.assertFalse(app.exception)
        library.index_recording.assert_called_once()
        self.assertFalse(library.index_recording.call_args.kwargs["retry"])

    def test_indexing_failure_stays_visible_without_automatic_retry_on_rerun(self):
        queued = {**ASSET, "status": "queued"}
        app, library = self.run_with_library(assets=[queued])

        def fail(*args, **kwargs):
            library.database.list_assets.return_value = [{**ASSET, "status": "failed", "error": "Provider timeout"}]
            raise RagError("TwelveLabs returned HTTP 429. Wait, then retry.")

        library.index_recording.side_effect = fail
        find(app.button, "Index recording").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("HTTP 429" in item.value for item in app.error))
        self.assertTrue(any(item.state == "error" for item in app.status))
        self.assertTrue(find(app.button, "Retry indexing"))
        app.run()
        library.index_recording.assert_called_once()

    def test_missing_twelvelabs_key_disables_upload_index_and_retry(self):
        settings = Settings(database_url="postgresql://test", minio_access_key="test", minio_secret_key="test")
        queued = {**ASSET, "id": "queued", "status": "queued"}
        failed = {**ASSET, "id": "failed", "status": "failed", "error": "Provider timeout"}
        app, library = self.run_with_library(assets=[queued, failed], settings=settings)
        for label in ["Upload & index", "Index recording", "Retry indexing"]:
            self.assertTrue(find(app.button, label).disabled)
        library.index_recording.assert_not_called()

    def test_new_upload_is_indexed_immediately_and_becomes_searchable(self):
        uploaded = io.BytesIO(b"recording")
        uploaded.name = "recording.mp3"
        self.enterContext(patch("streamlit.file_uploader", return_value=uploaded))
        app, library = self.run_with_library(assets=[])
        library.add_upload.return_value = ({**ASSET, "status": "queued"}, True)
        library.index_recording.side_effect = lambda *args, **kwargs: setattr(
            library.database.list_assets, "return_value", [ASSET])
        find(app.text_input, "Recording title").set_value("New episode")
        find(app.button, "Upload & index").click().run()
        self.assertFalse(app.exception)
        library.add_upload.assert_called_once_with(uploaded, "New episode")
        library.index_recording.assert_called_once()
        self.assertFalse(find(app.button, "Search only").disabled)
        app.run()
        library.add_upload.assert_called_once()
        library.index_recording.assert_called_once()

    def test_upload_of_ready_duplicate_does_not_index_again(self):
        uploaded = io.BytesIO(b"recording")
        uploaded.name = "recording.mp3"
        self.enterContext(patch("streamlit.file_uploader", return_value=uploaded))
        app, library = self.run_with_library()
        library.add_upload.return_value = (ASSET, False)
        find(app.button, "Upload & index").click().run()
        self.assertFalse(app.exception)
        library.index_recording.assert_not_called()
        self.assertTrue(any("already in the library" in item.value for item in app.info))


if __name__ == "__main__":
    unittest.main()
