"""Photo frontend behavior using mocked inference and infrastructure."""

import io
from pathlib import Path
import unittest
from unittest.mock import Mock, patch

from PIL import Image
import streamlit as st
from streamlit.testing.v1 import AppTest

from media_rag.config import RagError, Settings


APP = str(Path(__file__).resolve().parents[1] / "apps" / "ask_my_photo" / "app.py")
SETTINGS = Settings(database_url="postgresql://test", minio_access_key="test", minio_secret_key="test", gemini_api_key="test")
PHOTO = {"id": "photo-1", "title": "Holiday", "status": "ready", "width": 120, "height": 80,
         "image_key": "photos/1/image.jpg", "thumbnail_key": "photos/1/thumbnail.jpg",
         "object_key": "photos/1/original.png", "score": 0.9}
ANSWER = {"status": "answered", "message": "", "claims": [{"text": "A bicycle is visible.", "source_ids": [1]}]}


def find(elements, label):
    return next(item for item in elements if item.label == label)


class PhotoAppTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()

    def app(self, photos=None, settings=SETTINGS):
        self.library = Mock()
        self.library.database.list_photos.return_value = [PHOTO] if photos is None else photos
        self.library.database.counts.return_value = {"ready": 1} if photos is None else {}
        self.library.retrieve.return_value = [PHOTO]
        self.library.answer.return_value = ANSWER, []
        self.library.storage.playback_url.return_value = "https://example.test/private-photo"
        self.settings = self.enterContext(patch("photo_rag.app.load_settings", return_value=settings))
        self.enterContext(patch("photo_rag.app.PhotoLibrary", return_value=self.library))
        app = AppTest.from_file(APP, default_timeout=15).run()
        self.assertFalse(app.exception)
        return app

    def ask(self, app, label="Ask & cite"):
        find(app.text_area, "What would you like to know?").set_value("Which photo has a bicycle?")
        find(app.button, label).click().run()
        self.assertFalse(app.exception)

    def test_frontend_has_upload_gallery_and_cited_answers_without_sidebar(self):
        app = self.app()
        self.assertEqual(app.title[0].value, "Ask My Photo")
        self.assertTrue(app.get("file_uploader"))
        self.assertFalse(app.sidebar)
        self.ask(app)
        self.assertTrue(any("A bicycle is visible" in item.value for item in app.markdown))
        app.run()
        self.library.retrieve.assert_called_once()
        self.library.answer.assert_called_once()

    def test_no_ready_photos_disables_question_buttons(self):
        app = self.app(photos=[])
        self.assertTrue(find(app.button, "Ask & cite").disabled)
        self.assertTrue(find(app.button, "Find photos").disabled)
        self.library.retrieve.assert_not_called()

    def test_search_only_and_answer_retry_preserve_retrieval(self):
        app = self.app()
        self.ask(app, "Find photos")
        self.library.answer.assert_not_called()
        self.library.answer.side_effect = [RagError("Gemini quota reached."), (ANSWER, [])]
        find(app.button, "Answer from these photos").click().run()
        self.assertTrue(any("quota" in item.value for item in app.error))
        find(app.button, "Retry answer").click().run()
        self.assertFalse(app.error)
        self.library.retrieve.assert_called_once()
        self.assertEqual(self.library.answer.call_count, 2)

    def test_no_hits_do_not_generate_and_text_is_escaped(self):
        app = self.app()
        self.library.retrieve.return_value = []
        self.ask(app)
        self.library.answer.assert_not_called()
        self.assertTrue(any("No matching photos" in item.value for item in app.info))
        self.library.retrieve.return_value = [PHOTO]
        self.library.answer.return_value = {**ANSWER, "claims": [{"text": "![track](https://evil.example)", "source_ids": [1]}]}, []
        self.ask(app)
        self.assertTrue(any("\\!\\[track\\]" in item.value for item in app.markdown))

    def test_partial_secret_tables_merge_with_photo_overrides(self):
        with patch("streamlit.secrets", {"media_rag": {"MEDIA_RAG_STORAGE_PROVIDER": "neon", "GEMINI_API_KEY": "media"},
                                         "youtube_rag": {"GEMINI_API_KEY": "youtube"},
                                         "ask_my_photo": {"GEMINI_API_KEY": "photo"}}):
            self.app()
            self.settings.assert_called_with({"MEDIA_RAG_STORAGE_PROVIDER": "neon", "GEMINI_API_KEY": "photo"})

    def test_upload_and_index_happen_once_and_failed_photo_can_be_retried(self):
        content = io.BytesIO()
        Image.new("RGB", (10, 10), "red").save(content, format="PNG")
        content.name = "photo.png"
        self.enterContext(patch("streamlit.file_uploader", return_value=[content]))
        app = self.app(photos=[])
        self.library.add_upload.return_value = ({**PHOTO, "status": "queued"}, True)
        find(app.button, "Add photos").click().run()
        self.assertFalse(app.exception)
        self.library.add_upload.assert_called_once()
        self.library.index_photo.assert_called_once_with("photo-1", retry=False)
        app.run()
        self.library.add_upload.assert_called_once()

    def test_failed_indexing_is_visible_and_retry_targets_selected_photo(self):
        app = self.app(photos=[{**PHOTO, "status": "failed", "error": "Quota reached."}])
        find(app.button, "Retry indexing").click().run()
        self.assertFalse(app.exception)
        self.library.index_photo.assert_called_once_with("photo-1", retry=True)


if __name__ == "__main__":
    unittest.main()
