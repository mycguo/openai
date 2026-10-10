"""Separate URL-based Streamlit entrypoint; providers and infrastructure are mocked."""

from pathlib import Path
import unittest
from unittest.mock import Mock, patch

import streamlit as st
from streamlit.testing.v1 import AppTest

from media_rag.config import RagError, Settings
from media_rag.models import Evidence, Hit


APP = str(Path(__file__).resolve().parents[1] / "apps" / "youtube_rag" / "app.py")
URL = "https://www.youtube.com/watch?v=jNQXAC9IVRw"
SETTINGS = Settings(database_url="postgresql://test", minio_access_key="test", minio_secret_key="test",
                    twelvelabs_api_key="test", gemini_api_key="test")
ASSET = {"id": "youtube-asset", "title": "YouTube video", "status": "ready", "duration": 30, "kind": "video",
         "stage": "Ready to search", "embedding_count": 2, "source_url": URL}
HIT = Hit("youtube-asset", "YouTube video", "original.mp4", "video", 30, 5, 15, 0.8, ("audio",), True, URL)


def find(elements, label):
    return next(item for item in elements if item.label == label)


class YouTubeAppTests(unittest.TestCase):
    def setUp(self):
        st.cache_resource.clear()

    def app(self, assets=None, settings=SETTINGS):
        library = Mock()
        library.database.list_youtube_assets.return_value = [ASSET] if assets is None else assets
        library.retrieve.return_value = [HIT]
        library.storage.playback_url.return_value = "https://example.test/clip"
        library.answer.return_value = (
            {"status": "answered", "message": "", "claims": [{"text": "Cited answer", "source_ids": [1]}]},
            [Evidence(1, "YouTube video", "video", 5, 15, "clip.mp4", URL)],
        )
        self.enterContext(patch("media_rag.config.Settings.load", return_value=settings))
        self.enterContext(patch("youtube_rag.service.YouTubeLibrary", return_value=library))
        self.enterContext(patch("media_rag.media.check_tools"))
        app = AppTest.from_file(APP, default_timeout=15).run()
        self.assertFalse(app.exception)
        return app, library

    def test_new_app_has_url_input_and_only_lists_youtube_assets(self):
        app, library = self.app()
        self.assertEqual(app.title[0].value, "youtube-rag")
        self.assertTrue(find(app.text_input, "YouTube URL"))
        self.assertFalse(app.get("file_uploader"))
        self.assertTrue(find(app.button, "Import & index"))
        library.database.list_assets.assert_not_called()

    def test_import_indexes_immediately_and_reruns_do_not_repeat_it(self):
        app, library = self.app(assets=[])
        library.add_youtube.return_value = ({**ASSET, "status": "queued"}, True)
        library.index_recording.side_effect = lambda *args, **kwargs: setattr(
            library.database.list_youtube_assets, "return_value", [ASSET])
        find(app.text_input, "YouTube URL").set_value(URL)
        find(app.button, "Import & index").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(library.add_youtube.call_args.args, (URL, ""))
        library.index_recording.assert_called_once_with("youtube-asset", on_progress=unittest.mock.ANY, retry=False)
        self.assertFalse(find(app.button, "Ask & cite").disabled)
        find(app.button, "Refresh library").click().run()
        library.add_youtube.assert_called_once()
        library.index_recording.assert_called_once()

    def test_ready_duplicate_does_not_index_again(self):
        app, library = self.app()
        library.add_youtube.return_value = (ASSET, False)
        find(app.text_input, "YouTube URL").set_value(URL)
        find(app.button, "Import & index").click().run()
        self.assertFalse(app.exception)
        library.index_recording.assert_not_called()

    def test_download_error_does_not_start_indexing_and_is_visible(self):
        app, library = self.app()
        library.add_youtube.side_effect = RagError("YouTube blocked the download from this server.")
        find(app.text_input, "YouTube URL").set_value(URL)
        find(app.button, "Import & index").click().run()
        self.assertFalse(app.exception)
        self.assertTrue(any("YouTube blocked" in item.value for item in app.error))
        library.index_recording.assert_not_called()

    def test_ask_reuses_citation_flow_and_links_to_original_youtube_timestamp(self):
        app, library = self.app()
        find(app.text_area, "What would you like to know?").set_value("What was explained?")
        find(app.button, "Ask & cite").click().run()
        self.assertFalse(app.exception)
        self.assertEqual(library.retrieve.call_args.args[1], ["youtube-asset"])
        self.assertTrue(any("Cited answer" in item.value for item in app.markdown))
        self.assertTrue(any(item.proto.url == URL + "&t=5s" for item in app.get("link_button")))
        app.run()
        library.answer.assert_called_once()

    def test_youtube_secret_table_is_used_and_missing_key_disables_import(self):
        settings = Settings(database_url="postgresql://test", minio_access_key="test", minio_secret_key="test")
        with patch("streamlit.secrets", {"youtube_rag": {"GEMINI_API_KEY": "youtube-test"},
                                         "media_rag": {"GEMINI_API_KEY": "media-test"}}):
            app, _ = self.app(settings=settings)
            Settings.load.assert_called_with({"GEMINI_API_KEY": "youtube-test"})
        self.assertTrue(find(app.button, "Import & index").disabled)


if __name__ == "__main__":
    unittest.main()
