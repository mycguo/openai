"""YouTube URL, bounded download, source deduplication, and retrieval regressions."""

import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import MagicMock, Mock, patch

from media_rag.config import MAX_UPLOAD_BYTES, RagError, Settings
from media_rag.models import Evidence, Hit, merge_hits
from youtube_rag.download import DownloadedVideo, download_video
from youtube_rag.download_worker import DownloadRejected, download, validate_metadata
from youtube_rag.service import YouTubeLibrary
from media_rag.youtube_urls import YouTubeSource, parse_youtube_url, timestamp_url


VIDEO_ID = "jNQXAC9IVRw"
URL = "https://www.youtube.com/watch?v=" + VIDEO_ID
SOURCE = parse_youtube_url(URL)
SETTINGS = Settings(twelvelabs_api_key="test", gemini_api_key="test")


class URLTests(unittest.TestCase):
    def test_supported_forms_canonicalize_to_one_video_and_strip_tracking_playlist_and_timestamps(self):
        for url in [URL + "&list=PL123&t=90", "https://youtu.be/" + VIDEO_ID + "?si=tracking",
                    "https://m.youtube.com/watch?v=" + VIDEO_ID,
                    "https://www.youtube.com/shorts/" + VIDEO_ID,
                    "https://www.youtube.com/embed/" + VIDEO_ID,
                    "https://www.youtube.com/live/" + VIDEO_ID]:
            with self.subTest(url=url):
                self.assertEqual(parse_youtube_url(url), SOURCE)
        self.assertEqual(timestamp_url(URL, 12.9), URL + "&t=12s")

    def test_non_youtube_and_ambiguous_urls_are_rejected_before_network_access(self):
        for url in ["http://127.0.0.1/private", "https://169.254.169.254/metadata", "file:///etc/passwd",
                    "https://youtube.com.evil.test/watch?v=" + VIDEO_ID,
                    "https://youtube.com@evil.test/watch?v=" + VIDEO_ID,
                    "https://evil.test@youtube.com/watch?v=" + VIDEO_ID,
                    "https://@youtube.com/watch?v=" + VIDEO_ID, URL + "&v=",
                    "https://youtube.com:443/watch?v=" + VIDEO_ID,
                    "https://youtube.com/playlist?list=PL123", URL + "&v=BaW_jenozKc",
                    "https://youtu.be/" + VIDEO_ID + "/extra", "https://youtube.com/watch?v=../secret",
                    URL + "\n", "https://[invalid", "https://youtube.com/redirect?q=" + URL]:
            with self.subTest(url=url), patch("youtube_rag.download.subprocess.Popen") as popen, self.assertRaises(RagError):
                download_video(YouTubeSource("unused", url), Path("/tmp"))
            popen.assert_not_called()


class WorkerTests(unittest.TestCase):
    def metadata(self, **extra):
        return {"id": VIDEO_ID, "title": "Example video", "duration": 19, "availability": "public", **extra}

    def test_metadata_rejects_live_restricted_long_or_oversize_media_before_download(self):
        cases = [({"id": "other"}, "invalid_video"), ({"_type": "playlist"}, "invalid_video"),
                 ({"is_live": True}, "live_video"), ({"live_status": "is_upcoming"}, "live_video"),
                 ({"duration": None}, "duration"), ({"duration": float("nan")}, "duration"),
                 ({"duration": 3601}, "duration"), ({"availability": "private"}, "restricted"),
                 ({"age_limit": 18}, "restricted"),
                 ({"requested_formats": [{"filesize": MAX_UPLOAD_BYTES}, {"filesize": 1}]}, "oversize")]
        for extra, code in cases:
            with self.subTest(extra=extra), self.assertRaisesRegex(DownloadRejected, code):
                validate_metadata(self.metadata(**extra), VIDEO_ID)

    def test_download_uses_raw_audio_video_and_checks_metadata_before_fetching_streams(self):
        factory = MagicMock()
        client = factory.return_value.__enter__.return_value = Mock()
        client.extract_info.return_value = self.metadata()
        client.process_ie_result.return_value = self.metadata()
        result = download(SOURCE, Path("/tmp/private-import"), factory)
        self.assertEqual(result, {"video_id": VIDEO_ID, "title": "Example video"})
        client.extract_info.assert_called_once_with(URL, download=False, ie_key="Youtube")
        client.process_ie_result.assert_called_once_with(client.extract_info.return_value, download=True)
        options = factory.call_args.args[0]
        self.assertIn("+ba", options["format"])
        self.assertEqual(options["outtmpl"], "/tmp/private-import/source.%(ext)s")
        self.assertTrue(options["noplaylist"])
        self.assertIsNone(options.get("max_downloads"))  # Raises even on the first successful video.
        self.assertFalse(options["enable_file_urls"])
        self.assertEqual(options["remote_components"], set())
        client.extract_info.return_value = self.metadata(is_live=True)
        client.process_ie_result.reset_mock()
        with self.assertRaises(DownloadRejected):
            download(SOURCE, Path("/tmp/private-import"), factory)
        client.process_ie_result.assert_not_called()


class DownloadTests(unittest.TestCase):
    def setUp(self):
        self.directory = Path(self.enterContext(tempfile.TemporaryDirectory()))

    def process(self, result, returncode=0):
        process = Mock(returncode=returncode, pid=123456)
        process.poll.return_value = returncode
        process.communicate.return_value = (json.dumps(result), "")
        self.enterContext(patch("youtube_rag.download.subprocess.Popen", return_value=process))
        return process

    def test_child_receives_no_provider_credentials_proxies_or_python_injection_variables(self):
        (self.directory / "source.mp4").write_bytes(b"video")
        process = self.process({"video_id": VIDEO_ID, "title": "Example"})
        with patch.dict(os.environ, {"PATH": "/usr/bin", "GEMINI_API_KEY": "private-key", "HTTPS_PROXY": "private-proxy",
                                     "DATABASE_URL": "private-db", "PYTHONPATH": "private-code"}, clear=True):
            result = download_video(SOURCE, self.directory)
        self.assertEqual(result.title, "Example")
        popen = subprocess.Popen
        self.assertEqual(popen.call_args.kwargs["env"], {"PATH": "/usr/bin", "YTDLP_NO_PLUGINS": "1"})
        self.assertTrue(popen.call_args.kwargs["start_new_session"])
        process.communicate.assert_called_once()

    def test_missing_empty_or_wrong_video_files_are_not_ingested(self):
        for payload, contents in [({"video_id": VIDEO_ID, "title": "Example"}, b""),
                                  ({"video_id": "other", "title": "Example"}, b"video")]:
            with self.subTest(payload=payload):
                (self.directory / "source.mp4").write_bytes(contents)
                self.process(payload)
                with self.assertRaises(RagError):
                    download_video(SOURCE, self.directory)

    def test_provider_failures_are_masked_and_no_private_payload_is_logged(self):
        self.process({"error": "unknown-private-key", "message": "private-key https://signed.example/token"}, 1)
        with self.assertLogs("youtube_rag.download") as logs, self.assertRaises(RagError) as raised:
            download_video(SOURCE, self.directory)
        self.assertNotIn("private-key", str(raised.exception) + " ".join(logs.output))

    def test_malformed_child_error_codes_still_produce_a_safe_message(self):
        self.process({"error": ["private-provider-message"]}, 1)
        with self.assertLogs("youtube_rag.download") as logs, self.assertRaisesRegex(RagError, "could not be downloaded"):
            download_video(SOURCE, self.directory)
        self.assertNotIn("private-provider", " ".join(logs.output))

    def test_timeout_kills_the_whole_download_process_group(self):
        process = self.process({})
        process.poll.return_value = None
        with patch("youtube_rag.download.time.monotonic", side_effect=[0, 601]), \
             patch("youtube_rag.download.os.killpg") as kill, self.assertRaisesRegex(RagError, "timed out"):
            download_video(SOURCE, self.directory)
        kill.assert_called_once()
        self.assertEqual(kill.call_args.args[0], process.pid)

    def test_disk_budget_stops_download_before_ingestion(self):
        process = self.process({})
        process.poll.return_value = None
        with patch("youtube_rag.download.directory_size", return_value=MAX_UPLOAD_BYTES * 3 + 1), \
             patch("youtube_rag.download.os.killpg") as kill, self.assertRaisesRegex(RagError, "200 MB"):
            download_video(SOURCE, self.directory)
        kill.assert_called_once()


class ServiceTests(unittest.TestCase):
    def library(self):
        database, storage = Mock(), Mock()
        database.find_youtube.return_value = None
        return YouTubeLibrary(SETTINGS, database, storage), database, storage

    def test_existing_video_is_reused_without_download_or_storage_calls(self):
        library, database, storage = self.library()
        database.find_youtube.return_value = {"id": "asset", "status": "ready"}
        with patch("youtube_rag.service.download_video") as downloader:
            asset, created = library.add_youtube("https://youtu.be/" + VIDEO_ID)
        self.assertFalse(created)
        self.assertEqual(asset["id"], "asset")
        downloader.assert_not_called()
        storage.upload.assert_not_called()

    def test_url_import_uses_same_native_upload_pipeline_and_saves_canonical_provenance(self):
        library, database, _ = self.library()
        folder = Path(self.enterContext(tempfile.TemporaryDirectory()))
        path = folder / "source.mp4"
        path.write_bytes(b"raw video bytes")
        database.link_youtube.return_value = ({"id": "canonical-asset", "status": "queued"}, True)
        observed = []

        def upload(recording, title):
            observed.append((recording.read(), title))
            return {"id": "asset"}, True

        with patch("youtube_rag.service.download_video", return_value=DownloadedVideo(path, "YouTube title")), \
             patch.object(library, "add_upload", side_effect=upload):
            result = library.add_youtube(URL + "&t=30&list=PL123")
        self.assertEqual(observed, [(b"raw video bytes", "YouTube title")])
        database.link_youtube.assert_called_once_with(VIDEO_ID, URL, "asset")
        self.assertEqual(result[0]["id"], "canonical-asset")

    def test_failed_download_does_not_save_a_source_or_call_paid_indexing(self):
        library, database, storage = self.library()
        with patch("youtube_rag.service.download_video", side_effect=RagError("YouTube blocked the download")), \
             self.assertRaises(RagError):
            library.add_youtube(URL)
        database.link_youtube.assert_not_called()
        database.create_asset.assert_not_called()
        storage.upload.assert_not_called()

    def test_search_is_scoped_to_ready_youtube_assets_even_without_explicit_selection(self):
        library, database, _ = self.library()
        database.list_youtube_assets.return_value = [{"id": "youtube-asset", "status": "ready"},
                                                    {"id": "pending", "status": "indexing"}]
        with patch("media_rag.service.MediaLibrary.retrieve", return_value=[]) as retrieve:
            library.retrieve("Question")
        self.assertEqual(retrieve.call_args.args[1], ["youtube-asset"])
        with self.assertRaisesRegex(RagError, "YouTube library"):
            library.retrieve("Question", ["file-upload-asset"])

    def test_youtube_source_survives_context_expansion_and_evidence_creation(self):
        library, _, storage = self.library()
        hit = Hit("asset", "Title", "source.mp4", "video", 30, 10, 20, 0.8, ("visual",), True, URL)
        hits = merge_hits([hit])
        self.assertEqual(hits[0].source_url, URL)
        storage.exists.return_value = True
        with patch("media_rag.service.Gemini") as gemini:
            gemini.return_value.answer.return_value = {"status": "insufficient_evidence", "claims": []}
            _, evidence = library.answer("Question", hits)
        self.assertEqual(evidence[0].source_url, URL)
        self.assertEqual(evidence[0].start, 2)


if __name__ == "__main__":
    unittest.main()
