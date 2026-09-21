"""Offline tests: no YouTube traffic, credentials, or paid API calls."""

import unittest
from pathlib import Path
from unittest.mock import patch

from streamlit.testing.v1 import AppTest
from youtube_transcript_api import FetchedTranscript, FetchedTranscriptSnippet, RequestBlocked

from youtube_captions import CaptionError, fetch_captions, video_id_from_url


APP = str(Path(__file__).resolve().parents[1] / "youtube.py")
VIDEO_ID = "dQw4w9WgXcQ"


def sample_transcript():
    return FetchedTranscript(
        snippets=[FetchedTranscriptSnippet(text="Beginning " + "word " * 8000, start=0, duration=59.9996),
                  FetchedTranscriptSnippet(text="THE FINAL WORDS", start=59.9996, duration=1)],
        video_id=VIDEO_ID, language="English", language_code="en", is_generated=False,
    )


class CaptionTests(unittest.TestCase):
    def test_url_forms(self):
        for url in [f"https://www.youtube.com/watch?v={VIDEO_ID}&t=20",
                    f"https://youtu.be/{VIDEO_ID}?si=example", f"youtube.com/shorts/{VIDEO_ID}",
                    f"https://m.youtube.com/live/{VIDEO_ID}", f"https://youtube.com/embed/{VIDEO_ID}"]:
            with self.subTest(url=url):
                self.assertEqual(video_id_from_url(url), VIDEO_ID)

    def test_invalid_urls(self):
        for url in ["", "file:///etc/passwd", "https://example.com/watch?v=" + VIDEO_ID,
                    "https://youtube.com.evil.com/watch?v=" + VIDEO_ID,
                    "https://youtube.com@127.0.0.1/watch?v=" + VIDEO_ID,
                    "https://youtube.com/playlist?list=abc", "https://youtube.com/watch?v=short",
                    "https://youtube.com:8080/watch?v=" + VIDEO_ID]:
            with self.subTest(url=url), self.assertRaises(ValueError):
                video_id_from_url(url)

    @patch("youtube_captions.YouTubeTranscriptApi")
    def test_fetch_preserves_full_text(self, api):
        api.return_value.fetch.return_value = sample_transcript()
        result = fetch_captions(VIDEO_ID, ["es", "en"])
        api.return_value.fetch.assert_called_once_with(VIDEO_ID, languages=["es", "en"])
        self.assertGreater(len(result[0].text), 15000)
        self.assertEqual(result[-1].text, "THE FINAL WORDS")

    @patch("youtube_captions.YouTubeTranscriptApi")
    def test_blocked_is_actionable(self, api):
        api.return_value.fetch.side_effect = RequestBlocked(VIDEO_ID)
        with self.assertRaisesRegex(CaptionError, "blocked"):
            fetch_captions(VIDEO_ID, ["en"])

    @patch("youtube_captions.YouTubeTranscriptApi")
    def test_empty_track(self, api):
        api.return_value.fetch.return_value = []
        with self.assertRaisesRegex(CaptionError, "empty"):
            fetch_captions(VIDEO_ID, ["en"])

    def test_ui_without_secrets(self):
        with patch("streamlit.runtime.secrets.Secrets.get", side_effect=FileNotFoundError), patch.dict("os.environ", {}, clear=True):
            app = AppTest.from_file(APP).run()
        self.assertFalse(app.exception)
        self.assertFalse(app.button[0].disabled)

    @patch("youtube_captions.fetch_captions", return_value=sample_transcript())
    def test_ui_keeps_full_transcript_for_download_without_rendering_it(self, fetch):
        app = AppTest.from_file(APP).run()
        app.text_input[1].set_value(f"https://youtu.be/{VIDEO_ID}")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertFalse(any(area.label == "Transcript" for area in app.text_area))
        self.assertIn("THE FINAL WORDS", app.session_state["yt_transcript_text"])
        self.assertGreater(len(app.session_state["yt_transcript_text"]), 40000)
        self.assertTrue(any(button.label == "💾 Download as Text" for button in app.get("download_button")))
        self.assertTrue(any(button.label == "💾 Download as SRT" for button in app.get("download_button")))
        self.assertIn("00:01:00,000", app.session_state["yt_transcript_srt"])
        self.assertNotIn(",1000", app.session_state["yt_transcript_srt"])
        app.text_input[1].set_value("https://example.com").run()
        self.assertFalse(any(area.label == "Transcript" for area in app.text_area))
        app.button[0].click().run()
        self.assertTrue(app.error)
        fetch.assert_called_once()

    @patch("youtube_captions.fetch_captions", side_effect=CaptionError("YouTube blocked this request"))
    def test_ui_error_no_paid_fallback(self, fetch):
        app = AppTest.from_file(APP).run()
        app.text_input[1].set_value(f"https://youtu.be/{VIDEO_ID}")
        app.button[0].click().run()
        self.assertFalse(app.exception)
        self.assertIn("blocked", app.error[0].value)
        self.assertFalse(any(area.label == "Transcript" for area in app.text_area))

    def test_audio_requires_consent(self):
        app = AppTest.from_file(APP).run()
        app.radio[0].set_value("Transcribe audio with AssemblyAI").run()
        self.assertTrue(app.button[0].disabled)
        self.assertFalse(app.checkbox[0].value)


if __name__ == "__main__":
    unittest.main()
