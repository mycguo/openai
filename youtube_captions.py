"""Caption retrieval without API credentials or paid audio transcription."""

import re
from urllib.parse import parse_qs, urlsplit

import requests
from youtube_transcript_api import YouTubeTranscriptApi
from youtube_transcript_api import (
    CouldNotRetrieveTranscript,
    NoTranscriptFound,
    RequestBlocked,
    TranscriptsDisabled,
    VideoUnavailable,
)


class CaptionError(RuntimeError):
    """A caption failure safe to display to the user."""


def video_id_from_url(value: str) -> str:
    """Accept only recognized YouTube video URLs, never arbitrary download URLs."""
    value = value.strip()
    if "://" not in value:
        value = "https://" + value
    try:
        url = urlsplit(value)
        if url.scheme not in {"https", "http"} or url.username or url.password or url.port:
            raise ValueError
        host = (url.hostname or "").lower()
        parts = url.path.strip("/").split("/")
        video_id = ""
        if host in {"youtu.be", "www.youtu.be"} and len(parts) == 1:
            video_id = parts[0]
        elif host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
            if url.path == "/watch":
                ids = parse_qs(url.query).get("v", [])
                if len(ids) == 1:
                    video_id = ids[0]
            elif len(parts) == 2 and parts[0] in {"shorts", "embed", "live"}:
                video_id = parts[1]
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError
        return video_id
    except ValueError:
        raise ValueError("Enter a valid YouTube video URL (watch, youtu.be, Shorts, or live).") from None


class TimeoutSession(requests.Session):
    def request(self, method, url, **kwargs):
        kwargs.setdefault("timeout", (10, 30))
        return super().request(method, url, **kwargs)


def fetch_captions(video_id: str, languages: list[str]):
    """Preserve all text and native timestamps; do not fall back to paid services."""
    try:
        with TimeoutSession() as session:
            transcript = YouTubeTranscriptApi(http_client=session).fetch(video_id, languages=languages)
        if not any(snippet.text.strip() for snippet in transcript):
            raise CaptionError("YouTube returned an empty caption track.")
        return transcript
    except NoTranscriptFound as exc:
        raise CaptionError("No captions match those language codes. Try the video's original language.") from exc
    except TranscriptsDisabled as exc:
        raise CaptionError("Captions are disabled for this video. You can explicitly choose audio transcription instead.") from exc
    except RequestBlocked as exc:
        raise CaptionError("YouTube blocked the caption request from this network. Try again later or run locally; repeated retries may not help.") from exc
    except VideoUnavailable as exc:
        raise CaptionError("This video is unavailable. Check that it is accessible without signing in.") from exc
    except (CouldNotRetrieveTranscript, requests.RequestException) as exc:
        raise CaptionError("Could not retrieve captions. Check the video and your connection, then try again later.") from exc
