"""Canonical single-video URLs; never hand arbitrary user URLs to a downloader."""

from dataclasses import dataclass
import re
from urllib.parse import parse_qs, urlsplit

from media_rag.config import RagError


@dataclass(frozen=True)
class YouTubeSource:
    video_id: str
    url: str


def parse_youtube_url(value):
    message = "Enter an HTTPS YouTube video URL, such as https://www.youtube.com/watch?v=VIDEO_ID."
    if not isinstance(value, str) or len(value) > 2048 or any(ord(char) < 32 for char in value):
        raise RagError(message)
    try:
        url = urlsplit(value.strip())
        if url.scheme != "https" or url.username is not None or url.password is not None or url.port is not None:
            raise ValueError("invalid origin")
        host = (url.hostname or "").lower()
        path = url.path.rstrip("/")
        if host == "youtu.be":
            video_id = path.removeprefix("/")
        elif host in {"youtube.com", "www.youtube.com", "m.youtube.com", "music.youtube.com"}:
            if path == "/watch":
                ids = parse_qs(url.query, keep_blank_values=True).get("v", [])
                video_id = ids[0] if len(ids) == 1 else ""
            elif re.fullmatch(r"/(shorts|embed|live)/[A-Za-z0-9_-]{11}", path):
                video_id = path.rsplit("/", 1)[1]
            else:
                video_id = ""
        else:
            video_id = ""
        if not re.fullmatch(r"[A-Za-z0-9_-]{11}", video_id):
            raise ValueError("not a single video")
    except ValueError as exc:
        raise RagError(message) from exc
    return YouTubeSource(video_id, "https://www.youtube.com/watch?v=" + video_id)


def timestamp_url(value, start=0):
    source = parse_youtube_url(value)
    return source.url + f"&t={max(0, int(start))}s"
