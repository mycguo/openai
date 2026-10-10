"""Credential-free yt-dlp adapter. Only bounded metadata or error codes leave it."""

import json
import math
from pathlib import Path
import sys

from media_rag.config import MAX_UPLOAD_BYTES

from media_rag.youtube_urls import parse_youtube_url


MAX_DURATION = 3600


class DownloadRejected(Exception):
    pass


class QuietLogger:
    def debug(self, message):
        pass

    warning = debug
    error = debug


def validate_metadata(info, video_id):
    if not isinstance(info, dict) or info.get("id") != video_id or info.get("_type", "video") != "video":
        raise DownloadRejected("invalid_video")
    if info.get("is_live") or info.get("live_status") in {"is_live", "is_upcoming", "post_live"}:
        raise DownloadRejected("live_video")
    duration = info.get("duration")
    if type(duration) not in {int, float} or not math.isfinite(duration) or not 0 < duration <= MAX_DURATION:
        raise DownloadRejected("duration")
    if info.get("availability") not in {None, "public", "unlisted"} or info.get("age_limit", 0):
        raise DownloadRejected("restricted")
    formats = info.get("requested_formats") or [info]
    sizes = [item.get("filesize") or item.get("filesize_approx") or 0 for item in formats]
    if sum(sizes) > MAX_UPLOAD_BYTES:
        raise DownloadRejected("oversize")


def download(source, directory, ydl_factory=None):
    if ydl_factory is None:
        from yt_dlp import YoutubeDL
        ydl_factory = YoutubeDL
    options = {
        "format": "bv[height<=720][ext=mp4][protocol=https]+ba[ext=m4a][protocol=https]/"
                  "b[height<=720][ext=mp4][protocol=https]/bv[height<=720][ext=mp4][protocol=https]",
        "merge_output_format": "mp4",
        "outtmpl": str(directory / "source.%(ext)s"),
        # The fixed Youtube extractor and ID checks enforce one video. max_downloads
        # raises even after a successful download and breaks the two-stage flow.
        "noplaylist": True, "max_filesize": MAX_UPLOAD_BYTES,
        "socket_timeout": 20, "retries": 1, "fragment_retries": 1,
        "concurrent_fragment_downloads": 1,
        "cachedir": False, "quiet": True, "no_warnings": True, "logger": QuietLogger(),
        "allowed_extractors": ["youtube"], "enable_file_urls": False,
        "js_runtimes": {"deno": {}}, "remote_components": set(),
        "plugin_dirs": [], "proxy": "", "geo_bypass": False,
        "postprocessor_args": {"merger+ffmpeg_i": ["-protocol_whitelist", "file", "-enable_drefs", "0",
                                                    "-use_absolute_path", "0"]},
    }
    with ydl_factory(options) as ydl:
        info = ydl.extract_info(source.url, download=False, ie_key="Youtube")
        validate_metadata(info, source.video_id)
        result = ydl.process_ie_result(info, download=True)
    validate_metadata(result, source.video_id)
    return {"video_id": source.video_id, "title": str(result.get("title") or source.video_id)[:200]}


def main():
    try:
        source = parse_youtube_url(sys.argv[1])
        result = download(source, Path(sys.argv[2]))
    except DownloadRejected as exc:
        result = {"error": exc.args[0]}
    except ImportError:
        result = {"error": "missing_dependencies"}
    except Exception as exc:
        # Provider messages can contain signed URLs. Never return them or a traceback.
        message = str(exc).lower()
        if any(word in message for word in ("sign in", "bot", "http error 403", "http error 429")):
            code = "blocked"
        elif any(word in message for word in ("not available", "unavailable", "private video", "video has been removed")):
            code = "format" if "format" in message else "unavailable"
        else:
            code = "download_failed"
        result = {"error": code}
    print(json.dumps(result))
    return 1 if result.get("error") else 0


if __name__ == "__main__":
    sys.exit(main())
