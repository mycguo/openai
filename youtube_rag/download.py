"""Bounded YouTube imports in a child process without application credentials."""

from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import signal
import subprocess
import sys
import time

from media_rag.config import MAX_UPLOAD_BYTES, RagError
from media_rag.media import subprocess_environment

from media_rag.youtube_urls import parse_youtube_url


DOWNLOAD_TIMEOUT = 600
logger = logging.getLogger(__name__)
MAX_TEMP_BYTES = MAX_UPLOAD_BYTES * 3
ERRORS = {
    "invalid_video": "YouTube did not return the requested video.",
    "live_video": "Live and upcoming streams cannot be imported. Use a completed video.",
    "duration": "Import a video with a known duration of at most 60 minutes.",
    "restricted": "This video requires an account or is restricted. Use a publicly accessible video.",
    "oversize": "This video exceeds the 200 MB import limit. Use a shorter video.",
    "blocked": "YouTube blocked the download from this server. Retry later or run youtube-rag locally.",
    "unavailable": "This YouTube video is unavailable. Check the URL or choose another video.",
    "format": "No compatible MP4 stream was available. Update yt-dlp or try another video.",
    "missing_dependencies": "Install requirements-youtube-rag.txt, including yt-dlp and Deno, then restart.",
    "download_failed": "The YouTube video could not be downloaded. Check that it is accessible and retry.",
}


@dataclass(frozen=True)
class DownloadedVideo:
    path: Path
    title: str


def directory_size(directory):
    total = 0
    for path in directory.rglob("*"):
        try:
            if path.is_file():
                total += path.stat().st_size
        except FileNotFoundError:
            pass  # A completed download may remove its intermediate files.
    return total


def stop_process(process):
    # Include FFmpeg and the JavaScript runtime, not just their Python parent.
    try:
        os.killpg(process.pid, signal.SIGKILL)
    except ProcessLookupError:
        pass
    process.communicate(timeout=10)


def download_video(source, directory):
    # Revalidate at the network boundary, including callers constructing a source directly.
    source = parse_youtube_url(source.url)
    directory = Path(directory).resolve()
    process = subprocess.Popen(
        [sys.executable, "-m", "youtube_rag.download_worker", source.url, str(directory)],
        cwd=Path(__file__).resolve().parents[1], env={**subprocess_environment(), "YTDLP_NO_PLUGINS": "1"},
        stdout=subprocess.PIPE, stderr=subprocess.DEVNULL, text=True, start_new_session=True,
    )
    deadline = time.monotonic() + DOWNLOAD_TIMEOUT
    try:
        while process.poll() is None:
            if time.monotonic() >= deadline:
                raise RagError("The YouTube download timed out. Try a shorter video or retry later.")
            if directory_size(directory) > MAX_TEMP_BYTES:
                raise RagError(ERRORS["oversize"])
            time.sleep(0.2)
        output, _ = process.communicate(timeout=10)
    except BaseException:
        stop_process(process)
        raise
    try:
        result = json.loads(output)
        if not isinstance(result, dict):
            raise ValueError("invalid result")
    except (ValueError, TypeError) as exc:
        raise RagError(ERRORS["download_failed"]) from exc
    if process.returncode or result.get("error"):
        code = result.get("error")
        if not isinstance(code, str) or code not in ERRORS:
            code = "download_failed"
        logger.warning("YouTube import failed: error_code=%s", code)
        raise RagError(ERRORS[code])
    path = directory / "source.mp4"
    if path.is_symlink() or not path.is_file() or not 0 < path.stat().st_size <= MAX_UPLOAD_BYTES:
        raise RagError(ERRORS["oversize"] if path.is_file() and path.stat().st_size > MAX_UPLOAD_BYTES
                       else ERRORS["download_failed"])
    title = result.get("title")
    if result.get("video_id") != source.video_id or not isinstance(title, str) or not title.strip():
        raise RagError(ERRORS["invalid_video"])
    return DownloadedVideo(path, title.strip()[:200])
