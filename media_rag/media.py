"""Probe uploads and extract playable evidence with bounded FFmpeg processes."""

import json
import math
import os
from pathlib import Path
import shutil
import subprocess

from .config import RagError


SUPPORTED_EXTENSIONS = {".mp3", ".wav", ".mp4", ".mov", ".webm"}
MIME_TYPES = {
    ".mp3": "audio/mpeg", ".wav": "audio/wav", ".mp4": "video/mp4",
    ".mov": "video/quicktime", ".webm": "video/webm",
}
INPUT_FORMATS = {".mp3": "mp3", ".wav": "wav", ".mp4": "mov", ".mov": "mov", ".webm": "matroska"}


def input_options(path):
    extension = Path(path).suffix.lower()
    if extension not in INPUT_FORMATS:
        raise RagError("Supported formats: MP3, WAV, MP4, MOV, and WebM.")
    # Do not auto-detect playlists or follow network/file references from uploads.
    options = ["-protocol_whitelist", "file", "-f", INPUT_FORMATS[extension]]
    if INPUT_FORMATS[extension] == "mov":
        options += ["-enable_drefs", "0", "-use_absolute_path", "0"]
    return options


def subprocess_environment():
    # Media parsers do not need provider keys, database passwords, or proxy settings.
    return {name: value for name, value in os.environ.items()
            if name in {"PATH", "LANG", "LC_ALL", "SYSTEMROOT"}}


def check_tools():
    if not shutil.which("ffmpeg") or not shutil.which("ffprobe"):
        raise RagError("Install FFmpeg (including ffprobe), then restart the app and worker.")


def probe(path):
    check_tools()
    try:
        result = subprocess.run(
            ["ffprobe", "-v", "error", *input_options(path),
             "-show_format", "-show_streams", "-of", "json", str(path)],
            capture_output=True, text=True, timeout=30, check=True, env=subprocess_environment(),
        )
        info = json.loads(result.stdout)
        duration = float(info["format"]["duration"])
        streams = info.get("streams", [])
    except (subprocess.SubprocessError, ValueError, KeyError) as exc:
        raise RagError("The file is unreadable or has no usable media duration.") from exc
    if not math.isfinite(duration) or duration <= 0:
        raise RagError("The file must contain a finite, positive media duration.")
    has_audio = any(stream.get("codec_type") == "audio" for stream in streams)
    has_video = any(stream.get("codec_type") == "video"
                    and not stream.get("disposition", {}).get("attached_pic", 0) for stream in streams)
    if not has_audio and not has_video:
        raise RagError("Upload a file containing audio or video.")
    extension = Path(path).suffix.lower()
    if extension not in SUPPORTED_EXTENSIONS:
        raise RagError("Supported formats: MP3, WAV, MP4, MOV, and WebM.")
    kind = "video" if has_video else "audio"
    if MIME_TYPES[extension].split("/")[0] != kind:
        raise RagError("The file extension does not match its audio/video content.")
    return {"duration": duration, "kind": kind, "has_audio": has_audio,
            "mime_type": MIME_TYPES[extension]}


def extract_clip(source, destination, start, end, kind, has_audio=True):
    check_tools()
    if not all(math.isfinite(value) for value in (start, end)) or not 0 <= start < end:
        raise RagError("Invalid evidence time range.")
    if end - start > 90.01:
        raise RagError("Evidence clips must be at most 90 seconds long.")
    command = ["ffmpeg", "-nostdin", "-v", "error", "-y", *input_options(source), "-i", str(source),
               "-ss", str(start), "-t", str(end - start)]
    if kind == "video":
        command += ["-map", "0:v:0", "-map", "0:a:0?", "-c:v", "libx264",
                    "-preset", "fast", "-crf", "23", "-pix_fmt", "yuv420p",
                    "-vf", "scale='trunc(min(1280,iw)/2)*2':-2", "-c:a", "aac", "-b:a", "128k",
                    "-movflags", "+faststart"]
    else:
        command += ["-vn", "-map", "0:a:0", "-c:a", "libmp3lame", "-b:a", "128k"]
    command.append(str(destination))
    try:
        subprocess.run(command, capture_output=True, timeout=180, check=True, env=subprocess_environment())
    except subprocess.SubprocessError as exc:
        raise RagError("FFmpeg could not extract this evidence clip.") from exc
