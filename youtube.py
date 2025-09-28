import os
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests
import streamlit as st
import yt_dlp


st.set_page_config(
    page_title="YouTube Transcript Extractor",
    page_icon="📺",
    layout="wide"
)


ASSEMBLYAI_API_KEY = st.secrets.get("ASSEMBLYAI_API_KEY")

UPLOAD_ENDPOINT = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_ENDPOINT = "https://api.assemblyai.com/v2/transcript"
CHUNK_SIZE = 5_242_880  # 5 MB


@dataclass
class TranscriptSegment:
    index: int
    start: float
    end: float
    text: str


def _download_audio(url: str, workdir: str) -> tuple[str, dict]:
    """Download best audio track with yt_dlp and return file path and metadata."""
    output_template = os.path.join(workdir, "%(id)s.%(ext)s")
    ydl_opts = {
        "format": "bestaudio/best",
        "postprocessors": [
            {
                "key": "FFmpegExtractAudio",
                "preferredcodec": "mp3",
                "preferredquality": "192",
            }
        ],
        "outtmpl": output_template,
        "quiet": True,
        "no_warnings": True,
    }

    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        info = ydl.extract_info(url, download=True)
        base_filename = ydl.prepare_filename(info)

    audio_path = os.path.splitext(base_filename)[0] + ".mp3"
    if not os.path.exists(audio_path):
        raise FileNotFoundError("Audio download failed; file not found after processing.")

    return audio_path, info


def _read_file_in_chunks(filepath: str, chunk_size: int = CHUNK_SIZE):
    with open(filepath, "rb") as file_stream:
        while True:
            data = file_stream.read(chunk_size)
            if not data:
                break
            yield data


def _upload_audio(filepath: str) -> str:
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    response = requests.post(UPLOAD_ENDPOINT, headers=headers, data=_read_file_in_chunks(filepath))
    response.raise_for_status()
    return response.json()["upload_url"]


def _request_transcription(audio_url: str) -> str:
    headers = {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/json",
    }
    payload = {
        "audio_url": audio_url,
        "auto_highlights": False,
        "speaker_labels": False,
        "language_detection": False,
        "punctuate": True,
    }

    response = requests.post(TRANSCRIPT_ENDPOINT, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["id"]


def _poll_transcription(transcript_id: str, placeholder) -> dict:
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    polling_url = f"{TRANSCRIPT_ENDPOINT}/{transcript_id}"

    while True:
        response = requests.get(polling_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        status = data.get("status", "")
        placeholder.info(f"Transcription status: {status}")

        if status == "completed":
            placeholder.empty()
            return data
        if status == "error":
            placeholder.empty()
            raise RuntimeError(data.get("error", "Unknown transcription error"))

        time.sleep(3)


def _segments_from_words(words: Optional[Iterable[dict]], max_duration: float = 10.0, max_words: int = 40) -> List[TranscriptSegment]:
    words = list(words or [])
    if not words:
        return []

    segments: List[TranscriptSegment] = []
    current_words: List[str] = []
    segment_start = words[0]["start"] / 1000.0
    segment_end = segment_start

    for idx, word in enumerate(words, start=1):
        token = word.get("text", "").strip()
        if not token:
            continue

        if not current_words:
            segment_start = word.get("start", 0) / 1000.0

        current_words.append(token)
        segment_end = word.get("end", 0) / 1000.0

        duration = segment_end - segment_start
        if duration >= max_duration or len(current_words) >= max_words:
            segments.append(
                TranscriptSegment(
                    index=len(segments) + 1,
                    start=segment_start,
                    end=segment_end,
                    text=" ".join(current_words),
                )
            )
            current_words = []

    if current_words:
        segments.append(
            TranscriptSegment(
                index=len(segments) + 1,
                start=segment_start,
                end=segment_end,
                text=" ".join(current_words),
            )
        )

    return segments


def _segments_to_srt(segments: Iterable[TranscriptSegment]) -> str:
    def format_timestamp(seconds: float) -> str:
        hours = int(seconds // 3600)
        minutes = int((seconds % 3600) // 60)
        secs = int(seconds % 60)
        millis = int(round((seconds - int(seconds)) * 1000))
        return f"{hours:02}:{minutes:02}:{secs:02},{millis:03}"

    lines: List[str] = []
    for segment in segments:
        lines.append(str(segment.index))
        lines.append(f"{format_timestamp(segment.start)} --> {format_timestamp(segment.end)}")
        lines.append(segment.text)
        lines.append("")

    return "\n".join(lines).strip()


def _clear_state():
    for key in [
        "yt_transcript_text",
        "yt_transcript_srt",
        "yt_transcript_metadata",
    ]:
        st.session_state.pop(key, None)


def main() -> None:
    st.title("📺 YouTube Transcript Extractor")
    st.markdown("Download a YouTube video's audio, transcribe it with AssemblyAI, and save the results.")

    if not ASSEMBLYAI_API_KEY:
        st.error("Missing ASSEMBLYAI_API_KEY in Streamlit secrets. Please configure it to continue.")
        return

    default_url = "https://www.youtube.com/watch?v=c_w0LaFahxk"
    url = st.text_input(
        "YouTube URL",
        value=default_url,
        placeholder="Paste a YouTube video link",
        help="The audio will be downloaded and transcribed via AssemblyAI.",
    )

    col1, col2 = st.columns([1, 1])
    with col1:
        fetch_clicked = st.button("🎬 Fetch Transcript", type="primary")
    with col2:
        if st.button("🧹 Clear Transcript"):
            _clear_state()
            st.rerun()

    if fetch_clicked:
        if not url or not url.strip():
            st.error("Please provide a valid YouTube URL.")
        else:
            with st.spinner("Downloading audio and requesting transcription..."):
                try:
                    with tempfile.TemporaryDirectory() as tmpdir:
                        audio_path, metadata = _download_audio(url.strip(), tmpdir)
                        upload_url = _upload_audio(audio_path)
                        transcript_id = _request_transcription(upload_url)

                        status_placeholder = st.empty()
                        transcript_data = _poll_transcription(transcript_id, status_placeholder)

                    text = transcript_data.get("text", "").strip()
                    if not text:
                        raise ValueError("Transcription completed but returned empty text.")

                    words = transcript_data.get("words", [])
                    segments = _segments_from_words(words)
                    if not segments:
                        segments = [
                            TranscriptSegment(index=1, start=0.0, end=max(len(text.split()) / 2, 1.0), text=text)
                        ]

                    srt_text = _segments_to_srt(segments)

                    st.session_state.yt_transcript_text = text
                    st.session_state.yt_transcript_srt = srt_text
                    st.session_state.yt_transcript_metadata = {
                        "title": metadata.get("title"),
                        "uploader": metadata.get("uploader"),
                        "duration": metadata.get("duration"),
                        "webpage_url": metadata.get("webpage_url"),
                    }

                    st.success("Transcript ready!")

                except Exception as exc:
                    st.error(f"Failed to generate transcript: {exc}")

    if st.session_state.get("yt_transcript_text"):
        metadata = st.session_state.get("yt_transcript_metadata", {}) or {}
        if metadata:
            st.caption(
                " | ".join(
                    filter(
                        None,
                        [
                            f"Title: {metadata.get('title')}" if metadata.get("title") else None,
                            f"Uploader: {metadata.get('uploader')}" if metadata.get("uploader") else None,
                            f"Duration: {metadata.get('duration')}s" if metadata.get("duration") else None,
                        ],
                    )
                )
            )

        st.subheader("📝 Transcript")
        st.text_area(
            "Transcript",
            value=st.session_state["yt_transcript_text"],
            height=400,
            disabled=True,
        )

        st.download_button(
            label="💾 Download as Text",
            data=st.session_state["yt_transcript_text"],
            file_name="transcript.txt",
            mime="text/plain",
        )

        st.download_button(
            label="💾 Download as SRT",
            data=st.session_state["yt_transcript_srt"],
            file_name="transcript.srt",
            mime="text/plain",
        )


if __name__ == "__main__":
    main()
