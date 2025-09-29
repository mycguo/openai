import os
import tempfile
import time
from dataclasses import dataclass
from typing import Iterable, List, Optional

import requests
import streamlit as st
import yt_dlp
from openai import OpenAI


st.set_page_config(
    page_title="YouTube Transcript Extractor",
    page_icon="📺",
    layout="wide"
)


ASSEMBLYAI_API_KEY = st.secrets.get("ASSEMBLYAI_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY")

UPLOAD_ENDPOINT = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_ENDPOINT = "https://api.assemblyai.com/v2/transcript"
CHUNK_SIZE = 5_242_880  # 5 MB

TRANSCRIPTS_DIR = os.path.join("Data", "transcripts")


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
        "noplaylist": True,
        "geo_bypass": True,
        "nocheckcertificate": True,
        "http_headers": {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/125.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9",
            "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,image/avif,image/webp,*/*;q=0.8",
            "Connection": "keep-alive",
        },
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
        "speaker_labels": True,
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


def _format_speaker_label(raw: Optional[str]) -> str:
    if raw is None:
        return ""
    label = str(raw).strip()
    if not label:
        return ""
    if label.lower().startswith("speaker"):
        return label.title()
    if len(label) == 1 and label.isalpha():
        return f"Speaker {label.upper()}"
    if label.isdigit():
        return f"Speaker {label}"
    return label


def _segments_from_utterances(utterances: Optional[Iterable[dict]]) -> tuple[List[TranscriptSegment], str]:
    utterances = list(utterances or [])
    if not utterances:
        return [], ""

    segments: List[TranscriptSegment] = []
    lines: List[str] = []

    for utterance in utterances:
        text = (utterance.get("text") or "").strip()
        if not text:
            continue

        speaker = _format_speaker_label(
            utterance.get("speaker") or utterance.get("speaker_label") or utterance.get("id")
        )

        start = float(utterance.get("start", 0)) / 1000.0
        end = float(utterance.get("end", 0)) / 1000.0

        line = f"{speaker}: {text}" if speaker else text
        lines.append(line)

        segments.append(
            TranscriptSegment(
                index=len(segments) + 1,
                start=start,
                end=end,
                text=line,
            )
        )

    return segments, "\n".join(lines)


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
        "yt_transcript_video_id",
        "yt_transcript_summary",
    ]:
        st.session_state.pop(key, None)


def _ensure_transcript_dir() -> str:
    os.makedirs(TRANSCRIPTS_DIR, exist_ok=True)
    return TRANSCRIPTS_DIR


def _sanitize_filename(name: str) -> str:
    keep = (" ", "-", "_", ".")
    sanitized = "".join(ch if ch.isalnum() or ch in keep else "_" for ch in name)
    return sanitized.strip().strip("._") or "transcript"


def save_transcript_to_section(text: str, metadata: dict) -> str:
    directory = _ensure_transcript_dir()
    video_id = metadata.get("id") or metadata.get("display_id") or metadata.get("video_id")
    base_name = video_id or metadata.get("title") or "transcript"
    filename = _sanitize_filename(base_name) + ".txt"
    path = os.path.join(directory, filename)

    with open(path, "w", encoding="utf-8") as file:
        file.write(text)

    return path


def generate_summary(text: str) -> str:
    if not OPENAI_API_KEY:
        raise RuntimeError("Missing OPENAI_API_KEY in Streamlit secrets.")

    client = OpenAI()

    prompt = (
        "You are an expert analyst preparing an executive briefing based on a podcast transcript."
        "Please write a concise summary of this article in a few paragraphs. Focus on clearly explaining the main topic and key points in a flowing narrative format. Just write it as you would naturally explain the article to someone."
        "\nDon't use Spearker A or Speaker B if you don't know who they are, focusing on the content and the main points."
        "\n\nGuidelines:"
        "\n Try to write a comprehensive summary and make it long and detailed."
        "\n- Keep sections clearly labeled with markdown headings and bullet points as appropriate."
        "\n- If the transcript lacks information for a section, explicitly note the gap."
        "\n\nTranscript excerpt (truncated if long):\n"
        f"{text[:15000]}"
    )

    completion = client.chat.completions.create(
        model="gpt-4o",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
        max_tokens=5000,
    )

    return completion.choices[0].message.content.strip()


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

                    utterances = transcript_data.get("utterances")
                    segments: List[TranscriptSegment] = []
                    text = ""

                    if utterances:
                        segments, text_with_speakers = _segments_from_utterances(utterances)
                        if text_with_speakers:
                            text = text_with_speakers

                    if not text:
                        text = transcript_data.get("text", "").strip()

                    words = transcript_data.get("words", [])
                    if not segments:
                        segments = _segments_from_words(words)

                    if not segments and text:
                        segments = [
                            TranscriptSegment(
                                index=1,
                                start=0.0,
                                end=max(len(text.split()) / 2, 1.0),
                                text=text,
                            )
                        ]

                    if not text:
                        text = "\n".join(segment.text for segment in segments)

                    if not text.strip():
                        raise ValueError("Transcription completed but returned empty text.")

                    srt_text = _segments_to_srt(segments)

                    st.session_state.yt_transcript_text = text
                    st.session_state.yt_transcript_srt = srt_text
                    st.session_state.yt_transcript_metadata = {
                        "title": metadata.get("title"),
                        "uploader": metadata.get("uploader"),
                        "duration": metadata.get("duration"),
                        "webpage_url": metadata.get("webpage_url"),
                        "id": metadata.get("id"),
                    }
                    st.session_state.yt_transcript_video_id = metadata.get("id") or metadata.get("display_id")

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

        dl_col1, dl_col2, dl_col3 = st.columns(3)

        with dl_col1:
            st.download_button(
                label="💾 Download as Text",
                data=st.session_state["yt_transcript_text"],
                file_name="transcript.txt",
                mime="text/plain",
                use_container_width=True,
            )

        with dl_col2:
            st.download_button(
                label="💾 Download as SRT",
                data=st.session_state["yt_transcript_srt"],
                file_name="transcript.srt",
                mime="text/plain",
                use_container_width=True,
            )

        with dl_col3:
            if st.button("📤 Upload to Transcript Section", use_container_width=True):
                try:
                    saved_path = save_transcript_to_section(
                        st.session_state["yt_transcript_text"],
                        metadata,
                    )
                    st.success(f"Transcript saved to {saved_path}")
                except Exception as exc:
                    st.error(f"Failed to save transcript: {exc}")

        if st.button("🧠 Generate Summary"):
            try:
                summary = generate_summary(st.session_state["yt_transcript_text"])
                st.session_state.yt_transcript_summary = summary
                st.success("Summary generated!")
            except Exception as exc:
                st.error(f"Failed to generate summary: {exc}")

        if st.session_state.get("yt_transcript_summary"):
            st.subheader("🧠 Transcript Summary")
            st.markdown(st.session_state["yt_transcript_summary"])

            st.download_button(
                label="💾 Download Summary",
                data=st.session_state["yt_transcript_summary"],
                file_name="transcript_summary.txt",
                mime="text/plain",
            )


if __name__ == "__main__":
    main()
