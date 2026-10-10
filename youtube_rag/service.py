"""Download once, save provenance, then reuse native media indexing and answers."""

from pathlib import Path
import tempfile

from media_rag.config import RagError
from media_rag.service import MediaLibrary

from .download import download_video
from media_rag.youtube_urls import parse_youtube_url


class YouTubeLibrary(MediaLibrary):
    def retrieve(self, question, asset_ids=None, modality=None, top_k=5, min_score=0.15):
        allowed = {str(asset["id"]) for asset in self.database.list_youtube_assets() if asset["status"] == "ready"}
        selected = list(allowed) if asset_ids is None else [str(value) for value in asset_ids]
        if any(value not in allowed for value in selected):
            raise RagError("Choose ready videos from this YouTube library.")
        return super().retrieve(question, selected, modality, top_k, min_score)

    def add_youtube(self, url, title="", on_progress=None):
        source = parse_youtube_url(url)
        title = title.strip()
        if len(title) > 200:
            raise RagError("Give the video a title of at most 200 characters.")
        existing = self.database.find_youtube(source.video_id)
        if existing:
            return existing, False
        with tempfile.TemporaryDirectory(prefix="youtube-rag-import-") as directory:
            if on_progress:
                on_progress("Downloading the YouTube video…")
            downloaded = download_video(source, Path(directory))
            if on_progress:
                on_progress("Saving the original video…")
            with downloaded.path.open("rb") as recording:
                asset, _ = self.add_upload(recording, title or downloaded.title)
            return self.database.link_youtube(source.video_id, source.url, asset["id"])
