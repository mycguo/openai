"""Upload, native retrieval, and evidence assembly without Streamlit dependencies."""

import hashlib
from pathlib import Path
import tempfile
import uuid

from .config import MAX_UPLOAD_BYTES, RagError
from .database import Database
from .gemini import Gemini
from .marengo import Marengo, parse_segments
from .media import MIME_TYPES, SUPPORTED_EXTENSIONS, extract_clip, probe
from .models import Evidence, merge_hits
from .storage import Storage


class MediaLibrary:
    def __init__(self, settings, database=None, storage=None):
        self.settings = settings
        self.database = database or Database(settings.database_url)
        self.storage = storage or Storage(settings)

    def initialize(self):
        self.database.initialize()
        self.storage.initialize()

    def add_upload(self, uploaded, title):
        title = title.strip()
        if not title or len(title) > 200:
            raise RagError("Give the media a title of 1–200 characters.")
        extension = Path(uploaded.name).suffix.lower()
        if extension not in SUPPORTED_EXTENSIONS:
            raise RagError("Supported formats: MP3, WAV, MP4, MOV, and WebM.")
        uploaded.seek(0)
        asset_id = str(uuid.uuid4())
        with tempfile.TemporaryDirectory(prefix="media-rag-upload-") as directory:
            path = Path(directory) / ("source" + extension)
            digest, size = hashlib.sha256(), 0
            with path.open("wb") as destination:
                while chunk := uploaded.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_UPLOAD_BYTES:
                        raise RagError("Direct uploads are limited to 200 MB. Split or compress this file first.")
                    digest.update(chunk)
                    destination.write(chunk)
            if not size:
                raise RagError("The uploaded file is empty.")
            existing = self.database.find_hash(digest.hexdigest())
            if existing:
                return existing, False
            info = probe(path)
            key = f"originals/{asset_id}/source{extension}"
            self.storage.upload(key, path, info["mime_type"])
            try:
                asset = self.database.create_asset({
                    "id": asset_id, "sha256": digest.hexdigest(), "title": title,
                    "object_key": key, "size_bytes": size, **info,
                })
            except Exception:
                # The database write may have committed before a network failure. Retain
                # the original rather than risk deleting media referenced by a committed row.
                raise
            if str(asset["id"]) != asset_id:
                self.storage.remove(key)  # Another uploader won the digest race.
                return asset, False
            return asset, True

    def retrieve(self, question, asset_ids=None, modality=None, top_k=5, min_score=0.15):
        question = question.strip()
        if not question or len(question) > 2000:
            raise RagError("Ask a question of 1–2,000 characters.")
        vector = Marengo(self.settings).embed_query(question)
        hits = self.database.search(vector, asset_ids=asset_ids, modality=modality,
                                    limit=max(40, top_k * 8), min_score=min_score)
        return merge_hits(hits, limit=top_k)

    def answer(self, question, hits):
        if not hits:
            return {"status": "insufficient_evidence", "message": "No relevant media moments were found.", "claims": []}, []
        # Validate the key before extracting clips or uploading evidence.
        gemini = Gemini(self.settings)
        evidence, paths, originals = [], [], {}
        with tempfile.TemporaryDirectory(prefix="media-rag-evidence-") as directory:
            directory = Path(directory)
            for source_id, hit in enumerate(hits, 1):
                extension = ".mp4" if hit.kind == "video" else ".mp3"
                key = f"clips/{hit.asset_id}/{round(hit.start * 1000)}-{round(hit.end * 1000)}{extension}"
                clip_path = directory / f"evidence-{source_id}{extension}"
                if self.storage.exists(key):
                    self.storage.download(key, clip_path)
                else:
                    if hit.asset_id not in originals:
                        original = directory / (hit.asset_id + Path(hit.object_key).suffix)
                        self.storage.download(hit.object_key, original)
                        originals[hit.asset_id] = original
                    extract_clip(originals[hit.asset_id], clip_path, hit.start, hit.end, hit.kind, hit.has_audio)
                    self.storage.upload(key, clip_path, MIME_TYPES[extension])
                evidence.append(Evidence(source_id, hit.title, hit.kind, hit.start, hit.end, key))
                paths.append(clip_path)
            return gemini.answer(question, evidence, paths), evidence


def index_asset(library, asset, marengo):
    """Resume saved provider IDs; publish all embeddings in one transaction."""
    database = library.database
    remote_id, task_id = asset.get("remote_asset_id"), asset.get("task_id")
    if not task_id:
        if not remote_id:
            database.progress(asset, "Uploading to TwelveLabs")
            with tempfile.TemporaryDirectory(prefix="media-rag-index-") as directory:
                path = Path(directory) / ("source" + Path(asset["object_key"]).suffix)
                library.storage.download(asset["object_key"], path)
                database.progress(asset, "Uploading to TwelveLabs")
                remote_id = marengo.upload(path, asset["mime_type"])
            database.progress(asset, "Preparing media", remote_asset_id=remote_id)
        marengo.wait_asset(remote_id, lambda: database.progress(asset, "Preparing media"))
        task_id = marengo.create_task(remote_id, asset["kind"], asset["has_audio"])
        database.progress(asset, "Creating native embeddings", task_id=task_id)
    payload = marengo.wait_task(task_id, lambda: database.progress(asset, "Creating native embeddings"))
    segments = parse_segments(payload, asset["duration"], asset["kind"], asset["has_audio"])
    database.complete(asset, segments)
