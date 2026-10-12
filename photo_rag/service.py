"""Persist photos and retrieve original visual evidence using the existing infrastructure."""

from dataclasses import dataclass
import hashlib
from pathlib import Path
import tempfile
import uuid

from media_rag.config import RagError
from media_rag.storage import create_storage
from .config import MAX_PHOTO_BYTES, validate_question
from .database import PhotoDatabase
from .gemini import PhotoGemini
from .images import prepare_image


@dataclass(frozen=True)
class PhotoEvidence:
    source_id: int
    asset_id: str
    title: str
    image_key: str
    thumbnail_key: str
    object_key: str


class PhotoLibrary:
    def __init__(self, settings, database=None, storage=None):
        self.settings = settings
        self.database = database or PhotoDatabase(settings.database_url, schema_url=settings.database_schema_url)
        self.storage = storage or create_storage(settings)

    def initialize(self):
        self.database.initialize()
        self.storage.initialize()

    def add_upload(self, uploaded, title=""):
        title = title.strip() or Path(getattr(uploaded, "name", "Photo")).stem
        if not title or len(title) > 200 or any(ord(char) < 32 for char in title):
            raise RagError("Give the photo a title of 1–200 characters without control characters.")
        uploaded.seek(0)
        with tempfile.TemporaryDirectory(prefix="ask-my-photo-upload-") as directory:
            directory = Path(directory)
            original = directory / "original"
            digest, size = hashlib.sha256(), 0
            with original.open("wb") as destination:
                while chunk := uploaded.read(1024 * 1024):
                    size += len(chunk)
                    if size > MAX_PHOTO_BYTES:
                        raise RagError("Photos are limited to 15 MB each. Resize this photo first.")
                    digest.update(chunk)
                    destination.write(chunk)
            if not size:
                raise RagError("The uploaded photo is empty.")
            existing = self.database.find_hash(digest.hexdigest())
            if existing:
                return existing, False
            prepared = prepare_image(original, directory)
            asset_id = str(uuid.uuid4())
            prefix = f"photos/{asset_id}/"
            keys = {"object_key": prefix + "original" + prepared["extension"],
                    "image_key": prefix + "image.jpg", "thumbnail_key": prefix + "thumbnail.jpg"}
            for key, path, mime in [(keys["object_key"], original, prepared["mime_type"]),
                                     (keys["image_key"], prepared["image_path"], "image/jpeg"),
                                     (keys["thumbnail_key"], prepared["thumbnail_path"], "image/jpeg")]:
                self.storage.upload(key, path, mime)
            # Retain objects on ambiguous failures; a DB write may already have committed.
            asset = self.database.create_photo({"id": asset_id, "sha256": digest.hexdigest(), "title": title,
                "size_bytes": size, "width": prepared["width"], "height": prepared["height"],
                "mime_type": prepared["mime_type"], **keys})
            if str(asset["id"]) != asset_id:
                for key in keys.values():
                    self.storage.remove(key)
                return asset, False
            return asset, True

    def index_photo(self, asset_id, retry=False):
        with PhotoGemini(self.settings) as provider:
            asset = self.database.claim(asset_id, retry=retry)
            if asset is None:
                raise RagError("This photo is ready or being indexed in another session. Refresh the library. "
                               "Interrupted indexing can be resumed after five minutes.")
            try:
                with tempfile.TemporaryDirectory(prefix="ask-my-photo-index-") as directory:
                    path = Path(directory) / "image.jpg"
                    self.storage.download(asset["image_key"], path)
                    self.database.complete(asset, provider.embed_photo(path))
            except Exception as exc:
                message = str(exc) if isinstance(exc, RagError) else "Photo indexing failed. Check storage and retry."
                self.database.fail(asset, message)
                raise RagError(message) from exc

    def retrieve(self, question, asset_ids=None, top_k=5, min_score=0.15):
        question = validate_question(question)
        if type(top_k) is not int or not 1 <= top_k <= 8 or not 0 <= min_score <= 1:
            raise RagError("Choose one to eight photos and a similarity threshold between zero and one.")
        with PhotoGemini(self.settings) as provider:
            vector = provider.embed_query(question)
        return self.database.search(vector, asset_ids=asset_ids, limit=top_k, min_score=min_score)

    def answer(self, question, hits):
        question = validate_question(question)
        if not hits:
            return {"status": "insufficient_evidence", "message": "No matching photos were found.", "claims": []}, []
        if len(hits) > 8:
            raise RagError("Choose at most eight evidence photos.")
        evidence = [PhotoEvidence(index, str(hit["id"]), hit["title"], hit["image_key"],
                                 hit["thumbnail_key"], hit["object_key"]) for index, hit in enumerate(hits, 1)]
        with PhotoGemini(self.settings) as provider, tempfile.TemporaryDirectory(prefix="ask-my-photo-answer-") as directory:
            paths = []
            for item in evidence:
                path = Path(directory) / f"photo-{item.source_id}.jpg"
                self.storage.download(item.image_key, path)
                paths.append(path)
            return provider.answer(question, evidence, paths), evidence
