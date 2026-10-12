"""Real local Postgres/MinIO with synthetic photos and mocked Gemini inference."""

from dataclasses import replace
import io
import os
from pathlib import Path
import unittest
from unittest.mock import patch
import uuid

from PIL import Image
import psycopg2
from psycopg2 import sql
from psycopg2.extensions import make_dsn, parse_dsn
import requests

from media_rag.config import RagError, Settings
from media_rag.database import Database
from photo_rag.config import DIMENSIONS
from photo_rag.service import PhotoLibrary


def vector(axis=0):
    values = [0.0] * DIMENSIONS
    values[axis] = 1.0
    return values


class FakeGemini:
    def __init__(self, settings):
        pass

    def __enter__(self):
        return self

    def __exit__(self, *args):
        pass

    def embed_photo(self, path):
        with Image.open(path) as image:
            assert image.format == "JPEG" and not image.getexif()
            return vector(0 if image.getpixel((0, 0))[0] > 128 else 1)

    def embed_query(self, question):
        return vector(0 if "red" in question else 1)

    def answer(self, question, evidence, paths):
        for path in paths:
            with Image.open(path) as image:
                assert image.format == "JPEG" and not image.getexif()
        return {"status": "answered", "message": "", "claims": [
            {"text": "Synthetic red photo", "source_ids": [evidence[0].source_id]}]}


@unittest.skipUnless(os.environ.get("MEDIA_RAG_INTEGRATION") == "1", "local photo integration tests are opt-in")
class PhotoIntegrationTests(unittest.TestCase):
    def setUp(self):
        settings = Settings.load()
        self.assertEqual(settings.storage_provider, "minio", "Local test stack only")
        params = parse_dsn(settings.database_url)
        self.assertIn(params.get("host"), {"localhost", "127.0.0.1"}, "Local Postgres only")
        self.assertIn(settings.minio_endpoint, {"localhost:9010", "127.0.0.1:9010"}, "Local MinIO only")
        suffix = uuid.uuid4().hex[:12]
        self.admin_dsn = settings.database_url
        self.database_name = "photo_rag_test_" + suffix
        self.bucket_name = "photo-rag-test-" + suffix
        self.addCleanup(self.cleanup_services)
        connection = psycopg2.connect(self.admin_dsn, connect_timeout=5)
        try:
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.database_name)))
        finally:
            connection.close()
        params["dbname"] = self.database_name
        self.settings = replace(settings, database_url=make_dsn(**params), database_schema_url="",
                                bucket=self.bucket_name, gemini_api_key="synthetic-test-key")
        self.library = PhotoLibrary(self.settings)
        self.library.initialize()
        self.enterContext(patch("photo_rag.service.PhotoGemini", FakeGemini))

    def cleanup_services(self):
        if hasattr(self, "library"):
            client = self.library.storage.client
            if client.bucket_exists(self.bucket_name):
                for item in client.list_objects(self.bucket_name, recursive=True):
                    client.remove_object(self.bucket_name, item.object_name)
                client.remove_bucket(self.bucket_name)
        connection = psycopg2.connect(self.admin_dsn, connect_timeout=5)
        try:
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute(sql.SQL("DROP DATABASE IF EXISTS {}").format(sql.Identifier(self.database_name)))
        finally:
            connection.close()

    def upload(self, color="red"):
        content = io.BytesIO()
        Image.new("RGB", (120, 80), color).save(content, format="PNG")
        content.seek(0)
        content.name = color + ".png"
        return self.library.add_upload(content)

    def test_upload_deduplicate_index_retrieve_answer_and_private_originals(self):
        red, created = self.upload()
        self.assertTrue(created)
        duplicate, created = self.upload()
        self.assertFalse(created)
        self.assertEqual(red["id"], duplicate["id"])
        blue, _ = self.upload("blue")
        self.assertEqual(self.library.retrieve("Find red"), [])
        self.library.index_photo(red["id"])
        self.library.index_photo(blue["id"])
        self.assertEqual(self.library.database.counts(), {"ready": 2})
        hits = self.library.retrieve("Find red")
        self.assertEqual(len(hits), 1)
        self.assertEqual(hits[0]["id"], red["id"])
        self.assertEqual(self.library.retrieve("Find red", [str(blue["id"])]), [])
        answer, evidence = self.library.answer("Find red", hits)
        self.assertEqual(answer["claims"][0]["source_ids"], [1])
        self.assertEqual(evidence[0].asset_id, str(red["id"]))
        url = self.library.storage.playback_url(red["object_key"])
        self.assertEqual(requests.get(url, timeout=10).status_code, 200)
        self.assertEqual(requests.get(url.split("?", 1)[0], timeout=10).status_code, 403)
        self.library.initialize()  # Additive schema initialization is idempotent.

    def test_failed_and_interrupted_indexing_cannot_be_overwritten_by_stale_lease(self):
        photo, _ = self.upload()
        database = self.library.database
        first = database.claim(photo["id"])
        self.assertIsNone(database.claim(photo["id"]))
        with self.assertRaises(RagError):
            database.complete(first, vector()[:-1])
        database.fail(first, "Synthetic timeout")
        self.assertIsNone(database.claim(photo["id"]))
        retried = database.claim(photo["id"], retry=True)
        with database.connect() as cursor:
            cursor.execute("UPDATE photo_rag.assets SET lease_until=now()-interval '1 minute' WHERE id=%s", (str(photo["id"]),))
        resumed = database.claim(photo["id"])
        with self.assertRaises(RagError):
            database.complete(retried, vector())
        database.fail(retried, "Stale failure")
        database.complete(resumed, vector())
        self.assertEqual(database.counts(), {"ready": 1})

    def test_photo_schema_coexists_with_media_schema_without_mixing_assets_or_vectors(self):
        media = Database(self.settings.database_url)
        media.initialize()
        media.create_asset({"id": str(uuid.uuid4()), "sha256": "synthetic-media", "title": "Media sentinel",
            "object_key": "originals/sentinel.mp4", "mime_type": "video/mp4", "kind": "video",
            "has_audio": False, "duration": 30, "size_bytes": 100})
        photo, _ = self.upload()
        self.library.index_photo(photo["id"])
        self.assertEqual(len(media.list_assets()), 1)
        self.assertEqual(len(self.library.database.list_photos()), 1)
        self.assertEqual(self.library.retrieve("Find red")[0]["title"], "red")


if __name__ == "__main__":
    unittest.main()
