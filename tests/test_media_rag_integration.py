"""Opt-in tests with real PostgreSQL, MinIO, and FFmpeg; no paid API calls."""

from dataclasses import replace
import io
import os
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest.mock import patch
import uuid

import psycopg2
from psycopg2 import sql
from psycopg2.extensions import make_dsn, parse_dsn
import requests

from media_rag.config import RagError, Settings
from media_rag.gemini import validate_answer
from media_rag.media import probe
from media_rag.models import Segment, merge_hits
from media_rag.service import MediaLibrary, index_asset


def vector(axis=0):
    result = [0.0] * 512
    result[axis] = 1.0
    return result


class FakeMarengo:
    """Only cloud inference is substituted; storage and job handling stay real."""

    def __init__(self, kind):
        self.kind = kind

    def upload(self, path, mime_type):
        assert probe(path)["kind"] == self.kind
        return "synthetic-asset"

    def wait_asset(self, asset_id, heartbeat):
        heartbeat()

    def create_task(self, asset_id, kind, has_audio):
        assert kind == self.kind
        assert has_audio == (kind == "audio")
        return "synthetic-task"

    def wait_task(self, task_id, heartbeat):
        heartbeat()
        return {"status": "ready", "metadata": {"embedding_dimension": 512}, "data": [
            {"embedding_scope": "clip", "embedding_option": "audio" if self.kind == "audio" else "visual",
             "start_sec": start, "end_sec": end, "embedding": vector(0 if self.kind == "audio" else 1)}
            for start, end in [(0, 2), (2, 4)]
        ]}


@unittest.skipUnless(os.environ.get("MEDIA_RAG_INTEGRATION") == "1", "local integration tests are opt-in")
class LocalIntegrationTests(unittest.TestCase):
    def setUp(self):
        settings = Settings.load()
        self.admin_dsn = settings.database_url
        parameters = parse_dsn(self.admin_dsn)
        self.assertIn(parameters.get("host"), {"127.0.0.1", "localhost"}, "Use the local test stack only")
        self.assertIn(settings.minio_endpoint, {"127.0.0.1:9010", "localhost:9010"})
        suffix = uuid.uuid4().hex[:12]
        self.database_name = "media_rag_test_" + suffix
        self.bucket_name = "media-rag-test-" + suffix
        self.addCleanup(self.cleanup_services)
        connection = psycopg2.connect(self.admin_dsn, connect_timeout=5)
        try:
            connection.autocommit = True
            with connection.cursor() as cursor:
                cursor.execute(sql.SQL("CREATE DATABASE {}").format(sql.Identifier(self.database_name)))
        finally:
            connection.close()
        parameters["dbname"] = self.database_name
        self.settings = replace(settings, database_url=make_dsn(**parameters), bucket=self.bucket_name,
                                twelvelabs_api_key="synthetic-test-key", gemini_api_key="synthetic-test-key")
        self.library = MediaLibrary(self.settings)
        self.library.initialize()
        self.directory = tempfile.TemporaryDirectory(prefix="media-rag-test-")
        self.addCleanup(self.directory.cleanup)
        self.audio = Path(self.directory.name) / "tone.wav"
        self.video = Path(self.directory.name) / "silent.mp4"
        self.generate("sine=frequency=440:duration=4", self.audio)
        self.generate("color=c=blue:s=320x240:d=4", self.video, "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p")

    def generate(self, source, output, *extra):
        subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", source, *extra, str(output)],
                       check=True, timeout=30, capture_output=True)

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

    def upload(self, path, title):
        uploaded = io.BytesIO(path.read_bytes())
        uploaded.name = path.name
        return self.library.add_upload(uploaded, title)

    def test_native_upload_index_search_clip_and_private_playback(self):
        audio, created = self.upload(self.audio, "Synthetic tone")
        self.assertTrue(created)
        duplicate, created = self.upload(self.audio, "Duplicate")
        self.assertFalse(created)
        self.assertEqual(duplicate["id"], audio["id"])
        video, created = self.upload(self.video, "Synthetic silent video")
        self.assertTrue(created)
        self.assertFalse(video["has_audio"])
        self.assertEqual(self.library.database.search(vector()), [])  # Queued media is excluded.
        for kind in ["audio", "video"]:
            asset = self.library.database.claim()
            self.assertEqual(asset["kind"], kind)
            index_asset(self.library, asset, FakeMarengo(kind))
        self.assertIsNone(self.library.database.claim())
        self.assertTrue(all(item["status"] == "ready" and item["embedding_count"] == 2
                            for item in self.library.database.list_assets()))
        hits = self.library.database.search(vector(), asset_ids=[str(audio["id"])], modality="audio")
        self.assertEqual(len(hits), 2)
        self.assertTrue(all(hit.score > 0.99 for hit in hits))
        self.assertEqual(self.library.database.search(vector(), asset_ids=[str(video["id"])], modality="audio"), [])
        with patch("media_rag.service.Marengo") as provider:
            provider.return_value.embed_query.return_value = vector()
            retrieved = self.library.retrieve("What sound?", [str(audio["id"])], "audio")
        self.assertEqual(len(retrieved), 1)
        video_hits = merge_hits(self.library.database.search(vector(1), modality="visual"))
        self.assertEqual(len(video_hits), 1)
        test = self

        class FakeGemini:
            def __init__(self, settings):
                pass

            def answer(self, question, evidence, paths):
                for item, path in zip(evidence, paths, strict=True):
                    info = probe(path)
                    test.assertEqual(info["kind"], item.kind)
                    test.assertAlmostEqual(info["duration"], item.end - item.start, delta=0.3)
                return validate_answer({"status": "answered", "claims": [
                    {"text": "Synthetic evidence", "source_ids": [item.source_id for item in evidence]}
                ]}, evidence)

        with patch("media_rag.service.Gemini", FakeGemini):
            answer, evidence = self.library.answer("Inspect both clips", retrieved + video_hits)
            self.assertEqual(answer["claims"][0]["source_ids"], [1, 2])
            # Exercise cached clips too, including silent video playback.
            self.library.answer("Inspect both clips again", retrieved + video_hits)
        for item in evidence:
            self.assertEqual((item.start, item.end), (0, 4))
            url = self.library.storage.playback_url(item.object_key)
            response = requests.get(url, timeout=10)
            self.assertEqual(response.status_code, 200)
            self.assertGreater(len(response.content), 100)
            anonymous = requests.get(url.split("?", 1)[0], timeout=10)
            self.assertEqual(anonymous.status_code, 403)

    def test_failed_transaction_and_retry_preserve_saved_job_ids(self):
        self.upload(self.audio, "Retry fixture")
        asset = self.library.database.claim()
        self.assertIsNone(self.library.database.claim())
        self.library.database.progress(asset, "Saved task", remote_asset_id="remote", task_id="task")
        with self.assertRaises(RagError):
            self.library.database.complete(asset, [Segment(0, 4, "audio", vector()[:-1])])
        row = self.library.database.list_assets()[0]
        self.assertEqual((row["status"], row["embedding_count"]), ("indexing", 0))
        self.library.database.fail(asset, "Synthetic timeout")
        self.assertTrue(self.library.database.retry(asset["id"]))
        retried = self.library.database.claim()
        self.assertEqual((retried["remote_asset_id"], retried["task_id"]), ("remote", "task"))
        self.library.database.complete(retried, [Segment(0, 4, "audio", vector())])
        with self.assertRaises(RagError):
            self.library.database.progress(asset, "Stale worker")
        self.assertEqual(self.library.database.list_assets()[0]["status"], "ready")

    def test_streamlit_indexes_selected_recording_without_claiming_an_older_job(self):
        audio, _ = self.upload(self.audio, "Older queued audio")
        video, _ = self.upload(self.video, "Selected silent video")
        provider = FakeMarengo("video")
        with patch("media_rag.service.Marengo", return_value=provider):
            self.library.index_recording(video["id"])
        rows = {str(row["id"]): row for row in self.library.database.list_assets()}
        self.assertEqual(rows[str(audio["id"])]["status"], "queued")
        self.assertEqual(rows[str(video["id"])]["status"], "ready")
        self.assertEqual(rows[str(video["id"])]["embedding_count"], 2)
        self.assertIsNone(self.library.database.claim(asset_id=str(video["id"])))
        self.assertEqual(str(self.library.database.claim()["id"]), str(audio["id"]))

    def test_expired_lease_is_reclaimed_and_old_worker_cannot_publish(self):
        self.upload(self.audio, "Lease fixture")
        old = self.library.database.claim()
        with self.library.database.connect() as cursor:
            cursor.execute("UPDATE media_rag.assets SET lease_until=now()-interval '1 second' WHERE id=%s",
                           (str(old["id"]),))
        current = self.library.database.claim()
        self.assertNotEqual(old["lease_token"], current["lease_token"])
        with self.assertRaises(RagError):
            self.library.database.complete(old, [Segment(0, 4, "audio", vector())])
        self.library.database.fail(old, "Stale failure")
        self.assertEqual(self.library.database.list_assets()[0]["status"], "indexing")
        index_asset(self.library, current, FakeMarengo("audio"))
        self.assertEqual(self.library.database.list_assets()[0]["status"], "ready")


if __name__ == "__main__":
    unittest.main()
