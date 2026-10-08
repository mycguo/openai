"""Credential isolation and hostile-upload regressions for the native media app."""

from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import os
from pathlib import Path
import secrets
import shutil
import subprocess
import tempfile
import threading
import unittest
from unittest.mock import patch

from psycopg2.extensions import parse_dsn

from media_rag.config import RagError, Settings
from media_rag.media import extract_clip, probe, subprocess_environment


ROOT = Path(__file__).resolve().parents[1]


class SecretConfigurationTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="media-rag-security-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)
        self.enterContext(patch("media_rag.config.ROOT", self.root))
        self.enterContext(patch.dict(os.environ, {}, clear=True))

    def test_file_secrets_override_environment_without_mutating_it(self):
        secret = secrets.token_hex(16) + " ' + \\ "
        path = self.root / "database"
        path.write_text(secret + "\n")
        # Special characters must remain a password, never extra DSN parameters.
        expected = secret.strip()
        os.environ.update(MEDIA_RAG_DATABASE_PASSWORD_FILE=str(path),
                          MEDIA_RAG_DATABASE_PASSWORD="unused",
                          MEDIA_RAG_DATABASE_URL="postgresql://media_rag@127.0.0.1:5544/media_rag")
        settings = Settings.load()
        parsed = parse_dsn(settings.database_url)
        self.assertEqual(parsed["password"], expected)
        self.assertEqual(parsed["host"], "127.0.0.1")
        self.assertNotIn(expected, repr(settings))
        self.assertEqual(os.environ["MEDIA_RAG_DATABASE_PASSWORD"], "unused")

    def test_unreadable_secret_fails_closed_instead_of_using_fallback(self):
        path = self.root / "missing"
        with self.assertRaises(RagError) as raised:
            Settings.load({"MEDIA_RAG_MINIO_SECRET_KEY_FILE": str(path), "MEDIA_RAG_MINIO_SECRET_KEY": secrets.token_hex(16)})
        self.assertIn("MEDIA_RAG_MINIO_SECRET_KEY", str(raised.exception))
        self.assertNotIn(str(path), str(raised.exception))

    def test_missing_optional_provider_secret_disables_inference_without_fallback(self):
        settings = Settings.load({"TWELVELABS_API_KEY_FILE": str(self.root / "missing"),
                                  "TWELVELABS_API_KEY": secrets.token_hex(16)})
        self.assertEqual(settings.twelvelabs_api_key, "")

    def test_explicit_empty_gemini_file_does_not_inherit_another_apps_google_key(self):
        path = self.root / "empty"
        path.write_text("")
        settings = Settings.load({"GEMINI_API_KEY_FILE": str(path), "GOOGLE_API_KEY": secrets.token_hex(16)})
        self.assertEqual(settings.gemini_api_key, "")

    def test_invalid_secret_files_are_rejected_without_echoing_contents(self):
        path = self.root / "invalid"
        for content in [b"x" * 8193, b"\xff"]:
            path.write_bytes(content)
            with self.subTest(size=len(content)), self.assertRaises(RagError):
                Settings.load({"TWELVELABS_API_KEY_FILE": str(path)})

    def test_media_subprocesses_do_not_inherit_credentials_or_proxies(self):
        secret = secrets.token_hex(16)
        os.environ.update(PATH="/usr/bin", LANG="C", TWELVELABS_API_KEY=secret,
                          MEDIA_RAG_DATABASE_URL=secret, HTTPS_PROXY=secret, GOOGLE_API_KEY=secret,
                          LD_PRELOAD=secret)
        self.assertEqual(subprocess_environment(), {"PATH": "/usr/bin", "LANG": "C"})


@unittest.skipUnless(shutil.which("ffmpeg") and shutil.which("ffprobe"), "FFmpeg is required")
class NativeMediaSecurityTests(unittest.TestCase):
    def setUp(self):
        self.directory = tempfile.TemporaryDirectory(prefix="media-rag-hostile-media-")
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def test_playlist_disguised_as_supported_media_cannot_read_another_file(self):
        audio = self.root / "target.wav"
        subprocess.run(["ffmpeg", "-v", "error", "-f", "lavfi", "-i", "sine=duration=1", str(audio)],
                       check=True, timeout=30, capture_output=True)
        disguised = self.root / "playlist.wav"
        disguised.write_text("ffconcat version 1.0\nfile 'target.wav'\n")
        # Ordinary WAV remains valid, while the renamed concat script is rejected.
        self.assertEqual(probe(audio)["kind"], "audio")
        with self.assertRaises(RagError):
            probe(disguised)
        with self.assertRaises(RagError):
            extract_clip(disguised, self.root / "result.mp3", 0, 1, "audio")

    @unittest.skipUnless(os.environ.get("MEDIA_RAG_INTEGRATION") == "1", "loopback network test is opt-in")
    def test_renamed_hls_playlist_cannot_contact_a_loopback_server(self):
        requests = []

        class Handler(BaseHTTPRequestHandler):
            def do_GET(self):
                requests.append(self.path)
                self.send_response(404)
                self.end_headers()

            def log_message(self, *args):
                pass

        server = ThreadingHTTPServer(("127.0.0.1", 0), Handler)
        thread = threading.Thread(target=lambda: server.serve_forever(poll_interval=0.01), daemon=True)
        thread.start()
        try:
            disguised = self.root / "playlist.mp4"
            disguised.write_text("#EXTM3U\n#EXT-X-TARGETDURATION:1\n#EXTINF:1,\n"
                                 f"http://127.0.0.1:{server.server_port}/internal\n#EXT-X-ENDLIST\n")
            with self.assertRaises(RagError):
                probe(disguised)
            with self.assertRaises(RagError):
                extract_clip(disguised, self.root / "result.mp4", 0, 1, "video")
            self.assertEqual(requests, [])
        finally:
            server.shutdown()
            server.server_close()
            thread.join(timeout=5)


@unittest.skipUnless(os.environ.get("MEDIA_RAG_INTEGRATION") == "1", "local Docker configuration test is opt-in")
class ComposeSecurityTests(unittest.TestCase):
    def test_container_environment_has_no_secret_values_and_worker_has_no_gemini_key(self):
        result = subprocess.run(["docker", "compose", "--env-file", ".env.media-rag", "-f",
                                 "compose.media-rag.yaml", "--profile", "worker", "config", "--format", "json"],
                                cwd=ROOT, check=True, capture_output=True, text=True, timeout=30)
        config = json.loads(result.stdout)
        for name, service in config["services"].items():
            environment = service.get("environment", {})
            for key in environment:
                if "PASSWORD" in key or "KEY" in key:
                    self.assertTrue(key.endswith("_FILE"), f"{name} has a credential in its environment")
            if name in {"app", "worker"}:
                self.assertNotIn("password", parse_dsn(environment["MEDIA_RAG_DATABASE_URL"]))
                self.assertEqual(service["cap_drop"], ["ALL"])
                self.assertIn("no-new-privileges:true", service["security_opt"])
        worker_secrets = {item["source"] for item in config["services"]["worker"]["secrets"]}
        self.assertNotIn("gemini_api_key", worker_secrets)


if __name__ == "__main__":
    unittest.main()
