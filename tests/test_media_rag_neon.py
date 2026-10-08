"""Neon configuration isolation, S3 access, and pooled transaction regressions."""

import os
from pathlib import Path
import tempfile
import unittest
from unittest.mock import MagicMock, Mock, patch
from urllib.parse import parse_qs, urlsplit

from botocore.exceptions import ClientError
from botocore.stub import Stubber

from media_rag.config import RagError, Settings
from media_rag.database import Database
from media_rag.neon_storage import NeonStorage
from media_rag.storage import create_storage


NEON = Settings(storage_provider="neon", bucket="rag", database_url="postgresql://test",
                s3_endpoint="https://example.storage.neon.tech", s3_region="us-east-2",
                s3_access_key="test-access", s3_secret_key="test-secret")


class NeonConfigurationTests(unittest.TestCase):
    def setUp(self):
        directory = self.enterContext(tempfile.TemporaryDirectory())
        self.root = Path(directory)
        self.enterContext(patch("media_rag.config.ROOT", self.root))
        self.enterContext(patch.dict(os.environ, {}, clear=True))

    def test_neon_credentials_are_loaded_only_from_the_selected_env_file(self):
        (self.root / ".env.media-rag").write_text("MEDIA_RAG_DATABASE_URL=postgresql://local\n")
        neon_file = self.root / ".env.neon"
        neon_file.write_text("DATABASE_URL=postgresql://neon\nDATABASE_URL_UNPOOLED=postgresql://direct\n"
                             "AWS_ACCESS_KEY_ID=neon-access\n"
                             "AWS_SECRET_ACCESS_KEY=neon-secret\n"
                             "AWS_ENDPOINT_URL_S3=https://example.storage.neon.tech\n")
        local = Settings.load()
        self.assertEqual(local.storage_provider, "minio")
        self.assertEqual(local.database_url, "postgresql://local")
        settings = Settings.load({"MEDIA_RAG_ENV_FILE": str(neon_file), "MEDIA_RAG_STORAGE_PROVIDER": "neon"})
        self.assertEqual(settings.database_url, "postgresql://neon")
        self.assertEqual(settings.database_schema_url, "postgresql://direct")
        self.assertEqual(settings.bucket, "rag")
        self.assertEqual(settings.missing_infrastructure(), [])
        self.assertNotIn("neon-secret", repr(settings))
        self.assertNotIn("neon-access", repr(settings))
        self.assertNotIn("DATABASE_URL", os.environ)

    def test_minio_does_not_use_another_apps_generic_database_url(self):
        settings = Settings.load({"DATABASE_URL": "postgresql://unrelated"})
        self.assertEqual(settings.database_url, "")

    def test_explicit_database_and_storage_secrets_override_neon_defaults(self):
        secret_file = self.root / "aws-secret"
        secret_file.write_text("secret-from-file")
        settings = Settings.load({"MEDIA_RAG_STORAGE_PROVIDER": "neon",
                                  "DATABASE_URL": "postgresql://fallback",
                                  "MEDIA_RAG_DATABASE_URL": "postgresql://explicit",
                                  "AWS_SECRET_ACCESS_KEY_FILE": str(secret_file),
                                  "AWS_SECRET_ACCESS_KEY": "unused",
                                  "MEDIA_RAG_STORAGE_BUCKET": "other-private-bucket"})
        self.assertEqual(settings.database_url, "postgresql://explicit")
        self.assertEqual(settings.s3_secret_key, "secret-from-file")
        self.assertEqual(settings.bucket, "other-private-bucket")

    def test_neon_secret_file_fails_closed(self):
        with self.assertRaises(RagError):
            Settings.load({"MEDIA_RAG_STORAGE_PROVIDER": "neon",
                           "AWS_SECRET_ACCESS_KEY_FILE": str(self.root / "missing"),
                           "AWS_SECRET_ACCESS_KEY": "unused"})

    def test_explicit_empty_schema_secret_does_not_fall_back_to_another_url(self):
        empty = self.root / "empty"
        empty.write_text("")
        settings = Settings.load({"MEDIA_RAG_STORAGE_PROVIDER": "neon",
                                  "MEDIA_RAG_DATABASE_SCHEMA_URL_FILE": str(empty),
                                  "DATABASE_URL_UNPOOLED": "postgresql://unrelated",
                                  "MEDIA_RAG_MINIO_SECRET_KEY_FILE": str(self.root / "unused-missing")})
        self.assertEqual(settings.database_schema_url, "")
        self.assertIn("DATABASE_URL_UNPOOLED", settings.missing_infrastructure())

    def test_invalid_backend_is_rejected(self):
        with self.assertRaises(RagError):
            Settings.load({"MEDIA_RAG_STORAGE_PROVIDER": "invalid"})


class NeonStorageTests(unittest.TestCase):
    def test_factory_uses_neon_with_path_style_sigv4(self):
        storage = create_storage(NEON)
        self.addCleanup(storage.client.close)
        self.assertIsInstance(storage, NeonStorage)
        self.assertEqual(storage.client.meta.config.signature_version, "s3v4")
        self.assertEqual(storage.client.meta.config.s3["addressing_style"], "path")
        self.assertEqual(storage.client.meta.region_name, "us-east-2")
        parsed = urlsplit(storage.playback_url("clips/example one.mp3"))
        query = parse_qs(parsed.query)
        self.assertEqual(parsed.path, "/rag/clips/example%20one.mp3")
        self.assertEqual(query["X-Amz-Expires"], ["900"])
        self.assertEqual(query["X-Amz-Algorithm"], ["AWS4-HMAC-SHA256"])
        self.assertNotIn(NEON.s3_secret_key, parsed.query)

    def test_initialization_checks_existing_bucket_without_creating_one(self):
        storage = NeonStorage(NEON)
        self.addCleanup(storage.client.close)
        with Stubber(storage.client) as stub:
            stub.add_response("head_bucket", {}, {"Bucket": "rag"})
            storage.initialize()
            stub.assert_no_pending_responses()

    def test_missing_object_is_distinct_from_permission_failure(self):
        storage = NeonStorage(NEON)
        self.addCleanup(storage.client.close)
        with Stubber(storage.client) as stub:
            stub.add_client_error("head_object", service_error_code="404", http_status_code=404,
                                  expected_params={"Bucket": "rag", "Key": "missing"})
            stub.add_client_error("head_object", service_error_code="AccessDenied", http_status_code=403,
                                  expected_params={"Bucket": "rag", "Key": "denied"})
            self.assertFalse(storage.exists("missing"))
            with self.assertRaises(ClientError):
                storage.exists("denied")
            stub.assert_no_pending_responses()

    def test_bucket_errors_have_sanitized_user_messages(self):
        storage = NeonStorage(NEON)
        self.addCleanup(storage.client.close)
        with Stubber(storage.client) as stub:
            stub.add_client_error("head_bucket", service_error_code="AccessDenied",
                                  service_message="private-provider-body", http_status_code=403,
                                  expected_params={"Bucket": "rag"})
            with self.assertRaises(RagError) as raised:
                storage.initialize()
            self.assertNotIn("private-provider-body", str(raised.exception))

    def test_insecure_or_credential_bearing_endpoint_is_rejected_before_sdk_use(self):
        for endpoint in ["http://example.test", "https://user:secret@example.test",
                         "https://example.test?token=secret", "https://example.test#secret"]:
            with self.subTest(endpoint=endpoint), patch("media_rag.neon_storage.boto3.client") as client:
                with self.assertRaises(RagError):
                    NeonStorage(Settings(storage_provider="neon", s3_endpoint=endpoint))
                client.assert_not_called()


class PooledDatabaseTests(unittest.TestCase):
    def test_schema_initialization_uses_the_direct_endpoint(self):
        with patch("media_rag.database.Database.connect", autospec=True) as connect:
            Database("postgresql://pooled", schema_url="postgresql://direct").initialize()
        self.assertEqual(connect.call_args.args[0].url, "postgresql://direct")
        cursor = connect.return_value.__enter__.return_value
        self.assertIn("CREATE EXTENSION IF NOT EXISTS vector", cursor.execute.call_args.args[0])

    def test_timeout_is_transaction_local_instead_of_a_pooler_startup_parameter(self):
        connection, cursor = MagicMock(), MagicMock()
        connection.cursor.return_value.__enter__.return_value = cursor
        ordering = Mock()
        ordering.attach_mock(cursor.execute, "execute")
        with patch("media_rag.database.psycopg2.connect", return_value=connection) as connect:
            with patch("media_rag.database.register_vector") as register:
                ordering.attach_mock(register, "register")
                with Database("postgresql://test").connect(vectors=True) as active:
                    active.execute("SELECT 1")
        connect.assert_called_once_with("postgresql://test", connect_timeout=5)
        self.assertEqual(ordering.mock_calls[0].args, ("SET LOCAL statement_timeout = 30000",))
        self.assertEqual(ordering.mock_calls[1].args, (connection,))
        self.assertEqual(ordering.mock_calls[2].args, ("SELECT 1",))
        connection.close.assert_called_once()


if __name__ == "__main__":
    unittest.main()
