"""Configuration isolated from the other applications in this repository."""

from dataclasses import dataclass, field
import os
from pathlib import Path
from typing import Mapping

from dotenv import dotenv_values
from psycopg2.extensions import make_dsn


ROOT = Path(__file__).resolve().parents[1]
MODEL = "marengo3.5"
DIMENSIONS = 512
MAX_UPLOAD_BYTES = 200_000_000  # TwelveLabs direct-upload limit, in decimal bytes.


class RagError(RuntimeError):
    """A user-facing error whose message does not contain credentials."""


def secret_value(values, name, optional=False):
    """Explicit file references take precedence and fail closed if unreadable."""
    path = values.get(name + "_FILE")
    if not path:
        return values.get(name) or ""
    try:
        with Path(path).open("rb") as handle:
            content = handle.read(8193)
        if len(content) > 8192:
            raise ValueError("oversized secret")
        return content.decode("utf-8").strip()
    except FileNotFoundError as exc:
        # Compose does not materialize empty environment-backed optional secrets.
        # An absent provider key disables inference; never use a fallback value.
        if optional:
            return ""
        raise RagError(f"Could not read the configured {name} secret file.") from exc
    except (OSError, ValueError) as exc:
        raise RagError(f"Could not read the configured {name} secret file.") from exc


@dataclass(frozen=True)
class Settings:
    database_url: str = field(default="", repr=False)
    database_schema_url: str = field(default="", repr=False)
    storage_provider: str = "minio"
    s3_endpoint: str = ""
    s3_region: str = "us-east-2"
    s3_access_key: str = field(default="", repr=False)
    s3_secret_key: str = field(default="", repr=False)
    minio_endpoint: str = "127.0.0.1:9010"
    minio_public_endpoint: str = "127.0.0.1:9010"
    minio_access_key: str = field(default="", repr=False)
    minio_secret_key: str = field(default="", repr=False)
    minio_secure: bool = False
    bucket: str = "media-rag"
    twelvelabs_api_key: str = field(default="", repr=False)
    gemini_api_key: str = field(default="", repr=False)
    gemini_model: str = "gemini-3.8-flash"
    processing_timeout: int = 1800
    poll_interval: float = 5.0

    @classmethod
    def load(cls, overrides: Mapping | None = None):
        explicit = {**os.environ, **(overrides or {})}
        env_file = explicit.get("MEDIA_RAG_ENV_FILE") or ROOT / ".env.media-rag"
        values = {**dotenv_values(env_file), **explicit}
        provider = str(values.get("MEDIA_RAG_STORAGE_PROVIDER") or "minio").lower()
        if provider not in {"minio", "neon"}:
            raise RagError("MEDIA_RAG_STORAGE_PROVIDER must be minio or neon.")
        database_url = secret_value(values, "MEDIA_RAG_DATABASE_URL")
        if provider == "neon" and not database_url and not values.get("MEDIA_RAG_DATABASE_URL_FILE"):
            database_url = secret_value(values, "DATABASE_URL")
        schema_url = ""
        if provider == "neon":
            schema_url = secret_value(values, "MEDIA_RAG_DATABASE_SCHEMA_URL")
            if not schema_url and not values.get("MEDIA_RAG_DATABASE_SCHEMA_URL_FILE"):
                schema_url = secret_value(values, "DATABASE_URL_UNPOOLED")
        password = secret_value(values, "MEDIA_RAG_DATABASE_PASSWORD")
        if database_url and password:
            try:
                database_url = make_dsn(database_url, password=password)
            except Exception as exc:
                raise RagError("The media database connection settings are invalid.") from exc
        gemini_key = secret_value(values, "GEMINI_API_KEY", optional=True)
        if not values.get("GEMINI_API_KEY_FILE") and not gemini_key:
            gemini_key = secret_value(values, "GOOGLE_API_KEY")
        return cls(
            database_url=database_url,
            database_schema_url=schema_url,
            storage_provider=provider,
            s3_endpoint=values.get("AWS_ENDPOINT_URL_S3") or "",
            s3_region=values.get("AWS_REGION") or "us-east-2",
            s3_access_key=secret_value(values, "AWS_ACCESS_KEY_ID") if provider == "neon" else "",
            s3_secret_key=secret_value(values, "AWS_SECRET_ACCESS_KEY") if provider == "neon" else "",
            minio_endpoint=values.get("MEDIA_RAG_MINIO_ENDPOINT") or "127.0.0.1:9010",
            minio_public_endpoint=values.get("MEDIA_RAG_MINIO_PUBLIC_ENDPOINT") or "127.0.0.1:9010",
            minio_access_key=secret_value(values, "MEDIA_RAG_MINIO_ACCESS_KEY") if provider == "minio" else "",
            minio_secret_key=secret_value(values, "MEDIA_RAG_MINIO_SECRET_KEY") if provider == "minio" else "",
            minio_secure=str(values.get("MEDIA_RAG_MINIO_SECURE", "false")).lower() == "true",
            bucket=(values.get("MEDIA_RAG_STORAGE_BUCKET") or "rag") if provider == "neon" else
                   (values.get("MEDIA_RAG_MINIO_BUCKET") or "media-rag"),
            twelvelabs_api_key=secret_value(values, "TWELVELABS_API_KEY", optional=True),
            gemini_api_key=gemini_key,
            gemini_model=values.get("MEDIA_RAG_GEMINI_MODEL") or "gemini-3.8-flash",
        )

    def missing_infrastructure(self):
        required = [("MEDIA_RAG_DATABASE_URL", self.database_url)]
        if self.storage_provider == "neon":
            required += [("DATABASE_URL_UNPOOLED", self.database_schema_url),
                         ("AWS_ENDPOINT_URL_S3", self.s3_endpoint),
                         ("AWS_ACCESS_KEY_ID", self.s3_access_key),
                         ("AWS_SECRET_ACCESS_KEY", self.s3_secret_key)]
        else:
            required += [("MEDIA_RAG_MINIO_ACCESS_KEY", self.minio_access_key),
                         ("MEDIA_RAG_MINIO_SECRET_KEY", self.minio_secret_key)]
        return [name for name, value in required if not value]
