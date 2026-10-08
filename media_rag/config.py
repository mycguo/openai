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
        values = {**dotenv_values(ROOT / ".env.media-rag"), **os.environ, **(overrides or {})}
        database_url = secret_value(values, "MEDIA_RAG_DATABASE_URL")
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
            minio_endpoint=values.get("MEDIA_RAG_MINIO_ENDPOINT") or "127.0.0.1:9010",
            minio_public_endpoint=values.get("MEDIA_RAG_MINIO_PUBLIC_ENDPOINT") or "127.0.0.1:9010",
            minio_access_key=secret_value(values, "MEDIA_RAG_MINIO_ACCESS_KEY"),
            minio_secret_key=secret_value(values, "MEDIA_RAG_MINIO_SECRET_KEY"),
            minio_secure=str(values.get("MEDIA_RAG_MINIO_SECURE", "false")).lower() == "true",
            bucket=values.get("MEDIA_RAG_MINIO_BUCKET") or "media-rag",
            twelvelabs_api_key=secret_value(values, "TWELVELABS_API_KEY", optional=True),
            gemini_api_key=gemini_key,
            gemini_model=values.get("MEDIA_RAG_GEMINI_MODEL") or "gemini-3.8-flash",
        )

    def missing_infrastructure(self):
        return [name for name, value in (
            ("MEDIA_RAG_DATABASE_URL", self.database_url),
            ("MEDIA_RAG_MINIO_ACCESS_KEY", self.minio_access_key),
            ("MEDIA_RAG_MINIO_SECRET_KEY", self.minio_secret_key),
        ) if not value]
