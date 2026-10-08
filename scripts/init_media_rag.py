"""Create a private local config with random infrastructure passwords; never overwrite it."""

import os
from pathlib import Path
import secrets


def main():
    path = Path(__file__).resolve().parents[1] / ".env.media-rag"
    password = secrets.token_hex(24)
    content = f"""# Local development only. This file is ignored by Git.
MEDIA_RAG_POSTGRES_PASSWORD={password}
MEDIA_RAG_DATABASE_URL=postgresql://media_rag:{password}@127.0.0.1:5544/media_rag
MEDIA_RAG_MINIO_ENDPOINT=127.0.0.1:9010
MEDIA_RAG_MINIO_PUBLIC_ENDPOINT=127.0.0.1:9010
MEDIA_RAG_MINIO_ACCESS_KEY=media-rag-local
MEDIA_RAG_MINIO_SECRET_KEY={secrets.token_hex(24)}
MEDIA_RAG_MINIO_SECURE=false
MEDIA_RAG_MINIO_BUCKET=media-rag
TWELVELABS_API_KEY=
GEMINI_API_KEY=
MEDIA_RAG_GEMINI_MODEL=gemini-3.8-flash
"""
    try:
        descriptor = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
    except FileExistsError:
        print(".env.media-rag already exists; its values were preserved.")
        return
    with os.fdopen(descriptor, "w") as handle:
        handle.write(content)
    print("Created private .env.media-rag with random local passwords. Add your TwelveLabs and Gemini API keys.")


if __name__ == "__main__":
    main()
