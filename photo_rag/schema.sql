CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS photo_rag;

CREATE TABLE IF NOT EXISTS photo_rag.assets (
    id uuid PRIMARY KEY,
    sha256 text NOT NULL UNIQUE,
    title text NOT NULL,
    object_key text NOT NULL UNIQUE,
    image_key text NOT NULL UNIQUE,
    thumbnail_key text NOT NULL UNIQUE,
    mime_type text NOT NULL,
    width integer NOT NULL CHECK (width > 0),
    height integer NOT NULL CHECK (height > 0),
    size_bytes bigint NOT NULL CHECK (size_bytes > 0),
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'indexing', 'ready', 'failed')),
    error text,
    lease_token uuid,
    lease_until timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS photo_rag.embeddings (
    asset_id uuid PRIMARY KEY REFERENCES photo_rag.assets(id) ON DELETE CASCADE,
    model text NOT NULL CHECK (model = 'gemini-embedding-2'),
    embedding vector(768) NOT NULL
);
CREATE INDEX IF NOT EXISTS photo_embeddings_cosine_idx
    ON photo_rag.embeddings USING hnsw (embedding vector_cosine_ops);
