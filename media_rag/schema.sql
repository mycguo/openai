CREATE EXTENSION IF NOT EXISTS vector;
CREATE SCHEMA IF NOT EXISTS media_rag;

CREATE TABLE IF NOT EXISTS media_rag.assets (
    id uuid PRIMARY KEY,
    sha256 text NOT NULL UNIQUE,
    title text NOT NULL,
    object_key text NOT NULL UNIQUE,
    mime_type text NOT NULL,
    kind text NOT NULL CHECK (kind IN ('audio', 'video')),
    has_audio boolean NOT NULL,
    duration double precision NOT NULL CHECK (duration > 0),
    size_bytes bigint NOT NULL CHECK (size_bytes > 0),
    status text NOT NULL DEFAULT 'queued' CHECK (status IN ('queued', 'indexing', 'ready', 'failed')),
    stage text NOT NULL DEFAULT 'Waiting to be indexed',
    error text,
    remote_asset_id text,
    task_id text,
    lease_token uuid,
    lease_until timestamptz,
    created_at timestamptz NOT NULL DEFAULT now(),
    updated_at timestamptz NOT NULL DEFAULT now()
);

CREATE TABLE IF NOT EXISTS media_rag.embeddings (
    asset_id uuid NOT NULL REFERENCES media_rag.assets(id) ON DELETE CASCADE,
    model text NOT NULL CHECK (model = 'marengo3.5'),
    modality text NOT NULL CHECK (modality IN ('audio', 'visual')),
    start_sec double precision NOT NULL CHECK (start_sec >= 0),
    end_sec double precision NOT NULL CHECK (end_sec > start_sec),
    embedding vector(512) NOT NULL,
    PRIMARY KEY (asset_id, modality, start_sec, end_sec)
);

CREATE INDEX IF NOT EXISTS embeddings_cosine_idx
    ON media_rag.embeddings USING hnsw (embedding vector_cosine_ops);
CREATE INDEX IF NOT EXISTS assets_queue_idx ON media_rag.assets (status, lease_until, created_at);

-- Preserve YouTube identity independently of content-hash deduplication.
CREATE TABLE IF NOT EXISTS media_rag.youtube_sources (
    video_id text PRIMARY KEY CHECK (video_id ~ '^[A-Za-z0-9_-]{11}$'),
    source_url text NOT NULL CHECK (source_url = 'https://www.youtube.com/watch?v=' || video_id),
    asset_id uuid NOT NULL REFERENCES media_rag.assets(id) ON DELETE CASCADE,
    created_at timestamptz NOT NULL DEFAULT now()
);
CREATE INDEX IF NOT EXISTS youtube_sources_asset_idx ON media_rag.youtube_sources (asset_id);
