"""Short PostgreSQL transactions and durable, leased indexing jobs."""

from contextlib import contextmanager
from pathlib import Path
import uuid

import numpy as np
from pgvector.psycopg2 import register_vector
import psycopg2
from psycopg2.extras import RealDictCursor, execute_values

from .config import MODEL, RagError
from .models import Hit, validate_vector


class Database:
    def __init__(self, url, schema_url=None):
        self.url = url
        self.schema_url = schema_url or url

    @contextmanager
    def connect(self, vectors=False):
        connection = psycopg2.connect(self.url, connect_timeout=5)
        try:
            with connection:
                with connection.cursor(cursor_factory=RealDictCursor) as cursor:
                    # Poolers reject statement_timeout as a startup parameter.
                    cursor.execute("SET LOCAL statement_timeout = 30000")
                    if vectors:
                        register_vector(connection)
                    yield cursor
        finally:
            connection.close()

    def initialize(self):
        # Schema setup uses the direct endpoint; application queries can use a pooler.
        with Database(self.schema_url).connect() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(88445101)")
            cursor.execute(Path(__file__).with_name("schema.sql").read_text())

    def list_assets(self):
        with self.connect() as cursor:
            cursor.execute("""SELECT a.*, (SELECT count(*) FROM media_rag.embeddings e
                           WHERE e.asset_id = a.id) AS embedding_count
                           FROM media_rag.assets a ORDER BY created_at DESC""")
            return [dict(row) for row in cursor.fetchall()]

    def find_hash(self, digest):
        with self.connect() as cursor:
            cursor.execute("SELECT * FROM media_rag.assets WHERE sha256 = %s", (digest,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def create_asset(self, asset):
        with self.connect() as cursor:
            cursor.execute("""INSERT INTO media_rag.assets
                (id, sha256, title, object_key, mime_type, kind, has_audio, duration, size_bytes)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (sha256) DO NOTHING RETURNING *""",
                tuple(asset[name] for name in ("id", "sha256", "title", "object_key", "mime_type",
                                              "kind", "has_audio", "duration", "size_bytes")))
            row = cursor.fetchone()
            if row:
                return dict(row)
            cursor.execute("SELECT * FROM media_rag.assets WHERE sha256 = %s", (asset["sha256"],))
            return dict(cursor.fetchone())

    def claim(self, asset_id=None):
        token = str(uuid.uuid4())
        with self.connect() as cursor:
            cursor.execute("""WITH candidate AS (
                SELECT id FROM media_rag.assets WHERE status IN ('queued', 'indexing')
                AND (lease_until IS NULL OR lease_until < now())
                AND (%s::uuid IS NULL OR id=%s::uuid)
                ORDER BY created_at FOR UPDATE SKIP LOCKED LIMIT 1
            ) UPDATE media_rag.assets a SET status='indexing', lease_token=%s,
                lease_until=now()+interval '5 minutes', updated_at=now()
              FROM candidate c WHERE a.id=c.id RETURNING a.*""", (asset_id, asset_id, token))
            row = cursor.fetchone()
            return dict(row) if row else None

    def progress(self, asset, stage, remote_asset_id=None, task_id=None):
        with self.connect() as cursor:
            cursor.execute("""UPDATE media_rag.assets SET stage=%s,
                remote_asset_id=COALESCE(%s,remote_asset_id), task_id=COALESCE(%s,task_id),
                lease_until=now()+interval '5 minutes', updated_at=now()
                WHERE id=%s AND lease_token=%s AND status='indexing' RETURNING id""",
                (stage, remote_asset_id, task_id, str(asset["id"]), str(asset["lease_token"])))
            if cursor.fetchone() is None:
                raise RagError("Another session is indexing this recording. Refresh the library.")

    def complete(self, asset, segments):
        if not segments:
            raise RagError("The provider returned no searchable media segments.")
        with self.connect(vectors=True) as cursor:
            cursor.execute("""SELECT id FROM media_rag.assets
                WHERE id=%s AND lease_token=%s AND status='indexing' FOR UPDATE""",
                (str(asset["id"]), str(asset["lease_token"])))
            if cursor.fetchone() is None:
                raise RagError("Another session is indexing this recording. Refresh the library.")
            cursor.execute("DELETE FROM media_rag.embeddings WHERE asset_id=%s", (str(asset["id"]),))
            execute_values(cursor, """INSERT INTO media_rag.embeddings
                (asset_id,model,modality,start_sec,end_sec,embedding) VALUES %s""",
                [(str(asset["id"]), MODEL, item.modality, item.start, item.end,
                  np.array(validate_vector(item.vector))) for item in segments])
            cursor.execute("""UPDATE media_rag.assets SET status='ready', stage='Ready to search',
                error=NULL, lease_token=NULL, lease_until=NULL, updated_at=now() WHERE id=%s""",
                (str(asset["id"]),))

    def fail(self, asset, message, reset_asset=False, reset_task=False):
        with self.connect() as cursor:
            cursor.execute("""UPDATE media_rag.assets SET status='failed', stage='Indexing failed',
                error=%s, lease_token=NULL, lease_until=NULL, updated_at=now(),
                remote_asset_id=CASE WHEN %s THEN NULL ELSE remote_asset_id END,
                task_id=CASE WHEN %s THEN NULL ELSE task_id END
                WHERE id=%s AND lease_token=%s AND status='indexing'""",
                (message, reset_asset, reset_asset or reset_task,
                 str(asset["id"]), str(asset["lease_token"])))

    def retry(self, asset_id):
        with self.connect() as cursor:
            cursor.execute("""UPDATE media_rag.assets SET status='queued',
                stage='Waiting to be indexed', error=NULL, updated_at=now()
                WHERE id=%s AND status='failed' RETURNING id""", (str(asset_id),))
            return cursor.fetchone() is not None

    def search(self, query_vector, asset_ids=None, modality=None, limit=40, min_score=0.15):
        vector = np.array(validate_vector(query_vector))
        with self.connect(vectors=True) as cursor:
            # Iterative HNSW scanning prevents selected-asset/status filters starving candidates.
            cursor.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
            cursor.execute("SET LOCAL hnsw.ef_search = 100")
            cursor.execute("""SELECT a.id, a.title, a.object_key, a.kind, a.has_audio, a.duration,
                e.start_sec, e.end_sec, e.modality, 1-(e.embedding <=> %s) AS score
                FROM media_rag.embeddings e JOIN media_rag.assets a ON a.id=e.asset_id
                WHERE a.status='ready' AND e.model=%s
                AND (%s::uuid[] IS NULL OR a.id=ANY(%s::uuid[]))
                AND (%s::text IS NULL OR e.modality=%s)
                ORDER BY e.embedding <=> %s LIMIT %s""",
                (vector, MODEL, asset_ids, asset_ids, modality, modality, vector, limit))
            return [Hit(str(row["id"]), row["title"], row["object_key"], row["kind"],
                        row["duration"], row["start_sec"], row["end_sec"], row["score"],
                        (row["modality"],), row["has_audio"])
                    for row in cursor.fetchall() if row["score"] >= min_score]
