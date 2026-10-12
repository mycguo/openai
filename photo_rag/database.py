"""Photo-only schema with shared PostgreSQL connections and resumable inline indexing."""

from pathlib import Path
import uuid

import numpy as np

from media_rag.config import RagError
from media_rag.database import Database
from .config import MODEL, validate_vector


class PhotoDatabase(Database):
    def initialize(self):
        with Database(self.schema_url).connect() as cursor:
            cursor.execute("SELECT pg_advisory_xact_lock(88445103)")
            cursor.execute(Path(__file__).with_name("schema.sql").read_text())

    def list_photos(self):
        with self.connect() as cursor:
            cursor.execute("SELECT * FROM photo_rag.assets ORDER BY created_at DESC LIMIT 500")
            return [dict(row) for row in cursor.fetchall()]

    def counts(self):
        with self.connect() as cursor:
            cursor.execute("SELECT status, count(*) AS count FROM photo_rag.assets GROUP BY status")
            return {row["status"]: row["count"] for row in cursor.fetchall()}

    def find_hash(self, digest):
        with self.connect() as cursor:
            cursor.execute("SELECT * FROM photo_rag.assets WHERE sha256=%s", (digest,))
            row = cursor.fetchone()
            return dict(row) if row else None

    def create_photo(self, asset):
        names = ("id", "sha256", "title", "object_key", "image_key", "thumbnail_key", "mime_type",
                 "width", "height", "size_bytes")
        with self.connect() as cursor:
            cursor.execute("""INSERT INTO photo_rag.assets
                (id,sha256,title,object_key,image_key,thumbnail_key,mime_type,width,height,size_bytes)
                VALUES (%s,%s,%s,%s,%s,%s,%s,%s,%s,%s) ON CONFLICT (sha256) DO NOTHING RETURNING *""",
                tuple(asset[name] for name in names))
            row = cursor.fetchone()
            if row:
                return dict(row)
            cursor.execute("SELECT * FROM photo_rag.assets WHERE sha256=%s", (asset["sha256"],))
            return dict(cursor.fetchone())

    def claim(self, asset_id, retry=False):
        with self.connect() as cursor:
            cursor.execute("""UPDATE photo_rag.assets SET status='indexing', error=NULL,
                lease_token=%s, lease_until=now()+interval '5 minutes', updated_at=now()
                WHERE id=%s AND (status='queued' OR (status='failed' AND %s)
                    OR (status='indexing' AND lease_until < now())) RETURNING *""",
                (str(uuid.uuid4()), str(asset_id), retry))
            row = cursor.fetchone()
            return dict(row) if row else None

    def complete(self, asset, values):
        vector = np.array(validate_vector(values))
        with self.connect(vectors=True) as cursor:
            cursor.execute("""SELECT id FROM photo_rag.assets WHERE id=%s AND lease_token=%s
                AND status='indexing' FOR UPDATE""", (str(asset["id"]), str(asset["lease_token"])))
            if cursor.fetchone() is None:
                raise RagError("Another session is indexing this photo. Refresh the library.")
            cursor.execute("""INSERT INTO photo_rag.embeddings (asset_id,model,embedding) VALUES (%s,%s,%s)
                ON CONFLICT (asset_id) DO UPDATE SET model=EXCLUDED.model,embedding=EXCLUDED.embedding""",
                (str(asset["id"]), MODEL, vector))
            cursor.execute("""UPDATE photo_rag.assets SET status='ready', error=NULL,
                lease_token=NULL, lease_until=NULL, updated_at=now() WHERE id=%s""", (str(asset["id"]),))

    def fail(self, asset, message):
        with self.connect() as cursor:
            cursor.execute("""UPDATE photo_rag.assets SET status='failed', error=%s,
                lease_token=NULL, lease_until=NULL, updated_at=now()
                WHERE id=%s AND lease_token=%s AND status='indexing'""",
                (message, str(asset["id"]), str(asset["lease_token"])))

    def search(self, values, asset_ids=None, limit=5, min_score=0.15):
        vector = np.array(validate_vector(values))
        with self.connect(vectors=True) as cursor:
            cursor.execute("SET LOCAL hnsw.iterative_scan = 'strict_order'")
            cursor.execute("SET LOCAL hnsw.ef_search = 100")
            cursor.execute("""SELECT a.*, 1-(e.embedding <=> %s) AS score
                FROM photo_rag.embeddings e JOIN photo_rag.assets a ON a.id=e.asset_id
                WHERE a.status='ready' AND e.model=%s
                AND (%s::uuid[] IS NULL OR a.id=ANY(%s::uuid[]))
                ORDER BY e.embedding <=> %s LIMIT %s""", (vector, MODEL, asset_ids, asset_ids, vector, limit))
            return [dict(row) for row in cursor.fetchall() if row["score"] >= min_score]
