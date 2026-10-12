# Ask My Photo

A separate Streamlit frontend for uploading photos, browsing a persistent library,
and asking questions grounded in matching images. It reuses the media app's
PostgreSQL connection, private MinIO or Neon bucket, storage adapters, Gemini API
key, and answer model. No new infrastructure account or indexing worker is needed.

Photos use `gemini-embedding-2` with 768 dimensions; answers use your configured
`MEDIA_RAG_GEMINI_MODEL` (default `gemini-3.8-flash`). Text questions and actual
photo pixels are embedded in the same space. Retrieved photos are sent as images
to Gemini, with server-validated references such as `[1]` beside the matching
thumbnails. This app does not use TwelveLabs, FFmpeg, or YouTube downloading.

## Run locally

Use Python 3.12. Reuse `.env.media-rag` and the existing local PostgreSQL/MinIO services:

```sh
uv venv .venv-ask-my-photo --python 3.12
uv pip install --python .venv-ask-my-photo/bin/python -r requirements-ask-my-photo.txt
docker compose --env-file .env.media-rag -f compose.media-rag.yaml up -d postgres minio
.venv-ask-my-photo/bin/streamlit run ask_my_photo_app.py --server.address 127.0.0.1 --server.port 8505
```

Open http://127.0.0.1:8505. The transcript, media, and YouTube apps keep their
existing ports. `GEMINI_API_KEY` (or the existing `GOOGLE_API_KEY` fallback) enables
indexing and answers. Without it, uploads are saved and can be indexed later from
**Photo library**. The embedding model must be available to that Gemini project.

The optional Docker service reuses the same infrastructure and secret files:

```sh
docker compose --env-file .env.media-rag -f compose.media-rag.yaml \
  --profile ask-my-photo up -d --build ask-my-photo
```

For an existing Neon configuration, start with
`MEDIA_RAG_ENV_FILE=.env.neon MEDIA_RAG_STORAGE_PROVIDER=neon` as in
[the Neon media guide](media-rag-neon.md). This does not provision another project
or bucket. Test the photo schema on a Neon child branch before using production.

## Community Cloud

After merging, create a separate app with:

- Repository: `mycguo/openai`
- Branch: `main`
- Main file: `apps/ask_my_photo/app.py`
- Python: `3.12`
- Suggested URL: `ask-my-photo` (if available)

Copy the existing media/YouTube app's secrets into this new app's Secrets settings.
Community Cloud does not share secrets automatically between apps. The photo app
merges top-level settings, `[media_rag]`, `[youtube_rag]`, and optional
`[ask_my_photo]` overrides in that order. Existing database, bucket, storage keys,
Gemini key, and model settings are enough; no new API key is required. Do not add
another copy of a TOML table that already exists. Restrict access to trusted users,
as described in [the Community Cloud guide](media-rag-community-cloud.md).

## Use

1. Open **Add photos** and upload JPEG, PNG, WebP, HEIC, or HEIF images.
2. Keep the page open while uploads are saved and indexed. Each batch supports
   up to ten photos, 15 MB and 50 megapixels per photo. Animated/multi-frame files
   are rejected. Identical uploads reuse their original photo rather than index again.
3. Open **Ask your photos**, ask a question, and optionally select specific photos.
   **Find photos** retrieves images; **Ask & cite** also generates an answer.
4. Check the matching thumbnails and open their original photos to verify claims.
5. Use **Photo library** to browse or retry failed indexing. Interrupted indexing
   can be resumed after its five-minute lease expires. There is no background worker.

The gallery shows twelve photos per page, browsing the latest 500 entries. An
unscoped question searches all ready photos in the database, including older ones.
Search results are not exhaustive; the app cannot guarantee exact library-wide
counts or dates/locations that are not visibly supported. EXIF/GPS is not indexed.

## Data and compatibility

The additive, idempotent [schema](../photo_rag/schema.sql) creates only
`photo_rag.assets`, `photo_rag.embeddings`, their constraints/index, and the vector
extension if absent. Schema initialization uses your existing direct/unpooled
endpoint; normal queries use the pooled/application endpoint. The database role
needs the same schema-creation permissions as the media app. Audio/video tables
and 512-dimensional Marengo embeddings are unchanged. Photo objects live under
`photos/<uuid>/` in the existing bucket.

Original photos remain private and retain their metadata. Orientation-corrected
JPEGs (up to 2048 pixels, 1.5 MB) and thumbnails omit source EXIF/GPS. Those JPEGs
go to Google for embedding and answering; small text may require viewing the
original or uploading a cropped image. Playback uses existing fifteen-minute
signed URLs. Everyone permitted to access the app shares one library, matching
the existing media setup; there is no added per-user login or tenant isolation.

Indexing failures keep originals for retry. An uncertain storage/database write
can leave unlinked objects; the app retains them instead of deleting possibly
committed data. A successful duplicate race removes only that upload's unused
objects. This version deliberately has no delete action.

Inference has a three-minute HTTP timeout and no automatic retries. Failed answers
can be retried without repeating retrieval. Ordinary reruns do not repeat uploads,
indexing, or model requests. Model text is escaped; provider bodies, photo contents,
and credentials are excluded from error messages. Only retrieved source IDs can
be accepted as answer citations.

## Validation

```sh
.venv-ask-my-photo/bin/python -m unittest discover -s tests -p 'test_photo_rag*.py'
MEDIA_RAG_STORAGE_PROVIDER=minio MEDIA_RAG_ENV_FILE=.env.media-rag MEDIA_RAG_INTEGRATION=1 \
  .venv-ask-my-photo/bin/python -m unittest discover -s tests -p 'test_photo_rag*.py'
```

Opt-in integration tests require the existing loopback PostgreSQL/MinIO stack.
They create and remove uniquely named test databases/buckets, use synthetic
photos and mocked Gemini calls, and exercise real storage, vector retrieval,
deduplication, leases, citations, and coexistence with the media schema. They
refuse Neon or non-loopback connections.

References: [Gemini image inputs](https://ai.google.dev/gemini-api/docs/generate-content/image-understanding),
[native multimodal embeddings](https://ai.google.dev/gemini-api/docs/embeddings).
