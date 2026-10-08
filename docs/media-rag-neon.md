# Native media RAG with Neon

The app and worker can use Neon's private S3-compatible object storage instead
of MinIO. Select `MEDIA_RAG_STORAGE_PROVIDER=neon`; the default remains the local
MinIO configuration. Existing configuration files and local media are preserved.
This starts a separate Neon library; it does not copy local recordings or rows.

## Project configuration

This directory is linked to Neon project `sparkling-tree-31338825`, branch
`production`. `neon.ts` declares the private `rag` bucket. Deploying that policy
does not create the application's PostgreSQL schema.

The Neon CLI exports credentials into the ignored `.env.neon` file. Keep it
private and never commit its contents. Refresh it with:

```bash
neon env pull --project-id sparkling-tree-31338825 --branch production \
  --file .env.neon --service postgres --service object-storage
chmod 600 .env.neon
```

Required keys are `DATABASE_URL`, `DATABASE_URL_UNPOOLED`, `AWS_ENDPOINT_URL_S3`,
`AWS_REGION`, `AWS_ACCESS_KEY_ID`, and `AWS_SECRET_ACCESS_KEY`. Both database
URLs and the storage endpoint must belong to the same Neon branch. The pooled
URL handles application queries; the direct URL handles schema initialization.
`MEDIA_RAG_DATABASE_URL` and `MEDIA_RAG_DATABASE_SCHEMA_URL` are explicit overrides.
`MEDIA_RAG_STORAGE_BUCKET` defaults to `rag`. Credential settings support `*_FILE`.

Add `TWELVELABS_API_KEY` and `GEMINI_API_KEY` to the new `.env.neon` file or inject
them through the environment. The worker needs only the TwelveLabs key.
`MEDIA_RAG_GEMINI_MODEL` can override the Gemini model as in the local setup.

## Schema and startup

Before the first production launch, test the repository's `media_rag/schema.sql`
on a Neon child branch. Review and approve the production schema change before
launching against production: app and worker initialization runs that SQL,
enabling pgvector and creating the `media_rag` tables and indexes. Use the
direct endpoint for schema work. No production schema changes were made during
the project setup.

Install the Python dependencies and FFmpeg/FFprobe as described in
[the local guide](media-rag.md). Start the worker:

```bash
MEDIA_RAG_ENV_FILE=.env.neon MEDIA_RAG_STORAGE_PROVIDER=neon \
  .venv/bin/python -m media_rag.worker
```

Start the UI in another terminal:

```bash
MEDIA_RAG_ENV_FILE=.env.neon MEDIA_RAG_STORAGE_PROVIDER=neon \
  .venv/bin/streamlit run media_rag_app.py --server.address 127.0.0.1 \
  --server.port 8503 --server.maxUploadSize 200 \
  --server.enableCORS true --server.enableXsrfProtection true \
  --server.allowedHosts localhost --server.allowedHosts 127.0.0.1
```

Neon storage uses HTTPS, path-style addressing, and SigV4. Initialization checks
the already-provisioned bucket instead of creating buckets or changing access
policies. Playback URLs expire after 15 minutes. Errors do not expose credentials
in the UI. Credentials are passed explicitly to the S3 client, so it cannot fall
back to an unrelated AWS profile or instance role.

For Streamlit Community Cloud, follow the
[deployment and schema guide](media-rag-community-cloud.md). The Cloud entrypoint
is `apps/media_rag/app.py`; its adjacent `requirements.txt` installs the RAG
dependencies, and the root `packages.txt` installs FFmpeg/FFprobe. Run the
indexing worker separately with the same branch settings. Keep access restricted
to the trusted user until application authentication and per-user retrieval
boundaries are added.

Project Codex skills are under `.agents/skills/`. The project OAuth MCP entry
is in `.codex/config.toml`; authenticate when Codex first connects. These files
contain no database or storage credentials.

Reference: [Neon S3 compatibility](https://neon.com/docs/storage/s3-compatibility),
[Neon storage quickstart](https://neon.com/docs/storage/get-started).
