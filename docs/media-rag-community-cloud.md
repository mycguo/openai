# Native media RAG on Streamlit Community Cloud

Community Cloud hosts the Streamlit UI. Neon hosts PostgreSQL and the private
`rag` bucket. Recordings are indexed directly in the Streamlit app; no separate
worker service is required.

## Deployment files

Use `apps/media_rag/app.py` as the Cloud entrypoint. Its adjacent
`requirements.txt` includes `../../requirements-media-rag.txt`, so Cloud installs
the RAG dependencies instead of the root requirements for the other apps.
The root `packages.txt` includes `ffmpeg`, which provides both FFmpeg and
FFprobe. Community Cloud runs Streamlit from the repository root; the entrypoint
adds that root to the Python import path before calling the shared app.

To install and launch the same entrypoint locally, run from the repository root:

```bash
python -m pip install -r apps/media_rag/requirements.txt
streamlit run apps/media_rag/app.py
```

Install FFmpeg separately for local development. See [the local guide](media-rag.md).

## Initialize the database before deploying

App and worker startup execute `media_rag/schema.sql`. The SQL enables pgvector
and creates the `media_rag` schema, two tables, and their indexes. Test it on an
isolated child branch before connecting the app to production. Neon project
setup in PR #9 did not initialize this application schema.

1. Open the [Neon Console](https://console.neon.tech/) and select the existing
   project `sparkling-tree-31338825`.
2. Create a branch named `media-rag-schema-test` with `production` as its parent.
   A child branch's writes do not change its parent. See
   [Neon's branch instructions](https://neon.com/docs/manage/branches#create-a-branch).
3. Open **Postgres database → SQL Editor**, select `media-rag-schema-test`, and
   select the database that the app will use. Choose a role permitted to install
   the `vector` extension and create schemas, tables, and indexes.
4. Paste the complete contents of [schema.sql](../media_rag/schema.sql) into a
   new query and click **Run**. Do not run only the `CREATE EXTENSION` statement;
   the app also needs both tables and the indexes.
5. Run these checks on the same branch and database:

   ```sql
   SELECT extversion FROM pg_extension WHERE extname = 'vector';

   SELECT to_regclass('media_rag.assets') AS assets,
          to_regclass('media_rag.embeddings') AS embeddings;

   SELECT tablename, indexname
   FROM pg_indexes
   WHERE schemaname = 'media_rag'
   ORDER BY tablename, indexname;
   ```

   Expect a pgvector version, two non-null table names, and indexes including
   `embeddings_cosine_idx` and `assets_queue_idx`.
6. Once the child-branch setup succeeds and you are ready for the production
   schema change, switch the SQL Editor to **production** and the intended
   database. Run the same complete `schema.sql`, then repeat the checks.
   A successful child-branch test does not apply the SQL to production.

The statements use `IF NOT EXISTS` and can be repeated to initialize an empty
library. They do not migrate incompatible existing tables. If the child branch
reports an error or already has an incompatible schema, resolve that before
applying the SQL to production.

If using `psql` or a migration tool instead of the SQL Editor, use the **direct**
Neon connection string for schema work. Use the **pooled** connection for normal
app queries. See [Neon pooling](https://neon.com/docs/connect/connection-pooling)
and [SQL Editor instructions](https://neon.com/docs/get-started/query-with-neon-sql-editor).

## Configure and deploy the UI

In Community Cloud, create an app with:

- Repository: `mycguo/openai`
- Branch: `main` after the deployment PR is merged
- Main file path: `apps/media_rag/app.py`
- Python version: `3.12`, under **Advanced settings**

Paste the following TOML into **Advanced settings → Secrets**, replacing each
placeholder with the values for the same Neon branch and database. Do not commit
the credentials. Local `.env.neon` files are ignored and are not deployed.

```toml
[media_rag]
MEDIA_RAG_STORAGE_PROVIDER = "neon"
MEDIA_RAG_STORAGE_BUCKET = "rag"
DATABASE_URL = "<pooled Neon connection string with sslmode=require>"
DATABASE_URL_UNPOOLED = "<direct Neon connection string with sslmode=require>"
AWS_ENDPOINT_URL_S3 = "<Neon storage endpoint>"
AWS_REGION = "<your Neon project region>"
AWS_ACCESS_KEY_ID = "<Neon storage access key>"
AWS_SECRET_ACCESS_KEY = "<Neon storage secret key>"
TWELVELABS_API_KEY = "<TwelveLabs API key>"
GEMINI_API_KEY = "<Gemini API key>"
```

Use **Sharing → Only specific people can view this app** and restrict access to
the trusted user. Apps deployed from a public repository are public by default.
This application currently shares one library and has no per-user authorization.
See [Community Cloud sharing](https://docs.streamlit.io/deploy/streamlit-community-cloud/share-your-app).

## Index recordings in the app

1. Open **Add media**, choose a recording, and click **Upload & index**.
2. Keep the page open while the app uploads to TwelveLabs and creates native
   embeddings. The status panel shows the current step. Once finished, the
   library refreshes and the recording is ready to search.
3. For previously queued recordings, click **Index recording** in **Library**.
   For a failed recording, **Retry indexing** runs the retry immediately in the
   app, reusing saved provider IDs when possible.
4. If the app restarts or indexing is interrupted, refresh **Library** and use
   **Resume indexing**. You may need to wait up to five minutes for the previous
   processing lease to expire. A timeout preserves the task ID; an explicitly
   failed or expired provider task is recreated on retry.

Indexing runs in the Streamlit session and can take several minutes. It is not
an always-on background service: app restarts or session reruns can interrupt
it. PostgreSQL retains progress and prevents another session from claiming the
same recording while the lease is active. Regular page refreshes do not
automatically start paid indexing calls.

After indexing a small recording, test search, a cited Gemini answer, and clip
playback. Existing local MinIO recordings and database rows are not copied to
Neon automatically. The optional command-line worker remains available for
unattended indexing; see [the local guide](media-rag.md).

## References

- [Community Cloud file organization](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/file-organization)
- [Community Cloud dependencies](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/app-dependencies)
- [Community Cloud deployment](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/deploy)
- [Community Cloud secrets](https://docs.streamlit.io/deploy/streamlit-community-cloud/deploy-your-app/secrets-management)
