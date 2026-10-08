# Native audio/video RAG

`media_rag_app.py` is a separate Streamlit application. It stores original media
and playable evidence clips in private MinIO objects, indexes native Marengo 3.5
audio/visual embeddings in PostgreSQL with pgvector, and sends retrieved raw
clips to Gemini for cited answers. No transcript is required.

## Start with Docker

From the repository root:

```bash
python3 scripts/init_media_rag.py
```

Edit the ignored `.env.media-rag` and set `TWELVELABS_API_KEY` and `GEMINI_API_KEY`.
The initializer generates random local infrastructure passwords and never
overwrites an existing file. Do not commit this file or publish the output of
`docker compose config --environment`. Compose reads its secrets from this file
and mounts only the secrets each service needs under `/run/secrets`; credential
values are not passed in container environment variables. The optional worker has
no Gemini secret. Restart/recreate the services after changing keys.
`MEDIA_RAG_GEMINI_MODEL`
defaults to `gemini-3.8-flash`; select a Gemini model supporting audio, video,
and structured JSON responses if your account uses a different model.

```bash
docker compose --env-file .env.media-rag -f compose.media-rag.yaml up -d --build
```

Open [the media library](http://127.0.0.1:8503).
[MinIO's console](http://127.0.0.1:9011) uses the access key and secret key from
your local configuration. PostgreSQL is on `127.0.0.1:5544` and the MinIO API is
on `127.0.0.1:9010`. These ports are bound to loopback. The Compose project has
its own network and named volumes, independent of the other applications.

Upload MP3, WAV, MP4, MOV, or WebM media (up to 200 MB), give it a title, and
click **Upload & index**. The app indexes the recording and displays progress;
keep the page open until it finishes. Existing queued recordings can be indexed
from **Library**. Search the ready recordings, inspect the retrieved moments, and
choose **Ask & cite** or **Answer this search with Gemini** for a grounded answer.
**Search only** uses Marengo without calling Gemini.

```bash
docker compose --env-file .env.media-rag -f compose.media-rag.yaml logs --tail 50 app
docker compose --env-file .env.media-rag -f compose.media-rag.yaml down
```

Stopping the stack preserves its volumes. Avoid `down -v` unless you intend to
delete the local media and database.

## Run the app on the host

Python 3.10+ and FFmpeg (including `ffprobe`) are required. Use a virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements-media-rag.txt
python3 scripts/init_media_rag.py
docker compose --env-file .env.media-rag -f compose.media-rag.yaml up -d postgres minio
.venv/bin/streamlit run media_rag_app.py --server.address 127.0.0.1 --server.port 8503 --server.maxUploadSize 200 --server.enableCORS true --server.enableXsrfProtection true --server.allowedHosts localhost --server.allowedHosts 127.0.0.1
```

The app reads `.env.media-rag`. Environment variables override this file.
The host app can alternatively read the same variable names from a `[media_rag]`
table in `.streamlit/secrets.toml`. Credentials are never entered into database rows
or shown in the app. Other applications' database settings are not reused.
Individual credentials can also use their corresponding `*_FILE` settings.
Unreadable infrastructure secret files stop startup. Missing optional provider
files disable that provider and do not inherit another key. An explicitly
configured Gemini file never falls back to a generic `GOOGLE_API_KEY`.

## Optional unattended worker

The app indexes recordings directly; a separate worker is optional. For
unattended processing of queued recordings, start the Compose worker explicitly:

```bash
docker compose --env-file .env.media-rag -f compose.media-rag.yaml --profile worker up -d worker
```

Alternatively, run `.venv/bin/python -m media_rag.worker` on the host. For Neon,
prefix that command with `MEDIA_RAG_ENV_FILE=.env.neon MEDIA_RAG_STORAGE_PROVIDER=neon`.
The worker needs environment settings or its dotenv file; it does not read
Streamlit secrets. Existing worker containers can be stopped with
`docker compose --env-file .env.media-rag -f compose.media-rag.yaml --profile worker stop worker`
when switching to indexing only in the app.

## Data flow and recovery

1. FFprobe checks the file contents and duration. The app streams it to a
   temporary file, computes its SHA-256, and saves the original in MinIO.
   Duplicate content reuses the existing library entry.
2. PostgreSQL stores the queued job. The app claims the selected recording with
   `FOR UPDATE SKIP LOCKED` and a five-minute renewable lease. Abandoned jobs can
   be reclaimed.
3. The app uploads the file directly to TwelveLabs; a localhost MinIO URL
   would not be reachable by the provider. It persists the remote asset and
   embedding task IDs before polling, and renews the lease at each poll.
4. Marengo 3.5 creates dynamic clip segments with separate `audio` and `visual`
   vectors for video, or `audio` vectors for audio. Silent video requests only
   visual embeddings. All queries and stored vectors use the same model and
   512 dimensions. Provider vectors, modalities, and original timestamps are
   validated before an atomic transaction marks the recording ready.
5. pgvector retrieves ready media within the selected recordings and modality.
   An HNSW cosine index uses iterative scanning for filtered searches. Overlapping
   hits are merged and expanded by eight seconds of context, up to 90 seconds
   per evidence clip. Longer provider segments use a centered 90-second window.
   The similarity threshold is a tunable retrieval cutoff,
   not a confidence probability.
6. FFmpeg creates H.264/AAC video clips or MP3 audio clips. Clips are cached in
   private MinIO objects. Gemini receives the actual clip files and absolute
   source offsets, then returns structured claims with source IDs. The server
   rejects unknown or missing citations. The app renders the cited clip players
   with their original-recording time ranges.

Failed ingestion stays out of search. **Retry indexing** resumes persisted IDs
when possible; a known failed/expired provider task is cleared so it can be
recreated. A timeout preserves the task ID. **Resume indexing** continues an
interrupted recording after its previous lease expires (up to five minutes).
App or optional worker restarts do not require re-uploading jobs with saved IDs.
A process crash or network timeout between a provider POST and saving its
returned ID can leave an orphan remote upload/task;
provider APIs do not provide an idempotency key for those calls.

## Scope and provider handling

This is a local library for one trusted user, without authentication or tenant
isolation. All server and console ports are local. Add authentication and enforced
per-user storage/retrieval boundaries before exposing it to other users.

The Docker app keeps CORS and XSRF protection enabled and restricts WebSocket
hosts to `localhost` and `127.0.0.1`. App and worker run as a non-root user with
all Linux capabilities dropped, privilege escalation disabled, and CPU, memory,
process, and temporary-storage limits. Environment-backed Compose secrets
require a writable container filesystem; the container does not mount the host
workspace or Docker socket. Local Docker administrators can still access secrets.

Media parsing forces the expected container format, allows only the `file`
protocol, disables MOV external references, and strips credentials and proxy
settings from FFmpeg/FFprobe environments. Renamed HLS/concat playlists are
rejected. Provider responses and titles are escaped before entering Markdown
widgets so they cannot insert images that contact external servers.

Indexing uploads the recording to TwelveLabs. Answering uploads only retrieved
clips to Google Gemini; uploaded Gemini files are deleted in a `finally` block
on a best-effort basis. TwelveLabs assets remain in your provider account.
MinIO retains originals and generated clips until its local volume is removed.
Model processing and retrieval can miss important evidence; a valid source ID
does not prove that every generated assertion is supported. Gemini is instructed
to return insufficient evidence when appropriate.

Native speech and sound search and native visual search are implemented. This
version does not add transcript/OCR keyword indexes or accept media-example
queries. Questions spanning an entire long recording may need broader coverage
than the retrieved clips. Evaluate retrieval time ranges and answer support on
your own recordings before tuning chunking or the similarity cutoff.

## Validation

Unit tests cover provider contracts, recovery, invalid vectors and
timestamps, evidence merging, citations, temporary Gemini-file cleanup, and
Streamlit behavior, secret isolation, hostile media, and Markdown rendering:

```bash
.venv/bin/python -m unittest discover -s tests -p 'test_media_rag*.py' -v
```

The integration tests use real MinIO, PostgreSQL/pgvector, and FFmpeg, with only
the paid providers replaced by deterministic test clients. Docker Compose must
be installed for the credential-scope check. Start the two local services, then run:

```bash
MEDIA_RAG_INTEGRATION=1 .venv/bin/python -m unittest discover -s tests -p 'test_media_rag*.py' -v
```

Each integration test creates and removes a uniquely named `media_rag_test_*`
database and `media-rag-test-*` bucket. It requires the loopback development
endpoints and never calls TwelveLabs or Gemini.

See [the security review](media-rag-security.md) for findings, remediation, and
validation limits.

Provider references:

- [Marengo 3.5](https://docs.twelvelabs.io/docs/concepts/models/marengo/marengo-3-5)
- [Direct uploads](https://docs.twelvelabs.io/api-reference/upload-files/direct-uploads/create)
- [Async Embed API v2](https://docs.twelvelabs.io/api-reference/create-embeddings-v2/create-async-embedding-task)
- [Query embeddings](https://docs.twelvelabs.io/docs/guides/create-embeddings/query)
- [Gemini audio](https://ai.google.dev/gemini-api/docs/audio)
- [Gemini video](https://ai.google.dev/gemini-api/docs/video-understanding)
- [pgvector](https://github.com/pgvector/pgvector)
