# youtube-rag

A separate Streamlit application for the native media RAG pipeline. Paste a
YouTube video URL instead of uploading a file. The app downloads the audio/video,
saves it privately, indexes timestamped Marengo 3.5 embeddings in PostgreSQL /
pgvector, and sends retrieved raw clips to Gemini for cited answers.

## Run locally

Use Python 3.12 and install FFmpeg/FFprobe. From the repository root:

```bash
uv venv .venv-youtube-rag --python 3.12
uv pip install --python .venv-youtube-rag/bin/python -r requirements-youtube-rag.txt
```

Reuse the existing `.env.media-rag` settings and local MinIO/PostgreSQL services.
If they are not configured yet, follow [the local media guide](media-rag.md).
Start only the infrastructure if necessary:

```bash
docker compose --env-file .env.media-rag -f compose.media-rag.yaml up -d postgres minio
.venv-youtube-rag/bin/streamlit run youtube_rag_app.py --server.address 127.0.0.1 --server.port 8504
```

Open http://127.0.0.1:8504. The existing upload app can keep running on port 8503.
To run youtube-rag in Docker using the same infrastructure:

```bash
docker compose --env-file .env.media-rag -f compose.media-rag.yaml \
  --profile youtube-rag up -d --build youtube-rag
```

The optional `youtube-rag` profile does not start or replace the original app.
The YouTube image includes FFmpeg, yt-dlp, its matching EJS package, and the
packaged Deno runtime. yt-dlp runs without cookies, application credentials,
proxies, local plugins, or remote EJS component downloads.

## Deploy a new Community Cloud app

After merging the PR, create another Streamlit app using:

- Repository: `mycguo/openai`
- Branch: `main`
- Main file path: `apps/youtube_rag/app.py`
- Python: `3.12`
- App URL: choose `youtube-rag` if that name is available

The adjacent requirements install the YouTube dependencies independently of
other applications. The existing root `packages.txt` supplies FFmpeg. No
separate worker service is needed.

Configure the same Neon infrastructure and provider keys as the existing media
app, under `[youtube_rag]`. An existing `[media_rag]` section is also accepted if
`[youtube_rag]` is absent. Credentials stay in Streamlit secrets, never Git:

```toml
[youtube_rag]
MEDIA_RAG_STORAGE_PROVIDER = "neon"
MEDIA_RAG_STORAGE_BUCKET = "rag"
DATABASE_URL = "<pooled Neon connection string with sslmode=require>"
DATABASE_URL_UNPOOLED = "<direct Neon connection string with sslmode=require>"
AWS_ENDPOINT_URL_S3 = "<Neon storage endpoint>"
AWS_REGION = "<Neon project region>"
AWS_ACCESS_KEY_ID = "<Neon storage access key>"
AWS_SECRET_ACCESS_KEY = "<Neon storage secret key>"
TWELVELABS_API_KEY = "<TwelveLabs API key>"
GEMINI_API_KEY = "<Gemini API key>"
```

`MEDIA_RAG_GEMINI_MODEL` remains optional. Local Neon startup uses
`MEDIA_RAG_ENV_FILE=.env.neon MEDIA_RAG_STORAGE_PROVIDER=neon` as described in
[the Neon guide](media-rag-neon.md).

## Database compatibility

The shared schema adds `media_rag.youtube_sources` and an index on `asset_id`.
Its canonical video ID points to an existing media asset. Existing recordings
and embeddings remain intact; there are no column alterations or re-embedding
steps. Initialization creates the new table/index with `IF NOT EXISTS`.
Test the updated [schema](../media_rag/schema.sql) on a Neon child branch before
applying it to the production library. The database role must be allowed to
initialize the schema; application startup executes that SQL on the direct
endpoint, as the existing app does.

By default both apps use the configured media database and private bucket.
youtube-rag lists and searches only ready assets with a saved YouTube source.
The original media app can also see these recordings. Configure a separate
database and bucket if separate libraries are desired.

## Use the application

1. Open **Add YouTube video**, paste the video URL, and optionally provide a title.
2. Click **Import & index** and keep the page open until processing finishes.
3. Open **Ask library**, ask a question, and choose **Search only** or **Ask & cite**.
4. Play the retrieved/cited clips, or use **Open on YouTube** to open the original
   recording at the matching timestamp.

Watch, short-link, Shorts, embed, and individual live-page URLs are normalized
to one canonical video URL. Playlist and tracking parameters on a video URL are
discarded. Playlist-only URLs, arbitrary websites, credentials in URLs, and
non-HTTPS links are rejected before networking.

Repeated imports of the same video reuse its saved asset and provider IDs.
Content-hash deduplication also works when identical media was uploaded through
the original app. Ready videos are not reindexed. Queued, failed, or interrupted
indexing uses the existing **Index recording**, **Retry indexing**, and
**Resume indexing** controls. A download interrupted before the original is saved
must be imported again; saved indexing work is resumable. Concurrent URL imports
converge on the first saved source mapping, though differing downloaded versions
can leave an extra unlinked media asset.

## Import limits and hosting behavior

Use videos you are allowed to download and send to the model providers. The
importer supports publicly accessible individual videos, up to 60 minutes and
200 MB, with MP4 video at up to 720p and audio when available. Live/upcoming
streams, account-restricted videos, and unknown durations are rejected.
Downloads have a ten-minute deadline and a 600 MB temporary-file budget;
timeouts and oversized downloads stop the entire child process group.

TwelveLabs requires raw media URLs/files, so a YouTube watch URL cannot simply
be passed to its upload endpoint. yt-dlp fetches the original before the app
uploads it directly to TwelveLabs.

YouTube may block downloads from Community Cloud or other data-center IPs.
Installing yt-dlp and Deno does not guarantee access to every public video.
The app reports blocked/unavailable/unsupported-stream errors and does not fall
back to transcript-only indexing. If the hosting IP is blocked, run the same app
locally or on a host permitted to retrieve that video. Previously saved videos
remain searchable without another YouTube download.

Restrict the Cloud app to the trusted user, as in the
[Community Cloud guide](media-rag-community-cloud.md). This version shares one
library; it does not add authentication or tenant isolation. Recordings go to
TwelveLabs for indexing, selected evidence clips go to Gemini for answering,
and private objects use short-lived signed playback URLs.

## Validation

```bash
.venv-youtube-rag/bin/python -m unittest discover -s tests -p 'test_youtube_rag*.py'
.venv-youtube-rag/bin/python -m unittest discover -s tests -p 'test_media_rag*.py'
```

The provider and UI tests are network-free. The opt-in local media integration
suite additionally tests real YouTube source persistence, deduplication,
retrieval, and raw clip extraction using synthetic media and mocked providers:

```bash
MEDIA_RAG_INTEGRATION=1 .venv-youtube-rag/bin/python -m unittest discover -s tests -p 'test_media_rag*.py'
```

References: [TwelveLabs media upload requirements](https://docs.twelvelabs.io/docs/concepts/upload-methods),
[yt-dlp](https://github.com/yt-dlp/yt-dlp),
[JavaScript runtime requirements](https://github.com/yt-dlp/yt-dlp/wiki/EJS).
