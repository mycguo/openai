# Security review: native audio/video RAG (PR #8)

Reviewed on October 8, 2026. Scope: the 22 files changed in the initial revision of
[PR #8](https://github.com/mycguo/openai/pull/8), including configuration,
uploads, subprocesses, provider requests, SQL, playback, and Streamlit rendering.

The GitGuardian alerts on the original Compose lines 10 and 25 were false
positives on required environment-variable interpolation. Neither occurrence
contained a literal password. The local initializer creates random credentials
in an ignored file with mode `0600`; those values were not committed. No
credential rotation is needed for these two occurrences. The PR commits are
consolidated to remove the scanner-triggering expressions from its scanned
history while preserving the final application and security fixes.

The review identified and fixed three concrete issues. The app remains scoped
to a single trusted local user; these changes do not establish a multi-user or
public deployment boundary.

## High severity

### SR-1: uploaded files could select a reference-following demuxer — fixed

Original `media_rag/media.py:27` passed uploads to FFprobe without forcing a
container or restricting protocols. Checking the extension and media kind
after probing did not prevent FFmpeg from recognizing an HLS or concat
playlist hidden behind a supported extension. Such a file could cause the
server to access referenced network resources or local media.

The fix at `media_rag/media.py:21` forces the corresponding supported demuxer,
restricts the input protocol to `file`, and disables MOV external data
references. Both probing and extraction use it. `media_rag/media.py:32` also
allows only basic executable/locale settings into parser subprocesses.

Validation: ordinary WAV and video still pass the native integration flow.
A disguised concat playlist is rejected by both tools. A disguised HLS playlist
referencing a controlled localhost server is rejected without any HTTP request.

## Medium severity

### SR-2: containers inherited unnecessary secrets — fixed

Original `compose.media-rag.yaml:8` injected all of `.env.media-rag` into both
application containers and interpolated the database password into their
connection URL. This put secret values in container environment metadata and
gave the indexing worker a Gemini credential it did not use.

The fix at `compose.media-rag.yaml:8` passes secret paths and a password-free
database URL. Service-specific Compose secrets grant the worker only its
database, MinIO, and TwelveLabs credentials; Gemini is granted to the app only.
PostgreSQL and MinIO use their `_FILE` mechanisms. `media_rag/config.py:22`
reads bounded secret files and assembles the credentialed DSN in memory without
exporting it to the environment. Required unreadable secret files fail closed.
Absent optional provider secrets disable inference and cannot select a fallback
key from another application.

Validation: the rendered Compose service environments have no credential values;
the worker has no Gemini secret. Regression tests cover special characters in
database passwords, unavailable/oversized/invalid secret files, key fallback
isolation, and settings redaction. The existing local database and MinIO volumes
remain accessible with their original credentials.

### SR-3: model output and metadata could render remote Markdown images — fixed

Original `media_rag_app.py:32` inserted provider claim text into Markdown;
source titles were also rendered as Markdown. HTML being disabled does not
prevent Markdown image syntax from triggering a browser request. A generated
image URL could transmit text placed into that URL.

`media_rag/ui.py:6` now escapes Markdown control characters in provider answers,
abstention messages, recording titles, and displayed search questions. The app
retains its own citation formatting while displaying untrusted content as text.
The Streamlit regression exercises an answer and title containing image syntax
and verifies the escaped output.

## Additional hardening

- Streamlit's CORS and XSRF protections are explicit. The Docker launch restricts
  WebSocket hosts to localhost/127.0.0.1; an unexpected Host header returned 403.
- App/worker retain a non-root user, drop all capabilities, prevent privilege
  escalation, and have CPU, memory, PID, and temporary-storage limits. Compose
  environment-backed secrets require a writable container filesystem.
- Originals and clips remain private in MinIO. Signed playback expires after
  15 minutes, and the integration test confirms anonymous access is denied.
- SQL remains parameterized; uploads use generated object IDs and temporary
  filenames rather than user-controlled filesystem paths.

## Validation and limits

All 48 unit, Streamlit, local integration, and security regression tests pass
with `MEDIA_RAG_INTEGRATION=1` using synthetic media. Paid model calls are
substituted. The Docker image and local app are exercised separately to verify
secret mounting and launch settings.

`pip-audit` resolved the application requirements and checked 58 packages;
it reported no known Python dependency vulnerabilities on the review date.
This is a point-in-time check, not an audit of the container OS or MinIO server
binary. The framework skill has no dedicated Streamlit reference; the review
uses the primary references below for framework-specific behavior.

Authentication, tenant boundaries, least-privilege production database/MinIO
accounts, and TLS for external access remain deployment work. The local stack
uses its dedicated administrative infrastructure accounts. Docker administrators
can read mounted secrets. A compromised application process can read the secrets
granted to that service; media restrictions are not a full decoder sandbox.
Model instructions and valid citation IDs do not prove factual support or
eliminate prompt injection. Live provider inference and retention behavior have
not been validated with credentials.

## Primary references

- [Docker Compose secrets](https://docs.docker.com/compose/how-tos/use-secrets/)
- [MinIO secret files](https://github.com/minio/minio/blob/master/docs/docker/README.md)
- [FFmpeg protocol restrictions](https://ffmpeg.org/ffmpeg-protocols.html)
- [FFmpeg format allowlists](https://github.com/FFmpeg/FFmpeg/blob/master/libavformat/options_table.h)
- [Streamlit configuration](https://docs.streamlit.io/develop/api-reference/configuration/config.toml)
