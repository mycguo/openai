# YouTube transcript app

The existing `youtube.py` Streamlit app now defaults to retrieving captions. It does
not require credentials or send audio to a paid provider in this mode.

## Run locally

From the repository root, with its dependencies installed:

```sh
python -m pip install -r requirements.txt
python -m streamlit run youtube.py --server.address 127.0.0.1 --server.port 8502
```

On this Mac the tested interpreter is `.venv/bin/python3.14`.

## Flow

1. Paste a watch, youtu.be, Shorts, embed, or live-video URL.
2. Enter preferred caption language codes, such as `en` or `es,en`.
3. Click Fetch Transcript. The app fetches available captions using
   `youtube-transcript-api==1.2.4`, preferring manually created tracks when available.
4. The complete text remains in the session without being rendered on screen.
   Download it as TXT or SRT; SRT keeps the caption timestamps. Source and
   language appear above the download controls.

This retrieves captions, not newly recognized speech. Auto-generated captions may
contain errors. Missing captions, unavailable languages, restricted videos, and
YouTube network blocking produce errors; there is no automatic paid fallback.
Public cloud IPs can be blocked, and running locally is not a guarantee of access.

## Optional audio transcription

The existing AssemblyAI audio workflow remains a separate mode. Configure
`ASSEMBLYAI_API_KEY` through the environment or `.streamlit/secrets.toml`, select
audio mode, and explicitly consent to sending the audio. Provider charges may
apply. Only process content you have permission to use. Download restrictions can
still prevent this path from working. No paid calls are made by the offline tests.

Results stay in the current Streamlit session unless you download them or click
Upload to Transcript Section, which saves a TXT file under `Data/transcripts`.
The existing optional summary feature is separate and requires its own API key;
this change does not alter that feature's transcript-excerpt behavior.

## Article, image, and LinkedIn publishing

`youtube.py` remains the entry point. `youtube_publishing.py` contains the new UI
and imports the existing article, image, and publishing services from `ai_podcast.py`.
Importing the podcast module does not configure another page or initialize its database.

1. Paste a YouTube URL. You can retrieve the transcript separately, or click
   **Generate Article** immediately; when no transcript is loaded, the app fetches
   it first and continues directly into article generation.
2. Edit **Article generation prompt**, or keep the default. **Reset article prompt**
   restores the default. The full transcript is automatically appended to the
   instructions. Prompts are session-local, not saved to disk.
3. **Generate Article** calls Claude using `ANTHROPIC_API_KEY` and the existing
   `ANTHROPIC_SONNET_MODEL` setting. Review/edit the resulting text and download it.
4. Optionally edit the image prompt and click **Generate Article Image**. This uses
   the podcast app's Google image model (`GOOGLE_API_KEY`, optional
   `GOOGLE_IMAGE_MODEL`) and Claude theme extraction. The edited article is used.
5. Connect LinkedIn, select whether to include the image, confirm public publishing,
   and click **Publish to LinkedIn**. This creates a feed post, not a LinkedIn
   long-form article/newsletter. Provider validation errors are shown to the user.

Changing the transcript clears article/image outputs; editing the article clears
the image and publishing confirmation. Caption retrieval never generates content
or publishes automatically. Image upload failures do not silently publish text-only.
Duplicate identical posts are blocked within the session. If a request times out,
check LinkedIn before explicitly permitting a retry. This is not cross-session
idempotency: refreshing/restarting the app can lose this protection.

### LinkedIn connection setup

Reuse `LINKEDIN_CLIENT_ID` and `LINKEDIN_CLIENT_SECRET`. Register the YouTube app's
exact redirect URL in the LinkedIn developer console (including trailing slash),
for example `http://localhost:8502/`. Optionally set
`YOUTUBE_LINKEDIN_REDIRECT_URI` to that URL. Otherwise the current app URL is used;
the podcast app's `LINKEDIN_REDIRECT_URI` is deliberately not reused.

Click **Start LinkedIn connection**, then open authorization in a new tab. After
LinkedIn redirects back, copy the complete callback URL into the password-masked
field in the original tab and click **Complete connection**. Do not share that URL.
This preserves the draft/session while enforcing random, single-use OAuth state
with a ten-minute expiry. Tokens stay in that session and are cleared on disconnect.
The LinkedIn app needs `w_member_social`, `openid`, and `profile` permissions.
The existing `LINKEDIN_API_VERSION` setting is shared with the podcast app.

Live OAuth and public posting must be verified by the user; automated tests mock
provider responses and never generate billable content or publish real posts.

## Tests

```sh
.venv/bin/python3.14 -m unittest discover -s tests -p 'test_youtube*.py' -v
```

Tests mock network retrieval and exercise URL validation, complete long text,
timestamp rounding, errors, missing secrets, consent, and stale-result clearing.
