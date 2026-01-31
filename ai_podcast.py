import asyncio
import logging
import os
import re
import tempfile
import time
import urllib.parse
from datetime import datetime

import requests
import streamlit as st
from anthropic import Anthropic
from openai import OpenAI

logger = logging.getLogger(__name__)

# ─── Page Config ─────────────────────────────────────────────────
st.set_page_config(
    page_title="AI Podcast to LinkedIn Article",
    page_icon="🎙️",
    layout="wide",
)

# ─── Secrets / Config ────────────────────────────────────────────
ANTHROPIC_API_KEY = st.secrets.get("ANTHROPIC_API_KEY") or os.getenv("ANTHROPIC_API_KEY")
ANTHROPIC_SONNET_MODEL = (
    st.secrets.get("ANTHROPIC_SONNET_MODEL")
    or os.getenv("ANTHROPIC_SONNET_MODEL")
    or "claude-sonnet-4-5"
)
ASSEMBLYAI_API_KEY = st.secrets.get("ASSEMBLYAI_API_KEY")
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
UPLOAD_ENDPOINT = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_ENDPOINT = "https://api.assemblyai.com/v2/transcript"
CHUNK_SIZE = 5_242_880  # 5 MB

PODCHASER_URL = "https://www.podchaser.com/podcasts/the-ai-daily-brief-artificial-5260567/episodes/recent"

SCRAPED_DIR = "scraped_results"
_SESSION_ARTICLE_CACHE = os.path.join(SCRAPED_DIR, ".pending_article.txt")

# LinkedIn
LINKEDIN_API_URL = "https://api.linkedin.com/rest/posts"
LINKEDIN_API_VERSION = os.getenv("LINKEDIN_API_VERSION", "202509")
LINKEDIN_IMAGE_INIT_URL = "https://api.linkedin.com/rest/images?action=initializeUpload"
DALLE_MODEL = "dall-e-3"
LINKEDIN_AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
LINKEDIN_TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
LINKEDIN_USERINFO_URL = "https://api.linkedin.com/v2/userinfo"
LINKEDIN_ME_URL = "https://api.linkedin.com/v2/me"

if ANTHROPIC_API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


# ─── Playwright helpers ──────────────────────────────────────────

def _ensure_playwright_browsers() -> None:
    """Install Playwright Chromium if not already present."""
    import subprocess
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            path = p.chromium.executable_path
            if os.path.exists(path):
                return
    except Exception:
        pass
    logger.info("Installing Playwright Chromium browser...")
    subprocess.run(["playwright", "install", "chromium"], check=True)


# ─── Step 1: Scrape latest episode ──────────────────────────────

async def _scrape_latest_episode_async():
    """Use Playwright to get the latest episode info + audio URL from Podchaser."""
    from playwright.async_api import async_playwright

    async with async_playwright() as p:
        launch_kwargs = {"headless": True, "args": [
            "--disable-blink-features=AutomationControlled",
        ]}
        for path in ["/usr/bin/chromium", "/usr/bin/chromium-browser",
                     "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable"]:
            if os.path.exists(path):
                launch_kwargs["executable_path"] = path
                break

        browser = await p.chromium.launch(**launch_kwargs)
        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/125.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 800},
            locale="en-US",
        )
        # Remove navigator.webdriver flag
        await context.add_init_script("""
            Object.defineProperty(navigator, 'webdriver', { get: () => undefined });
        """)
        page = await context.new_page()

        # Go to episodes list
        await page.goto(PODCHASER_URL, wait_until="domcontentloaded", timeout=60000)
        await page.wait_for_timeout(5000)

        # Extract latest episode info from the table
        # The page shows a table of episodes; each row has a bold link with the title.
        # The first table row (after the header) is the latest episode.
        episode = await page.evaluate("""
            () => {
                // Table rows: look for the first <a> inside a table row that links to an episode
                const rows = document.querySelectorAll('tr');
                for (const row of rows) {
                    const link = row.querySelector('a[href*="/episodes/"]');
                    if (!link) continue;
                    const title = link.textContent?.trim() || '';
                    const href = link.getAttribute('href') || '';
                    if (!href) continue;
                    // Get description from the next cell
                    const cells = row.querySelectorAll('td');
                    let description = '';
                    if (cells.length >= 2) {
                        description = cells[1]?.textContent?.trim() || '';
                    }
                    return { title, href, description };
                }
                // Fallback: any link containing /episodes/ with a bold child
                const allLinks = document.querySelectorAll('a[href*="/episodes/"]');
                for (const a of allLinks) {
                    const bold = a.querySelector('b, strong');
                    if (bold) {
                        return {
                            title: bold.textContent?.trim() || a.textContent?.trim() || '',
                            href: a.getAttribute('href') || '',
                            description: '',
                        };
                    }
                }
                // Last fallback: first episode link on page
                const first = document.querySelector('a[href*="/episodes/"]');
                if (first) {
                    return {
                        title: first.textContent?.trim() || '',
                        href: first.getAttribute('href') || '',
                        description: '',
                    };
                }
                return null;
            }
        """)

        if not episode or not episode.get("href"):
            await browser.close()
            raise RuntimeError("Could not find any episode on Podchaser page")

        # Build full episode URL
        ep_href = episode["href"]
        if ep_href.startswith("/"):
            ep_href = f"https://www.podchaser.com{ep_href}"

        await browser.close()

        # Extract episode ID from the URL (last numeric segment)
        # e.g. /episodes/100000-ai-agents-joined-their-281403571 -> 281403571
        ep_id = ""
        ep_id_match = re.search(r"-(\d{6,})$", ep_href)
        if ep_id_match:
            ep_id = ep_id_match.group(1)

        # Fetch audio URL via Podchaser API (avoids Cloudflare on episode pages)
        audio_url = ""
        description = episode.get("description", "")
        if ep_id:
            try:
                api_resp = requests.get(
                    f"https://api.podchaser.com/episodes/{ep_id}",
                    headers={
                        "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                                      "AppleWebKit/537.36 (KHTML, like Gecko) "
                                      "Chrome/125.0.0.0 Safari/537.36",
                        "Accept": "application/json",
                    },
                    timeout=15,
                )
                if api_resp.status_code == 200:
                    data = api_resp.json()
                    audio_url = data.get("audio_url", "") or data.get("enclosure_url", "") or ""
                    if not description:
                        description = data.get("description", "") or ""
            except Exception as exc:
                logger.warning("Podchaser API call failed: %s", exc)

        # Fallback: try to get audio URL from the podcast RSS feed
        if not audio_url:
            try:
                rss_resp = requests.get(
                    "https://anchor.fm/s/f7cac464/podcast/rss",
                    headers={"User-Agent": "Mozilla/5.0"},
                    timeout=15,
                )
                if rss_resp.status_code == 200:
                    from xml.etree import ElementTree
                    root = ElementTree.fromstring(rss_resp.content)
                    # Get the first <item> (latest episode)
                    item = root.find(".//item")
                    if item is not None:
                        enclosure = item.find("enclosure")
                        if enclosure is not None:
                            audio_url = enclosure.get("url", "")
                        if not description:
                            desc_el = item.find("description")
                            if desc_el is not None and desc_el.text:
                                description = desc_el.text[:500]
            except Exception as exc:
                logger.warning("RSS feed fallback failed: %s", exc)

        return {
            "title": episode.get("title", "Unknown Episode"),
            "date": episode.get("date", ""),
            "url": ep_href,
            "audio_url": audio_url or "",
            "description": description or "",
        }


def scrape_latest_episode():
    """Sync wrapper for scraping the latest podcast episode."""
    _ensure_playwright_browsers()
    return asyncio.run(_scrape_latest_episode_async())


# ─── Step 2: Download audio ─────────────────────────────────────

def download_audio(url: str, dest_dir: str) -> str:
    """Download audio file from URL to dest_dir, return file path."""
    resp = requests.get(url, stream=True, timeout=120)
    resp.raise_for_status()

    # Determine filename from URL or content-disposition
    filename = "episode_audio.mp3"
    cd = resp.headers.get("content-disposition", "")
    if "filename=" in cd:
        filename = cd.split("filename=")[-1].strip('" ')
    else:
        url_path = urllib.parse.urlparse(url).path
        if url_path and "." in os.path.basename(url_path):
            filename = os.path.basename(url_path)

    filepath = os.path.join(dest_dir, filename)
    total = int(resp.headers.get("content-length", 0))
    downloaded = 0

    progress = st.progress(0, text="Downloading audio...")
    with open(filepath, "wb") as f:
        for chunk in resp.iter_content(chunk_size=CHUNK_SIZE):
            f.write(chunk)
            downloaded += len(chunk)
            if total > 0:
                progress.progress(min(downloaded / total, 1.0), text=f"Downloading... {downloaded // 1024}KB")
    progress.empty()
    return filepath


# ─── Step 3: Transcribe with AssemblyAI ─────────────────────────

def _read_file_in_chunks(filepath: str):
    with open(filepath, "rb") as f:
        while True:
            data = f.read(CHUNK_SIZE)
            if not data:
                break
            yield data


def _upload_audio(filepath: str) -> str:
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    response = requests.post(UPLOAD_ENDPOINT, headers=headers, data=_read_file_in_chunks(filepath))
    response.raise_for_status()
    return response.json()["upload_url"]


def _request_transcription(audio_url: str) -> str:
    headers = {
        "authorization": ASSEMBLYAI_API_KEY,
        "content-type": "application/json",
    }
    payload = {
        "audio_url": audio_url,
        "speaker_labels": True,
        "punctuate": True,
    }
    response = requests.post(TRANSCRIPT_ENDPOINT, json=payload, headers=headers)
    response.raise_for_status()
    return response.json()["id"]


def _poll_transcription(transcript_id: str, placeholder) -> dict:
    headers = {"authorization": ASSEMBLYAI_API_KEY}
    polling_url = f"{TRANSCRIPT_ENDPOINT}/{transcript_id}"

    while True:
        response = requests.get(polling_url, headers=headers)
        response.raise_for_status()
        data = response.json()
        status = data.get("status", "")
        placeholder.info(f"Transcription status: {status}")

        if status == "completed":
            placeholder.empty()
            return data
        if status == "error":
            placeholder.empty()
            raise RuntimeError(data.get("error", "Unknown transcription error"))

        time.sleep(3)


def upload_and_transcribe(filepath: str) -> str:
    """Upload audio to AssemblyAI, transcribe, and return transcript text."""
    with st.spinner("Uploading audio to AssemblyAI..."):
        audio_url = _upload_audio(filepath)

    with st.spinner("Starting transcription..."):
        transcript_id = _request_transcription(audio_url)

    placeholder = st.empty()
    data = _poll_transcription(transcript_id, placeholder)

    # Build transcript with speaker labels if available
    utterances = data.get("utterances")
    if utterances:
        lines = []
        for u in utterances:
            lines.append(f"Speaker {u['speaker']}: {u['text']}")
        transcript = "\n\n".join(lines)
    else:
        transcript = data.get("text", "")

    # Auto-save transcript
    os.makedirs(SCRAPED_DIR, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    save_path = os.path.join(SCRAPED_DIR, f"podcast_transcript_{ts}.txt")
    with open(save_path, "w") as f:
        f.write(transcript)
    st.caption(f"Transcript saved to {save_path}")

    return transcript


# ─── Step 4: Generate LinkedIn article with Claude ──────────────

def _create_anthropic_client() -> Anthropic:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError("Anthropic API key is not configured. Set ANTHROPIC_API_KEY in secrets or env.")
    return Anthropic(api_key=ANTHROPIC_API_KEY)


@st.cache_resource
def _create_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI API key is not configured. Set OPENAI_API_KEY in secrets or env.")
    return OpenAI(api_key=OPENAI_API_KEY)


def generate_with_claude(prompt: str, temperature: float = 1.0, max_tokens: int = 4000) -> str:
    client = _create_anthropic_client()
    response = client.messages.create(
        model=ANTHROPIC_SONNET_MODEL,
        max_tokens=max_tokens,
        temperature=temperature,
        messages=[{"role": "user", "content": [{"type": "text", "text": prompt}]}],
    )
    text_segments = []
    for block in response.content:
        if getattr(block, "type", "") == "text":
            text_segments.append(block.text)
    full_text = "".join(text_segments).strip()
    if full_text:
        return full_text
    raise RuntimeError("Anthropic response did not contain text output")


def generate_linkedin_article(transcript: str, episode_title: str = "", max_attempts: int = 3) -> str:
    prompt = f"""You are a professional LinkedIn content writer specializing in AI and technology.

Analyze the following podcast transcript from "The AI Daily Brief" and create a compelling LinkedIn post.

Episode title: {episode_title}

STRICT REQUIREMENTS:
- Your ENTIRE output must be UNDER 2800 characters (hard limit). Count carefully.
- Do NOT include any preamble, explanation, or notes outside the post itself.
- Output ONLY the LinkedIn post text, nothing else.

Instructions:
1. Identify the top 3-4 most important AI news stories discussed
2. For each story, write 1-2 sentences max
3. Write in a professional but engaging tone suitable for LinkedIn
4. Start with a compelling one-line hook
5. Use short paragraphs and line breaks for readability
6. End with a question to drive engagement
7. Keep it concise — quality over quantity

Transcript:
{transcript[:15000]}
"""
    for attempt in range(max_attempts):
        article = generate_with_claude(prompt, temperature=0.7)
        if len(article) <= 3000:
            return article
        # If over limit, ask Claude to shorten
        shorten_prompt = f"""The following LinkedIn post is {len(article)} characters but MUST be under 2800 characters.
Shorten it while keeping the key insights and engaging tone. Output ONLY the shortened post, nothing else.

{article}"""
        article = generate_with_claude(shorten_prompt, temperature=0.5)
        if len(article) <= 3000:
            return article
    return article


def _build_article_image_prompt(article_text: str, episode_title: str = "") -> str:
    clean_text = re.sub(r"#\w+", "", article_text or "").strip()
    snippet = clean_text.replace("\n", " ")
    snippet = re.sub(r"\s+", " ", snippet)[:500]
    title = episode_title.strip() if episode_title else "AI podcast highlights"
    return (
        "Create a clean, modern, professional LinkedIn image that summarizes an article. "
        f"Episode title: {title}. "
        f"Key themes: {snippet}. "
        "Use bold typography, abstract tech shapes, and a polished gradient background. "
        "No logos, no trademarks, no faces."
    )


def generate_article_image(article_text: str, episode_title: str = "", prompt_override: str = ""):
    """Generate an image for the LinkedIn article using OpenAI's DALL-E 3 model."""
    try:
        if not article_text or not article_text.strip():
            return False, None, None, "No article content available to build an image."

        prompt_for_image = prompt_override.strip() if prompt_override else _build_article_image_prompt(article_text, episode_title)
        client = _create_openai_client()
        response = client.images.generate(
            model=DALLE_MODEL,
            prompt=prompt_for_image,
            size="1024x1024",
            quality="standard",
            n=1,
        )

        if response.data and len(response.data) > 0:
            image_url = response.data[0].url
            image_response = requests.get(image_url, timeout=30)
            if image_response.status_code == 200:
                payload = {
                    "bytes": image_response.content,
                    "mime_type": "image/png",
                    "alt_text": f"Illustration for {episode_title or 'AI podcast highlights'}",
                }
                return True, payload, prompt_for_image, None
            error_msg = f"Failed to download image from URL: {image_url}"
            return False, None, prompt_for_image, error_msg

        return False, None, prompt_for_image, "DALL-E 3 returned no image data"
    except Exception as e:
        return False, None, None, f"Error generating image: {str(e)}"


# ─── Step 5: LinkedIn OAuth & Publish ────────────────────────────

def get_linkedin_config():
    try:
        return {
            "client_id": st.secrets["LINKEDIN_CLIENT_ID"],
            "client_secret": st.secrets["LINKEDIN_CLIENT_SECRET"],
            "redirect_uri": st.secrets["LINKEDIN_REDIRECT_URI"],
        }
    except KeyError as e:
        st.error(f"Missing LinkedIn configuration: {e}")
        return None


def generate_auth_url(config):
    params = {
        "response_type": "code",
        "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"],
        "scope": "w_member_social openid email profile",
        "state": "ai_podcast_app",
    }
    return f"{LINKEDIN_AUTH_URL}?{urllib.parse.urlencode(params)}"


def _get_query_param(params, key):
    if params is None:
        return None
    value = None
    if isinstance(params, dict):
        value = params.get(key)
    else:
        getter = getattr(params, "get", None)
        if callable(getter):
            try:
                value = getter(key)
            except TypeError:
                value = getter(key, None)
    if isinstance(value, (list, tuple)):
        return value[0] if value else None
    return value


def exchange_code_for_token(code, config):
    params = {
        "grant_type": "authorization_code",
        "code": code,
        "redirect_uri": config["redirect_uri"],
        "client_id": config["client_id"],
        "client_secret": config["client_secret"],
    }
    try:
        response = requests.post(LINKEDIN_TOKEN_URL, data=params)
        if response.status_code == 200:
            return response.json()
        st.error(f"Token exchange failed: {response.status_code} - {response.text}")
        return None
    except requests.RequestException as e:
        st.error(f"Failed to get access token: {e}")
        return None


def fetch_authenticated_member_urn(access_token):
    try:
        resp = requests.get(LINKEDIN_USERINFO_URL, headers={"Authorization": f"Bearer {access_token}"})
        if resp.status_code == 200:
            subject = resp.json().get("sub") or resp.json().get("id")
            if subject:
                subject = str(subject)
                return subject if subject.startswith("urn:li:person:") else f"urn:li:person:{subject}"

        headers = {
            "Authorization": f"Bearer {access_token}",
            "X-Restli-Protocol-Version": "2.0.0",
            "Linkedin-Version": LINKEDIN_API_VERSION,
        }
        resp = requests.get(LINKEDIN_ME_URL, headers=headers, params={"projection": "(id)"})
        if resp.status_code == 200:
            member_id = resp.json().get("id")
            if member_id:
                return f"urn:li:person:{member_id}"
        return None
    except requests.RequestException:
        return None


def initialize_linkedin_image_upload(access_token: str, owner_urn: str):
    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
        "Linkedin-Version": LINKEDIN_API_VERSION,
    }
    payload = {"initializeUploadRequest": {"owner": owner_urn}}
    response = requests.post(LINKEDIN_IMAGE_INIT_URL, json=payload, headers=headers, timeout=30)
    if response.status_code >= 400:
        return False, None, None, f"{response.status_code}: {response.text}"
    data = response.json()
    value = data.get("value", {})
    upload_url = value.get("uploadUrl")
    image_urn = value.get("image")
    if not upload_url or not image_urn:
        return False, None, None, "LinkedIn image initialize response missing uploadUrl or image URN."
    return True, upload_url, image_urn, None


def upload_linkedin_image(upload_url: str, image_bytes: bytes, mime_type: str):
    headers = {"Content-Type": mime_type or "image/png"}
    response = requests.put(upload_url, data=image_bytes, headers=headers, timeout=60)
    if response.status_code >= 400:
        return False, f"{response.status_code}: {response.text}"
    return True, None


def post_to_linkedin(content, access_token, author_id, image_payload=None):
    if not author_id:
        return False, "Missing LinkedIn author ID. Please reconnect your account."

    author_urn = author_id if author_id.startswith("urn:") else f"urn:li:person:{author_id}"

    if author_urn.endswith(":~"):
        return False, "Invalid LinkedIn author identifier. Please reconnect your account."

    headers = {
        "Authorization": f"Bearer {access_token}",
        "Content-Type": "application/json",
        "X-Restli-Protocol-Version": "2.0.0",
        "Linkedin-Version": LINKEDIN_API_VERSION,
    }
    payload = {
        "author": author_urn,
        "commentary": content,
        "visibility": "PUBLIC",
        "distribution": {
            "feedDistribution": "MAIN_FEED",
            "targetEntities": [],
            "thirdPartyDistributionChannels": [],
        },
        "lifecycleState": "PUBLISHED",
        "isReshareDisabledByAuthor": False,
    }
    if image_payload:
        if not image_payload.get("bytes"):
            return False, "Image payload is empty."
        init_ok, upload_url, image_urn, init_error = initialize_linkedin_image_upload(
            access_token,
            author_urn,
        )
        if not init_ok:
            return False, f"Image upload init failed: {init_error}"
        upload_ok, upload_error = upload_linkedin_image(
            upload_url,
            image_payload.get("bytes", b""),
            image_payload.get("mime_type", "image/png"),
        )
        if not upload_ok:
            return False, f"Image upload failed: {upload_error}"
        alt_text = image_payload.get("alt_text") or "AI podcast illustration"
        payload["content"] = {"media": {"id": image_urn, "altText": alt_text}}
    try:
        response = requests.post(LINKEDIN_API_URL, json=payload, headers=headers)
        if response.status_code >= 400:
            logger.error("LinkedIn API error: %s %s", response.status_code, response.text)
            return False, f"{response.status_code}: {response.text}"
        if response.status_code == 201:
            post_id = response.headers.get("x-restli-id") or response.headers.get("X-Restli-Id")
            if post_id:
                return True, {"id": post_id}
        response.raise_for_status()
        if response.content:
            return True, response.json()
        return True, {}
    except requests.RequestException as e:
        return False, str(e)


def build_linkedin_post_url(post_urn: str) -> str:
    """Best-effort public URL for a LinkedIn post URN."""
    if not post_urn:
        return ""
    urn = post_urn.strip()
    if urn.startswith("urn:li:activity:"):
        return f"https://www.linkedin.com/feed/update/{urn}/"
    # Some posts can still resolve via feed/update with the URN, but this isn't guaranteed.
    if urn.startswith("urn:li:share:") or urn.startswith("urn:li:ugcPost:"):
        return f"https://www.linkedin.com/feed/update/{urn}/"
    return ""


# ─── LinkedIn OAuth callback handler ────────────────────────────

def _handle_linkedin_oauth():
    """Process LinkedIn OAuth callback if present."""
    config = get_linkedin_config()
    if not config:
        return

    try:
        query_params = st.query_params
    except AttributeError:
        query_params = st.experimental_get_query_params()

    code_param = _get_query_param(query_params, "code")
    state_param = _get_query_param(query_params, "state")
    token_active = (
        "linkedin_token" in st.session_state
        and time.time() < st.session_state.get("token_expires", 0)
    )

    if code_param and state_param == "ai_podcast_app" and not token_active:
        with st.spinner("Exchanging authorization code..."):
            token_data = exchange_code_for_token(code_param, config)
            if token_data and "access_token" in token_data:
                st.session_state.linkedin_token = token_data["access_token"]
                st.session_state.token_expires = time.time() + token_data.get("expires_in", 5184000)
                member_urn = fetch_authenticated_member_urn(st.session_state.linkedin_token)
                if member_urn:
                    st.session_state.author_id = member_urn
                # Restore article saved before OAuth redirect
                if os.path.exists(_SESSION_ARTICLE_CACHE):
                    with open(_SESSION_ARTICLE_CACHE) as f:
                        cached = f.read()
                    if cached.strip():
                        st.session_state.article = cached
                        st.session_state.article_editor = cached
                    os.remove(_SESSION_ARTICLE_CACHE)
                try:
                    st.query_params.clear()
                except AttributeError:
                    st.experimental_set_query_params()
                st.success("Connected to LinkedIn!")
                st.rerun()


# ─── Main App ────────────────────────────────────────────────────

def main():
    st.title("🎙️ AI Podcast to LinkedIn Article")
    st.markdown("Scrape **podcast**, transcribe, generate a LinkedIn article, and publish.")

    _handle_linkedin_oauth()

    # ── Connect LinkedIn Account ──
    st.subheader("Connect LinkedIn Account")
    token_active = (
        "linkedin_token" in st.session_state
        and time.time() < st.session_state.get("token_expires", 0)
    )

    if not token_active:
        config = get_linkedin_config()
        if config:
            if st.button("Connect LinkedIn Account"):
                # Save article to disk so it survives the OAuth redirect
                article = st.session_state.get("article", "").strip()
                if article:
                    os.makedirs(SCRAPED_DIR, exist_ok=True)
                    with open(_SESSION_ARTICLE_CACHE, "w") as f:
                        f.write(article)
                auth_url = generate_auth_url(config)
                st.markdown(f"[Click here to authorize]({auth_url})")
    else:
        st.success("LinkedIn connected!")
        if st.button("Disconnect LinkedIn"):
            for key in ["linkedin_token", "token_expires", "author_id"]:
                st.session_state.pop(key, None)
            st.rerun()

    # ── Step 1: Fetch Episode ──
    st.subheader("Step 1: Fetch Latest Episode")
    if st.button("Fetch Latest Episode", type="primary"):
        with st.spinner("Scraping Podchaser for latest episode..."):
            try:
                episode = scrape_latest_episode()
                st.session_state.episode = episode
                st.success(f"Found: **{episode['title']}**")
                logger.info("Episode URL: %s", episode.get("url", ""))
                logger.info("Audio URL: %s", episode.get("audio_url", ""))
                st.info(f"Episode URL: {episode.get('url', 'N/A')}")
                st.info(f"Audio URL: {episode.get('audio_url', 'N/A') or 'Not found'}")
            except Exception as e:
                st.error(f"Failed to scrape episode: {e}")

    if "episode" in st.session_state:
        ep = st.session_state.episode
        st.markdown(f"**Title:** {ep['title']}")
        if ep.get("date"):
            st.markdown(f"**Date:** {ep['date']}")
        if ep.get("description"):
            st.markdown(f"**Description:** {ep['description'][:300]}")
        st.markdown(f"**Episode URL:** {ep['url']}")
        if ep.get("audio_url"):
            st.markdown(f"**Audio URL:** `{ep['audio_url'][:100]}...`")
        else:
            st.warning("No audio URL found automatically. You can paste one below.")
            manual_url = st.text_input("Paste audio URL (mp3/m4a):", key="manual_audio_url")
            if manual_url:
                st.session_state.episode["audio_url"] = manual_url

    st.markdown("**— or paste an audio URL directly —**")
    col_url, col_btn = st.columns([3, 1])
    with col_url:
        direct_url = st.text_input("Audio URL (mp3/m4a):", key="direct_audio_url", label_visibility="collapsed", placeholder="Paste audio URL here...")
    with col_btn:
        if st.button("Use URL", disabled=not direct_url):
            st.session_state.episode = {
                "title": "Manual Audio",
                "date": "",
                "url": "",
                "audio_url": direct_url,
                "description": "",
            }
            st.success("Audio URL set!")
            st.rerun()

    # ── Step 2: Transcribe ──
    st.subheader("Step 2: Transcribe Audio")
    audio_url = st.session_state.get("episode", {}).get("audio_url", "")
    can_transcribe = bool(audio_url)

    # Load a previously saved transcript
    saved_transcripts = sorted(
        [f for f in os.listdir(SCRAPED_DIR) if f.startswith("podcast_transcript_") and f.endswith(".txt")]
        if os.path.isdir(SCRAPED_DIR) else [],
        reverse=True,
    )
    if saved_transcripts:
        selected_transcript = st.selectbox("Load saved transcript", ["(none)"] + saved_transcripts, key="transcript_selector")
        if selected_transcript != "(none)":
            if st.button("Load Transcript"):
                with open(os.path.join(SCRAPED_DIR, selected_transcript)) as f:
                    st.session_state.transcript = f.read()
                st.success(f"Loaded {selected_transcript}")
                st.rerun()

    col1, col2 = st.columns(2)
    with col1:
        save_audio = st.checkbox("Save audio to disk", value=True)
    with col2:
        if st.button("Download & Transcribe", disabled=not can_transcribe):
            if not ASSEMBLYAI_API_KEY:
                st.error("ASSEMBLYAI_API_KEY not configured in secrets.")
            else:
                try:
                    if save_audio:
                        os.makedirs(SCRAPED_DIR, exist_ok=True)
                        audio_dir = SCRAPED_DIR
                    else:
                        audio_dir = tempfile.mkdtemp()

                    st.info("Downloading audio...")
                    filepath = download_audio(audio_url, audio_dir)
                    file_size = os.path.getsize(filepath)
                    st.success(f"Downloaded: {os.path.basename(filepath)} ({file_size // 1024}KB)")

                    if save_audio:
                        # Rename with timestamp for easy identification
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        ext = os.path.splitext(filepath)[1] or ".mp3"
                        saved_name = f"podcast_audio_{ts}{ext}"
                        saved_path = os.path.join(SCRAPED_DIR, saved_name)
                        if filepath != saved_path:
                            os.rename(filepath, saved_path)
                            filepath = saved_path
                        st.caption(f"Audio saved to {saved_path}")

                    transcript = upload_and_transcribe(filepath)
                    st.session_state.transcript = transcript
                    st.success("Transcription complete!")

                    # Clean up temp dir if not saving
                    if not save_audio:
                        import shutil
                        shutil.rmtree(audio_dir, ignore_errors=True)
                except Exception as e:
                    st.error(f"Transcription failed: {e}")

    if "transcript" in st.session_state:
        with st.expander("View Transcript", expanded=False):
            st.text_area("Transcript", st.session_state.transcript, height=300, disabled=True)

    # ── Step 3: Generate Article ──
    st.subheader("Step 3: Generate LinkedIn Article")
    has_transcript = "transcript" in st.session_state

    # Load a previously saved article
    saved_articles = sorted(
        [f for f in os.listdir(SCRAPED_DIR) if f.startswith("podcast_article_") and f.endswith(".txt")]
        if os.path.isdir(SCRAPED_DIR) else [],
        reverse=True,
    )
    if saved_articles:
        selected_file = st.selectbox("Load saved article", ["(none)"] + saved_articles)
        if selected_file != "(none)":
            if st.button("Load"):
                with open(os.path.join(SCRAPED_DIR, selected_file)) as f:
                    content = f.read()
                st.session_state.article = content
                st.session_state.article_editor = content
                st.success(f"Loaded {selected_file}")
                st.rerun()

    if st.button("Generate Article", disabled=not has_transcript):
        try:
            ep_title = st.session_state.get("episode", {}).get("title", "")
            with st.spinner("Generating article with Claude..."):
                article = generate_linkedin_article(st.session_state.transcript, ep_title)
            st.session_state.article = article
            st.session_state.article_editor = article
            # Auto-save article
            os.makedirs(SCRAPED_DIR, exist_ok=True)
            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            save_path = os.path.join(SCRAPED_DIR, f"podcast_article_{ts}.txt")
            with open(save_path, "w") as f:
                f.write(article)
            st.success(f"Article generated! Saved to {save_path}")
        except Exception as e:
            st.error(f"Generation failed: {e}")

    if "article" in st.session_state and st.session_state.article:
        # Use the article value as default for a keyed widget; sync back on change
        if "article_editor" not in st.session_state:
            st.session_state.article_editor = st.session_state.article
        edited = st.text_area(
            "Edit Article",
            height=400,
            key="article_editor",
        )
        char_count = len(edited)
        if char_count > 3000:
            st.warning(f"{char_count}/3000 characters — over LinkedIn's limit. Shorten before publishing.")
        else:
            st.caption(f"{char_count}/3000 characters")
        st.session_state.article = edited

    st.markdown("**Optional: Generate a LinkedIn image from the article**")
    if not OPENAI_API_KEY:
        st.info("Set OPENAI_API_KEY to enable image generation.")
    else:
        has_article = bool(st.session_state.get("article", "").strip())
        if "article_image_prompt" not in st.session_state:
            st.session_state.article_image_prompt = ""
        ep_title = st.session_state.get("episode", {}).get("title", "")
        default_prompt = _build_article_image_prompt(
            st.session_state.get("article", ""),
            ep_title,
        )
        prompt_col, reset_col = st.columns([5, 1])
        with prompt_col:
            prompt_value = st.text_area(
                "Image prompt",
                value=st.session_state.article_image_prompt or default_prompt,
                height=140,
            )
        with reset_col:
            if st.button("Reset prompt"):
                st.session_state.article_image_prompt = default_prompt
                st.rerun()
        st.session_state.article_image_prompt = prompt_value
        if st.button("Generate Article Image", disabled=not has_article):
            with st.spinner("Generating image from article..."):
                success, image_payload, prompt_used, error = generate_article_image(
                    st.session_state.article,
                    ep_title,
                    prompt_override=st.session_state.article_image_prompt,
                )
            if success and image_payload:
                st.session_state.article_image = image_payload
                st.session_state.article_image_prompt = prompt_used
                st.success("Image generated!")
            else:
                st.error(error or "Failed to generate image.")

        if st.session_state.get("article_image"):
            st.image(
                st.session_state.article_image["bytes"],
                caption="Generated image",
                use_container_width=True,
            )
            st.download_button(
                label="Download image",
                data=st.session_state.article_image["bytes"],
                file_name="linkedin_article_image.png",
                mime=st.session_state.article_image.get("mime_type", "image/png"),
            )
            if st.button("Clear image"):
                st.session_state.pop("article_image", None)
                st.session_state.pop("article_image_prompt", None)
                st.rerun()

    # ── Step 4: Publish to LinkedIn ──
    st.subheader("Step 4: Publish to LinkedIn")

    if not token_active:
        st.warning("LinkedIn not connected.")
        st.info("Connect your LinkedIn account above to enable publishing.")
    else:
        st.success("LinkedIn connected!")
        include_image = False
        if st.session_state.get("article_image"):
            include_image = st.checkbox("Include generated image in post", value=True)
        if st.button("Publish to LinkedIn", type="primary"):
            content = st.session_state.get("article", "").strip()
            if not content:
                st.error("No article to publish. Generate or load one first.")
            elif len(content) > 3000:
                st.error(f"Article is {len(content)} characters. LinkedIn's limit is 3000. Please shorten it.")
            else:
                with st.spinner("Publishing..."):
                    image_payload = st.session_state.article_image if include_image else None
                    success, result = post_to_linkedin(
                        content,
                        st.session_state.linkedin_token,
                        st.session_state.author_id,
                        image_payload=image_payload,
                    )
                if success:
                    st.success("Published to LinkedIn!")
                    post_id = ""
                    if isinstance(result, dict):
                        post_id = result.get("id") or result.get("post_id") or ""
                    if post_id:
                        st.session_state.last_linkedin_post_urn = post_id
                        st.info(f"Post URN: `{post_id}`")
                        post_url = build_linkedin_post_url(post_id)
                        if post_url:
                            st.session_state.last_linkedin_post_url = post_url
                            st.text_input("Post URL (copy)", value=post_url)
                        else:
                            st.caption("Public post URL not available for this URN type.")
                    st.balloons()
                else:
                    st.error(f"Failed to publish: {result}")

if __name__ == "__main__":
    main()
