import asyncio
import html
import logging
import os
import re
import tempfile
import time
import urllib.parse
from datetime import datetime
from email.utils import parsedate_to_datetime
from typing import Optional, List, Dict, Any

import requests
import streamlit as st
from anthropic import Anthropic
from google import genai
from google.genai import types

logger = logging.getLogger(__name__)

# ─── Cloud Detection & Database ───────────────────────────────────
IS_STREAMLIT_CLOUD = os.getenv("STREAMLIT_RUNTIME_ENV") == "cloud" or os.getenv("STREAMLIT_SHARING_MODE") is not None

NEON_DATABASE_URL = "postgresql://neondb_owner:npg_J0ctsQkMWb1L@ep-shy-meadow-akzmy4xi-pooler.c-3.us-west-2.aws.neon.tech/neondb?sslmode=require&channel_binding=require"


def _get_db_connection():
    """Get a PostgreSQL database connection for cloud storage."""
    try:
        import psycopg2
        conn = psycopg2.connect(NEON_DATABASE_URL)
        return conn
    except ImportError:
        logger.error("psycopg2 not installed. Run: pip install psycopg2-binary")
        return None
    except Exception as e:
        logger.error("Failed to connect to database: %s", e)
        return None


def _init_db_tables():
    """Initialize database tables if they don't exist."""
    conn = _get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                CREATE TABLE IF NOT EXISTS podcast_articles (
                    id SERIAL PRIMARY KEY,
                    filename VARCHAR(255) UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    article_type VARCHAR(50) DEFAULT 'article',
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
            cur.execute("""
                CREATE TABLE IF NOT EXISTS oauth_state_cache (
                    id SERIAL PRIMARY KEY,
                    cache_key VARCHAR(255) UNIQUE NOT NULL,
                    content TEXT NOT NULL,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    expires_at TIMESTAMP
                )
            """)
            conn.commit()
        return True
    except Exception as e:
        logger.error("Failed to initialize database tables: %s", e)
        return False
    finally:
        conn.close()


def _db_save_article(filename: str, content: str, article_type: str = "article") -> bool:
    """Save an article to the database."""
    conn = _get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO podcast_articles (filename, content, article_type)
                VALUES (%s, %s, %s)
                ON CONFLICT (filename) DO UPDATE SET content = EXCLUDED.content, created_at = CURRENT_TIMESTAMP
            """, (filename, content, article_type))
            conn.commit()
        logger.info("Saved article to database: %s", filename)
        return True
    except Exception as e:
        logger.error("Failed to save article to database: %s", e)
        return False
    finally:
        conn.close()


def _db_load_article(filename: str) -> Optional[str]:
    """Load an article from the database."""
    conn = _get_db_connection()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("SELECT content FROM podcast_articles WHERE filename = %s", (filename,))
            row = cur.fetchone()
            if row:
                return row[0]
        return None
    except Exception as e:
        logger.error("Failed to load article from database: %s", e)
        return None
    finally:
        conn.close()


def _db_list_articles(article_type: str = "article", limit: int = 50) -> List[str]:
    """List article filenames from the database."""
    conn = _get_db_connection()
    if not conn:
        return []
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT filename FROM podcast_articles
                WHERE article_type = %s
                ORDER BY created_at DESC
                LIMIT %s
            """, (article_type, limit))
            rows = cur.fetchall()
            return [row[0] for row in rows]
    except Exception as e:
        logger.error("Failed to list articles from database: %s", e)
        return []
    finally:
        conn.close()


def _db_save_oauth_cache(cache_key: str, content: str) -> bool:
    """Save OAuth state to the database."""
    conn = _get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("""
                INSERT INTO oauth_state_cache (cache_key, content, expires_at)
                VALUES (%s, %s, CURRENT_TIMESTAMP + INTERVAL '1 hour')
                ON CONFLICT (cache_key) DO UPDATE SET content = EXCLUDED.content, expires_at = CURRENT_TIMESTAMP + INTERVAL '1 hour'
            """, (cache_key, content))
            conn.commit()
        return True
    except Exception as e:
        logger.error("Failed to save OAuth cache to database: %s", e)
        return False
    finally:
        conn.close()


def _db_load_oauth_cache(cache_key: str) -> Optional[str]:
    """Load OAuth state from the database."""
    conn = _get_db_connection()
    if not conn:
        return None
    try:
        with conn.cursor() as cur:
            cur.execute("""
                SELECT content FROM oauth_state_cache
                WHERE cache_key = %s AND expires_at > CURRENT_TIMESTAMP
            """, (cache_key,))
            row = cur.fetchone()
            if row:
                return row[0]
        return None
    except Exception as e:
        logger.error("Failed to load OAuth cache from database: %s", e)
        return None
    finally:
        conn.close()


def _db_delete_oauth_cache(cache_key: str) -> bool:
    """Delete OAuth state from the database."""
    conn = _get_db_connection()
    if not conn:
        return False
    try:
        with conn.cursor() as cur:
            cur.execute("DELETE FROM oauth_state_cache WHERE cache_key = %s", (cache_key,))
            conn.commit()
        return True
    except Exception as e:
        logger.error("Failed to delete OAuth cache from database: %s", e)
        return False
    finally:
        conn.close()


# Initialize database tables on cloud
if IS_STREAMLIT_CLOUD:
    _init_db_tables()

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
GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
UPLOAD_ENDPOINT = "https://api.assemblyai.com/v2/upload"
TRANSCRIPT_ENDPOINT = "https://api.assemblyai.com/v2/transcript"
CHUNK_SIZE = 5_242_880  # 5 MB

PODCAST_SOURCES = {
    "The AI Daily Brief": {
        "episodes_url": "https://www.podchaser.com/podcasts/the-ai-daily-brief-artificial-5260567/episodes/recent",
        "rss_url": "https://anchor.fm/s/f7cac464/podcast/rss",
    },
    "Y Combinator Startup Podcast": {
        "episodes_url": "https://www.podchaser.com/podcasts/y-combinator-startup-podcast-526094/episodes/recent",
        "rss_url": "https://anchor.fm/s/8c1524bc/podcast/rss",
    },
    "AI Fire Daily": {
        "episodes_url": "https://www.podchaser.com/podcasts/ai-fire-daily-6108838/episodes/recent",
        "rss_url": "https://media.rss.com/ai-fire-daily/feed.xml",
    },
    "Latent Space: the AI engineer Podcast": {
        "episodes_url": "https://www.podchaser.com/podcasts/latent-space-the-ai-engineer-p-5164925/episodes/recent",
        "rss_url": "https://rss.flightcast.com/vgnxzgiwwzwke85ym53fjnzu.xml",
    },
    "Lenny's podcast": {
        "episodes_url": "https://www.podchaser.com/podcasts/lennys-podcast-product-career-4750705/episodes/recent",
        "rss_url": "https://api.substack.com/feed/podcast/10845.rss",
    },
    "No Priors": {
        "episodes_url": "https://www.podchaser.com/podcasts/no-priors-artificial-intellige-5096296/episodes/recent",
        "rss_url": "https://feeds.megaphone.fm/nopriors",
    },
    "The Tennis Podcast": {
        "episodes_url": "https://www.podchaser.com/podcasts/the-tennis-podcast-31788/episodes/recent",
        "rss_url": "https://feeds.acast.com/public/shows/thetennispodcast",
    },
    "AI News Today": {
        "episodes_url": "https://podcasts.apple.com/us/podcast/ai-news-today-julian-goldie-podcast/id1851256047",
        "rss_url": "https://anchor.fm/s/10b0edd94/podcast/rss",
    },
    "How I AI": {
        "episodes_url": "https://www.podchaser.com/podcasts/how-i-ai-6074236/episodes/recent",
        "rss_url": "https://anchor.fm/s/1035b1568/podcast/rss",
    },
    "Chain of Thought": {
        "episodes_url": "https://www.podchaser.com/podcasts/chain-of-thought-ai-agents-inf-5932316/episodes/recent",
        "rss_url": "https://feeds.transistor.fm/chain-of-thought",
    },
    "TBPN": {
        "episodes_url": "https://www.podchaser.com/podcasts/tbpn-5850620/episodes/recent",
        "rss_url": "https://feeds.transistor.fm/technology-brother",
    },
}
ENABLE_BROWSER_SCRAPE = os.getenv("ENABLE_PLAYWRIGHT_SCRAPE", "").lower() in {"1", "true", "yes"}

SCRAPED_DIR = "scraped_results"
_SESSION_ARTICLE_CACHE = os.path.join(SCRAPED_DIR, ".pending_article.txt")
_SESSION_SOURCE_CACHE = os.path.join(SCRAPED_DIR, ".pending_source.json")

# LinkedIn
LINKEDIN_API_URL = "https://api.linkedin.com/rest/posts"
LINKEDIN_API_VERSION = os.getenv("LINKEDIN_API_VERSION", "202509")
LINKEDIN_IMAGE_INIT_URL = "https://api.linkedin.com/rest/images?action=initializeUpload"
NANO_BANANA_MODEL = (
    st.secrets.get("GOOGLE_IMAGE_MODEL")
    or os.getenv("GOOGLE_IMAGE_MODEL")
    or "gemini-3.1-flash-image-preview"
)
LINKEDIN_AUTH_URL = "https://www.linkedin.com/oauth/v2/authorization"
LINKEDIN_TOKEN_URL = "https://www.linkedin.com/oauth/v2/accessToken"
LINKEDIN_USERINFO_URL = "https://api.linkedin.com/v2/userinfo"
LINKEDIN_ME_URL = "https://api.linkedin.com/v2/me"

if ANTHROPIC_API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY
if GOOGLE_API_KEY:
    os.environ["GOOGLE_API_KEY"] = GOOGLE_API_KEY


# ─── Playwright helpers ──────────────────────────────────────────

def _ensure_playwright_browsers() -> bool:
    """Best-effort install for Playwright Chromium. Returns True when usable."""
    import subprocess
    try:
        from playwright.sync_api import sync_playwright
        with sync_playwright() as p:
            path = p.chromium.executable_path
            if os.path.exists(path):
                return True
    except Exception:
        logger.warning("Playwright is unavailable in this environment.")
        return False

    logger.info("Installing Playwright Chromium browser...")
    try:
        subprocess.run(["playwright", "install", "chromium"], check=True)
        return True
    except Exception as exc:
        logger.warning("Playwright browser install failed: %s", exc)
        return False


def _resolve_episode_audio(
    ep_href: str,
    description: str = "",
    rss_url: str = "",
) -> tuple[str, str]:
    """Resolve audio URL via Podchaser episode API, with optional podcast RSS fallback."""
    ep_id = ""
    ep_id_match = re.search(r"-(\d{6,})$", ep_href)
    if ep_id_match:
        ep_id = ep_id_match.group(1)

    audio_url = ""
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

    if not audio_url and rss_url:
        try:
            rss_resp = requests.get(
                rss_url,
                headers={"User-Agent": "Mozilla/5.0"},
                timeout=15,
            )
            if rss_resp.status_code == 200:
                from xml.etree import ElementTree

                root = ElementTree.fromstring(rss_resp.content)
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

    return audio_url, description


def _clean_html_text(value: str) -> str:
    if not value:
        return ""
    from bs4 import BeautifulSoup

    return BeautifulSoup(value, "html.parser").get_text(" ", strip=True)


def _format_pub_date(value: str) -> str:
    if not value:
        return ""
    try:
        return parsedate_to_datetime(value).strftime("%Y-%m-%d")
    except Exception:
        return value


def _scrape_latest_episode_rss(source_name: str, source_config: Dict[str, str]) -> dict:
    """Fetch latest episode info from RSS to avoid Podchaser HTML blocking."""
    from xml.etree import ElementTree

    rss_url = source_config.get("rss_url", "").strip()
    if not rss_url:
        raise RuntimeError(f"RSS URL is not configured for {source_name}.")

    response = requests.get(
        rss_url,
        headers={"User-Agent": "Mozilla/5.0"},
        timeout=25,
    )
    response.raise_for_status()

    root = ElementTree.fromstring(response.content)
    item = root.find(".//channel/item")
    if item is None:
        raise RuntimeError(f"No episodes found in RSS feed for {source_name}.")

    title = (item.findtext("title") or "Unknown Episode").strip()
    description = _clean_html_text(item.findtext("description") or "")
    episode_url = (item.findtext("link") or "").strip()
    pub_date = _format_pub_date((item.findtext("pubDate") or "").strip())

    audio_url = ""
    enclosure = item.find("enclosure")
    if enclosure is not None:
        audio_url = (enclosure.get("url") or "").strip()

    if not audio_url and episode_url:
        audio_url, description = _resolve_episode_audio(episode_url, description, rss_url=rss_url)

    return {
        "title": title,
        "date": pub_date,
        "url": episode_url or rss_url,
        "audio_url": audio_url,
        "description": description,
    }


def _scrape_latest_episode_http(podcast_url: str) -> dict:
    """Lightweight scrape that avoids browser dependencies when possible."""
    from bs4 import BeautifulSoup

    response = requests.get(
        podcast_url,
        headers={
            "User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
                          "AppleWebKit/537.36 (KHTML, like Gecko) "
                          "Chrome/125.0.0.0 Safari/537.36"
        },
        timeout=25,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    episode_link = soup.select_one("tr a[href*='/episodes/']") or soup.select_one("a[href*='/episodes/']")
    if not episode_link:
        raise RuntimeError("No episode links found in podcast page HTML.")

    href = (episode_link.get("href") or "").strip()
    if not href:
        raise RuntimeError("Episode link found but URL is missing.")

    title = episode_link.get_text(strip=True) or "Unknown Episode"
    description = ""

    row = episode_link.find_parent("tr")
    if row:
        cells = row.find_all("td")
        if len(cells) >= 2:
            description = cells[1].get_text(" ", strip=True)

    if href.startswith("/"):
        ep_href = f"https://www.podchaser.com{href}"
    else:
        ep_href = urllib.parse.urljoin(podcast_url, href)

    rss_url = ""
    rss_link = soup.select_one('link[type="application/rss+xml"]')
    if rss_link and rss_link.get("href"):
        rss_url = urllib.parse.urljoin(podcast_url, rss_link.get("href", "").strip())
    else:
        rss_anchor = soup.select_one('a[href*="rss"]')
        if rss_anchor and rss_anchor.get("href"):
            rss_url = urllib.parse.urljoin(podcast_url, rss_anchor.get("href", "").strip())

    audio_url, description = _resolve_episode_audio(ep_href, description, rss_url=rss_url)

    return {
        "title": title,
        "date": "",
        "url": ep_href,
        "audio_url": audio_url or "",
        "description": description or "",
    }


# ─── Step 1: Scrape latest episode ──────────────────────────────

async def _scrape_latest_episode_async(podcast_url: str):
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
        await page.goto(podcast_url, wait_until="domcontentloaded", timeout=60000)
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

        audio_url, description = _resolve_episode_audio(ep_href, episode.get("description", ""))

        return {
            "title": episode.get("title", "Unknown Episode"),
            "date": episode.get("date", ""),
            "url": ep_href,
            "audio_url": audio_url or "",
            "description": description or "",
        }


def scrape_latest_episode(source_name: str):
    """Fetch latest episode for a configured podcast source."""
    source_config = PODCAST_SOURCES[source_name]

    try:
        return _scrape_latest_episode_rss(source_name, source_config)
    except Exception as exc:
        logger.warning("RSS scrape failed for %s: %s", source_name, exc)

    if not ENABLE_BROWSER_SCRAPE:
        raise RuntimeError(
            "Unable to fetch the latest episode from RSS in this environment. "
            "Paste an audio URL manually, or set ENABLE_PLAYWRIGHT_SCRAPE=1 to allow browser fallback."
        )

    podcast_url = source_config.get("episodes_url", "").strip()
    if not podcast_url:
        raise RuntimeError(f"No browser fallback URL configured for {source_name}.")

    try:
        return _scrape_latest_episode_http(podcast_url)
    except Exception as exc:
        logger.warning("HTTP scrape fallback failed, trying Playwright: %s", exc)

    if not _ensure_playwright_browsers():
        raise RuntimeError(
            "Unable to scrape latest episode in this environment without a browser. "
            "Use manual audio URL input as fallback."
        )

    return asyncio.run(_scrape_latest_episode_async(podcast_url))


# ─── Step 2: Download audio ─────────────────────────────────────

def download_audio(url: str, dest_dir: str) -> str:
    """Download audio file from URL to dest_dir, return file path."""
    headers = {
        "User-Agent": (
            "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
            "AppleWebKit/537.36 (KHTML, like Gecko) "
            "Chrome/125.0.0.0 Safari/537.36"
        ),
        "Accept": "audio/mpeg, audio/*, */*",
        "Accept-Language": "en-US,en;q=0.9",
        "Referer": urllib.parse.urljoin(url, "/"),
    }
    resp = requests.get(url, stream=True, timeout=120, headers=headers)
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
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"podcast_transcript_{ts}.txt"
    if IS_STREAMLIT_CLOUD:
        if _db_save_article(filename, transcript, article_type="transcript"):
            st.caption(f"Transcript saved to database: {filename}")
        else:
            st.warning("Failed to save transcript to database")
    else:
        os.makedirs(SCRAPED_DIR, exist_ok=True)
        save_path = os.path.join(SCRAPED_DIR, filename)
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
def _create_google_client() -> genai.Client:
    if not GOOGLE_API_KEY:
        raise RuntimeError("Google API key is not configured. Set GOOGLE_API_KEY in secrets or env.")
    return genai.Client(api_key=GOOGLE_API_KEY)


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
    prompt = f"""You are a professional LinkedIn content writer specializing in the topic of podcasting.

Analyze the following podcast transcript and create a compelling LinkedIn post.

Episode title: {episode_title}

STRICT REQUIREMENTS:
- Your ENTIRE output must be UNDER 2800 characters (hard limit). Count carefully.
- Do NOT include any preamble, explanation, or notes outside the post itself.
- Output ONLY the LinkedIn post text, nothing else.
- Do NOT use any markdown formatting (no **bold**, no *italics*, no headers, no bullet points with - or *)
- LinkedIn does not render markdown, so use PLAIN TEXT only

FORMATTING STYLE:
- Use emojis + ALL CAPS for section headings, like:
  🚀 THE BIG NEWS
  🔹 KEY TAKEAWAY
  💡 WHAT THIS MEANS
  ⚡ WHY IT MATTERS
- try to use creative and engaging headlines and subheadlines.
- Add a blank line before and after each heading for visual separation
- Use emojis at the start of key points (🔹, ▸, →)
- Keep paragraphs short (2-3 sentences max)

Instructions:
1. Identify the top 3-4 most important topics discussed in the podcast
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
IMPORTANT: Use plain text only, NO markdown (no **bold**, no *italics*, no headers). LinkedIn does not render markdown.

{article}"""
        article = generate_with_claude(shorten_prompt, temperature=0.5)
        if len(article) <= 3000:
            return article
    return article


def _set_current_article(article_text: str, reset_image_prompt: bool = True) -> None:
    """Sync article-related session state after load/generate events."""
    article_value = article_text or ""
    st.session_state.article = article_value
    st.session_state.article_editor = article_value

    if reset_image_prompt:
        if article_value.strip():
            st.session_state.article_image_prompt = _build_article_image_prompt(article_value)
        else:
            st.session_state.article_image_prompt = ""
        st.session_state.pop("article_image", None)


def _extract_key_themes(article_text: str) -> str:
    """Extract key themes from article using Claude for better results."""
    if not article_text or not article_text.strip():
        return ""

    # Try to use Claude for intelligent extraction
    try:
        prompt = f"""Extract 3-5 key themes or topics from this LinkedIn post as a comma-separated list.
Output ONLY the themes, nothing else. Keep each theme to 2-4 words max.

Example output: AI automation, workplace productivity, machine learning trends, tech leadership

Post:
{article_text[:2000]}"""
        themes = generate_with_claude(prompt, temperature=0.3, max_tokens=100)
        # Clean up the response
        themes = themes.strip().strip('"').strip("'")
        if themes and len(themes) < 200:
            return themes
    except Exception as e:
        logger.warning("Failed to extract themes with Claude: %s", e)

    # Fallback: simple extraction from first few sentences
    clean_text = re.sub(r"#\w+", "", article_text).strip()
    clean_text = clean_text.replace("\n", " ")
    clean_text = re.sub(r"\s+", " ", clean_text)
    # Take first 300 chars, try to end at a sentence
    snippet = clean_text[:300]
    last_period = snippet.rfind(".")
    if last_period > 100:
        snippet = snippet[:last_period + 1]
    return snippet


def _build_article_image_prompt(article_text: str = "") -> str:
    return (
        "Create a clean, modern, professional LinkedIn image grounded in the article content. "
        "Use the article itself as the primary source of truth for the scene, mood, and concepts. "
        "Show the core idea visually with polished editorial art direction, abstract tech forms, "
        "smart composition, and a high-end business/technology feel. "
        "Do not add logos, brand names, screenshots, UI mockups, faces. "
        "Use small amount of text overlay."
    )


def _normalize_article_image_prompt(prompt_text: str) -> str:
    prompt = (prompt_text or "").strip()
    if not prompt:
        return ""

    legacy_prefix = "Create a clean, modern, professional LinkedIn image grounded in the article content below."
    if prompt.startswith(legacy_prefix):
        return _build_article_image_prompt()

    prompt = re.sub(r"\s*Article content:\s*.*$", "", prompt, flags=re.DOTALL).strip()
    return prompt


def _compose_article_image_generation_prompt(article_text: str, prompt_text: str) -> str:
    clean_article = re.sub(r"\s+", " ", (article_text or "").strip())
    themes = _extract_key_themes(article_text)
    prompt = _normalize_article_image_prompt(prompt_text) or _build_article_image_prompt()

    if themes:
        prompt = f"{prompt} Key themes from the article: {themes}."

    return f"{prompt} Article content for reference: {clean_article}"


def generate_article_image(article_text: str, prompt_override: str = ""):
    """Generate an image for the LinkedIn article using Google's Nano Banana model."""
    try:
        if not article_text or not article_text.strip():
            return False, None, None, "No article content available to build an image."

        prompt_for_image = (
            _normalize_article_image_prompt(prompt_override) if prompt_override else ""
        ) or _build_article_image_prompt(article_text)
        generation_prompt = _compose_article_image_generation_prompt(article_text, prompt_for_image)
        client = _create_google_client()
        response = client.models.generate_content(
            model=NANO_BANANA_MODEL,
            contents=generation_prompt,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )

        for part in response.parts:
            if part.inline_data is not None:
                payload = {
                    "bytes": part.inline_data.data,
                    "mime_type": part.inline_data.mime_type or "image/png",
                    "alt_text": "AI article illustration",
                }
                return True, payload, prompt_for_image, None

        return False, None, prompt_for_image, "Nano Banana returned no image data"
    except Exception as e:
        return False, None, None, f"Error generating image: {str(e)}"


# ─── Step 5: LinkedIn OAuth & Publish ────────────────────────────

def _get_secret_or_env(secret_key: str, env_key: Optional[str] = None) -> str:
    """Read a config value from Streamlit secrets first, then environment."""
    env_name = env_key or secret_key
    value = st.secrets.get(secret_key) or os.getenv(env_name) or ""
    return str(value).strip()


def _is_local_redirect_host(hostname: str) -> bool:
    host = (hostname or "").strip().lower()
    return host in {"localhost", "127.0.0.1", "0.0.0.0"}


def _get_runtime_url() -> str:
    try:
        return str(getattr(st.context, "url", "") or "").strip()
    except Exception:
        return ""


def _is_local_runtime() -> bool:
    runtime_url = _get_runtime_url()
    parsed_runtime = urllib.parse.urlsplit(runtime_url) if runtime_url else urllib.parse.SplitResult("", "", "", "", "")
    return _is_local_redirect_host(parsed_runtime.hostname or "")


def _strip_redirect_uri_extras(raw_uri: str) -> str:
    """Drop query/fragment while preserving the original scheme/host/path shape."""
    parsed = urllib.parse.urlsplit(str(raw_uri or "").strip())
    if not parsed.scheme or not parsed.netloc:
        return ""
    return urllib.parse.urlunsplit(
        (
            parsed.scheme,
            parsed.netloc,
            parsed.path,
            "",
            "",
        )
    )


def _normalize_redirect_uri(raw_uri: str) -> str:
    """Normalize redirect URIs and prefer the live app URL when config looks unsafe."""
    configured = str(raw_uri or "").strip()
    parsed_configured_raw = urllib.parse.urlsplit(configured) if configured else urllib.parse.SplitResult("", "", "", "", "")

    runtime_url = _get_runtime_url()
    normalized_runtime = _strip_redirect_uri_extras(runtime_url)
    parsed_runtime = urllib.parse.urlsplit(normalized_runtime) if normalized_runtime else urllib.parse.SplitResult("", "", "", "", "")

    if not configured:
        return normalized_runtime

    normalized_configured = _strip_redirect_uri_extras(configured)
    parsed_configured = urllib.parse.urlsplit(normalized_configured) if normalized_configured else urllib.parse.SplitResult("", "", "", "", "")

    if not normalized_runtime:
        return normalized_configured

    configured_has_extra_parts = bool(parsed_configured_raw.query or parsed_configured_raw.fragment)
    configured_mismatch = (
        parsed_configured.scheme != parsed_runtime.scheme
        or parsed_configured.netloc != parsed_runtime.netloc
        or (parsed_configured.path or "/") != (parsed_runtime.path or "/")
    )
    configured_is_local = _is_local_redirect_host(parsed_configured.hostname or "")
    runtime_is_local = _is_local_redirect_host(parsed_runtime.hostname or "")
    prefer_runtime = (
        configured_has_extra_parts
        or (configured_is_local and not runtime_is_local)
        or (not runtime_is_local and configured_mismatch)
    )
    if prefer_runtime:
        logger.warning(
            "Using runtime URL for LinkedIn redirect URI. configured=%s runtime=%s",
            normalized_configured,
            normalized_runtime,
        )
        return normalized_runtime

    return normalized_configured


def get_linkedin_config():
    config = {
        "client_id": _get_secret_or_env("LINKEDIN_CLIENT_ID"),
        "client_secret": _get_secret_or_env("LINKEDIN_CLIENT_SECRET"),
        "redirect_uri": _normalize_redirect_uri(_get_secret_or_env("LINKEDIN_REDIRECT_URI")),
    }
    missing_fields = [key for key, value in config.items() if not value]
    if missing_fields:
        missing_labels = ", ".join(missing_fields)
        st.error(f"Missing LinkedIn configuration: {missing_labels}")
        st.info(
            "Set LINKEDIN_CLIENT_ID, LINKEDIN_CLIENT_SECRET, and "
            "LINKEDIN_REDIRECT_URI in Streamlit secrets or environment variables."
        )
        return None
    return config


def _get_session_id() -> str:
    """Get or create a unique session identifier for OAuth state."""
    if "oauth_session_id" not in st.session_state:
        import uuid
        st.session_state.oauth_session_id = uuid.uuid4().hex[:16]
    return st.session_state.oauth_session_id


def generate_auth_url(config):
    # Use session-specific state to prevent cross-user cache conflicts
    session_id = _get_session_id()
    state_value = f"ai_podcast_app_{session_id}"
    params = {
        "response_type": "code",
        "client_id": config["client_id"],
        "redirect_uri": config["redirect_uri"],
        "scope": "w_member_social openid email profile",
        "state": state_value,
    }
    return f"{LINKEDIN_AUTH_URL}?{urllib.parse.urlencode(params)}"


def render_linkedin_auth_button(config: Dict[str, str]) -> None:
    """Render LinkedIn auth in the most reliable way for the current runtime."""
    if not config:
        return
    auth_url = generate_auth_url(config)

    if not _is_local_runtime():
        st.link_button("🔗 Authorize on LinkedIn", auth_url, type="primary")
        return

    escaped_url = html.escape(auth_url, quote=True)
    st.html(
        f"""
        <a
            href="{escaped_url}"
            target="_self"
            style="
                display: inline-flex;
                align-items: center;
                justify-content: center;
                background: #9ad1ff;
                border: 1px solid #7fbef2;
                color: #0b2e4b;
                border-radius: 0.5rem;
                padding: 0.5rem 0.9rem;
                font-size: 1rem;
                font-weight: 600;
                text-decoration: none;
            "
        >
            🔗 Authorize on LinkedIn
        </a>
        """,
        width="content",
    )


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
    headers = {
        "Content-Type": mime_type or "image/png",
        "Content-Length": str(len(image_bytes)),
        "Connection": "close",
    }
    last_error = None

    for attempt in range(1, 4):
        try:
            # Use a fresh session per attempt so TLS retries do not reuse a bad socket.
            with requests.Session() as session:
                response = session.put(upload_url, data=image_bytes, headers=headers, timeout=(10, 120))
            if response.status_code >= 400:
                return False, f"{response.status_code}: {response.text}"
            return True, None
        except (requests.exceptions.SSLError, requests.exceptions.ConnectionError, requests.exceptions.Timeout) as exc:
            last_error = str(exc)
            logger.warning("LinkedIn image upload attempt %s/3 failed: %s", attempt, exc)
            if attempt < 3:
                time.sleep(attempt)
        except requests.RequestException as exc:
            return False, str(exc)

    return False, last_error or "LinkedIn image upload failed after retries."


def post_to_linkedin(content, access_token, author_id, image_payload=None, allow_image_fallback: bool = True):
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
    upload_warning = None
    if image_payload:
        if not image_payload.get("bytes"):
            return False, "Image payload is empty."
        init_ok, upload_url, image_urn, init_error = initialize_linkedin_image_upload(
            access_token,
            author_urn,
        )
        if not init_ok:
            if allow_image_fallback:
                upload_warning = f"Image upload init failed; published without image. Details: {init_error}"
                logger.warning(upload_warning)
            else:
                return False, f"Image upload init failed: {init_error}"
        else:
            upload_ok, upload_error = upload_linkedin_image(
                upload_url,
                image_payload.get("bytes", b""),
                image_payload.get("mime_type", "image/png"),
            )
            if not upload_ok:
                if allow_image_fallback:
                    upload_warning = f"Image upload failed; published without image. Details: {upload_error}"
                    logger.warning(upload_warning)
                else:
                    return False, f"Image upload failed: {upload_error}"
            else:
                alt_text = image_payload.get("alt_text") or "AI podcast illustration"
                payload["content"] = {"media": {"id": image_urn, "altText": alt_text}}
    try:
        response = requests.post(LINKEDIN_API_URL, json=payload, headers=headers)
        if response.status_code >= 400:
            logger.error("LinkedIn API error: %s %s", response.status_code, response.text)
            return False, f"{response.status_code}: {response.text}"
        response.raise_for_status()
        result = {}
        if response.content:
            try:
                parsed_result = response.json()
                if isinstance(parsed_result, dict):
                    result = parsed_result
            except ValueError:
                logger.warning("LinkedIn post create response was not valid JSON.")
        post_id = (
            response.headers.get("x-restli-id")
            or response.headers.get("X-Restli-Id")
            or result.get("id")
            or result.get("post_id")
        )
        if post_id:
            result["id"] = post_id
            result["debug"] = fetch_linkedin_post_debug(
                access_token=access_token,
                post_urn=post_id,
                sent_commentary=content,
                has_image=bool(image_payload and payload.get("content")),
            )
        if upload_warning:
            result["warning"] = upload_warning
        return True, result
    except requests.RequestException as e:
        return False, str(e)


def fetch_linkedin_post_debug(access_token: str, post_urn: str, sent_commentary: str, has_image: bool) -> dict:
    """Fetch the created post back from LinkedIn to compare stored vs sent commentary."""
    debug_info = {
        "post_urn": post_urn,
        "has_image": has_image,
        "sent_commentary_length": len(sent_commentary or ""),
        "sent_commentary": sent_commentary or "",
    }
    headers = {
        "Authorization": f"Bearer {access_token}",
        "X-Restli-Protocol-Version": "2.0.0",
        "Linkedin-Version": LINKEDIN_API_VERSION,
    }
    encoded_urn = urllib.parse.quote(post_urn, safe="")

    try:
        response = requests.get(
            f"{LINKEDIN_API_URL}/{encoded_urn}",
            headers=headers,
            params={"viewContext": "AUTHOR"},
            timeout=30,
        )
        response.raise_for_status()
        data = response.json() if response.content else {}
        stored_commentary = data.get("commentary") or ""
        debug_info.update(
            {
                "stored_commentary_length": len(stored_commentary),
                "stored_commentary": stored_commentary,
                "commentary_matches_exactly": stored_commentary == (sent_commentary or ""),
                "stored_post_id": data.get("id") or post_urn,
                "stored_lifecycle_state": data.get("lifecycleState") or "",
            }
        )
    except requests.HTTPError as exc:
        response = getattr(exc, "response", None)
        if response is not None and response.status_code == 403:
            debug_info["retrieval_error"] = (
                "LinkedIn denied post retrieval. This app can create member posts with "
                "`w_member_social`, but reading member posts back typically requires "
                "`r_member_social`, which is currently a closed permission."
            )
        else:
            debug_info["retrieval_error"] = str(exc)
    except (requests.RequestException, ValueError) as exc:
        debug_info["retrieval_error"] = str(exc)

    return debug_info


def render_linkedin_post_debug(debug_info: dict, key_prefix: str) -> None:
    """Render LinkedIn post debug details in the Streamlit UI."""
    if not debug_info:
        return

    with st.expander("LinkedIn Post Debug", expanded=False):
        st.caption(f"Post URN: {debug_info.get('post_urn', '')}")
        st.caption(f"Image attached in payload: {debug_info.get('has_image', False)}")
        st.caption(f"Sent commentary length: {debug_info.get('sent_commentary_length', 0)}")

        retrieval_error = debug_info.get("retrieval_error")
        if retrieval_error:
            st.warning(f"LinkedIn post fetch failed: {retrieval_error}")
        else:
            st.caption(f"Stored commentary length: {debug_info.get('stored_commentary_length', 0)}")
            st.caption(f"Exact commentary match: {debug_info.get('commentary_matches_exactly', False)}")
            if debug_info.get("stored_lifecycle_state"):
                st.caption(f"Stored lifecycle state: {debug_info.get('stored_lifecycle_state')}")

        st.text_area(
            "Sent Commentary Snapshot",
            value=debug_info.get("sent_commentary", ""),
            height=220,
            disabled=True,
            key=f"{key_prefix}_sent_commentary_debug",
        )
        st.text_area(
            "Stored Commentary Snapshot",
            value=debug_info.get("stored_commentary", ""),
            height=220,
            disabled=True,
            key=f"{key_prefix}_stored_commentary_debug",
        )


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

    # Validate state parameter - must start with "ai_podcast_app_"
    # Extract session_id from state for cache key lookup
    valid_state = False
    session_id_from_state = ""
    if state_param and state_param.startswith("ai_podcast_app_"):
        valid_state = True
        session_id_from_state = state_param.replace("ai_podcast_app_", "")
    elif state_param == "ai_podcast_app":
        # Legacy support for old state format
        valid_state = True
        session_id_from_state = st.session_state.get("oauth_session_id", "legacy")

    if code_param and valid_state and token_active:
        try:
            st.query_params.clear()
        except AttributeError:
            st.experimental_set_query_params()
        return

    if code_param and valid_state and not token_active:
        with st.spinner("Exchanging authorization code..."):
            token_data = exchange_code_for_token(code_param, config)
            if token_data and "access_token" in token_data:
                st.session_state.linkedin_token = token_data["access_token"]
                st.session_state.token_expires = time.time() + token_data.get("expires_in", 5184000)
                st.session_state.pop("_oauth_ready", None)
                member_urn = fetch_authenticated_member_urn(st.session_state.linkedin_token)
                if member_urn:
                    st.session_state.author_id = member_urn
                # Restore article saved before OAuth redirect (use session-specific cache keys)
                article_cache_key = f"pending_article_{session_id_from_state}"
                source_cache_key = f"pending_source_{session_id_from_state}"
                if IS_STREAMLIT_CLOUD:
                    cached = _db_load_oauth_cache(article_cache_key)
                    if cached and cached.strip():
                        _set_current_article(cached)
                    _db_delete_oauth_cache(article_cache_key)

                    source_json = _db_load_oauth_cache(source_cache_key)
                    if source_json:
                        try:
                            import json
                            source_data = json.loads(source_json) or {}
                            if source_data.get("episode"):
                                st.session_state.episode = source_data["episode"]
                            if source_data.get("direct_audio_url"):
                                st.session_state.direct_audio_url = source_data["direct_audio_url"]
                            if source_data.get("source_mode"):
                                st.session_state.source_mode = source_data["source_mode"]
                        except Exception as exc:
                            logger.warning("Failed to restore source data: %s", exc)
                    _db_delete_oauth_cache(source_cache_key)
                else:
                    if os.path.exists(_SESSION_ARTICLE_CACHE):
                        with open(_SESSION_ARTICLE_CACHE) as f:
                            cached = f.read()
                        if cached.strip():
                            _set_current_article(cached)
                        os.remove(_SESSION_ARTICLE_CACHE)
                    if os.path.exists(_SESSION_SOURCE_CACHE):
                        try:
                            import json
                            with open(_SESSION_SOURCE_CACHE) as f:
                                source_data = json.load(f) or {}
                            if source_data.get("episode"):
                                st.session_state.episode = source_data["episode"]
                            if source_data.get("direct_audio_url"):
                                st.session_state.direct_audio_url = source_data["direct_audio_url"]
                            if source_data.get("source_mode"):
                                st.session_state.source_mode = source_data["source_mode"]
                        except Exception as exc:
                            logger.warning("Failed to restore source data: %s", exc)
                        finally:
                            os.remove(_SESSION_SOURCE_CACHE)
                try:
                    st.query_params.clear()
                except AttributeError:
                    st.experimental_set_query_params()
                st.success("Connected to LinkedIn!")
                return


# ─── Main App ────────────────────────────────────────────────────

def main():
    st.title("🎙️ AI Podcast to LinkedIn Article")
    st.markdown("Scrape **podcast**, transcribe, generate a LinkedIn article, and publish.")

    _handle_linkedin_oauth()

    st.markdown(
        """
        <style>
        div[data-testid="stHorizontalBlock"]:nth-of-type(1),
        div[data-testid="stHorizontalBlock"]:nth-of-type(2) {
            padding: 0.5rem;
            border-radius: 0.75rem;
        }
        div[data-testid="stHorizontalBlock"]:nth-of-type(1) {
            background: #f2f5ff;
        }
        div[data-testid="stHorizontalBlock"]:nth-of-type(2) {
            background: #f7f3ff;
        }
        div[data-testid="stHorizontalBlock"]:nth-of-type(1) > div,
        div[data-testid="stHorizontalBlock"]:nth-of-type(2) > div {
            padding: 0.75rem;
            border-radius: 0.6rem;
        }
        div[data-testid="stHorizontalBlock"]:nth-of-type(1) > div:nth-child(1) {
            background: #e6ecff;
        }
        div[data-testid="stHorizontalBlock"]:nth-of-type(1) > div:nth-child(2) {
            background: #e9f7ff;
        }
        div[data-testid="stHorizontalBlock"]:nth-of-type(2) > div:nth-child(1) {
            background: #fff0f3;
        }
        div[data-testid="stHorizontalBlock"]:nth-of-type(2) > div:nth-child(2) {
            background: #f0fff4;
        }
        button[kind="primary"] {
            background: #9ad1ff;
            border: 1px solid #7fbef2;
            color: #0b2e4b;
        }
        button[kind="primary"]:hover {
            background: #7fbef2;
            border-color: #6aaee6;
            color: #0b2e4b;
        }
        div[data-testid="stAlert"][data-baseweb="notification"] svg {
            color: #7fbef2;
        }
        input[type="checkbox"] {
            accent-color: #7fbef2;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    token_active = (
        "linkedin_token" in st.session_state
        and time.time() < st.session_state.get("token_expires", 0)
    )

    # ── Row 1 ──
    row1_left, row1_right = st.columns(2)
    with row1_left:
        st.subheader("Connect LinkedIn Account")
        if not token_active:
            config = get_linkedin_config()
            if config:
                if st.session_state.get("_oauth_ready"):
                    st.info("Click below to authorize with LinkedIn:")
                    render_linkedin_auth_button(config)
                    if st.button("Cancel", key="cancel_oauth"):
                        st.session_state.pop("_oauth_ready", None)
                        st.rerun()
                else:
                    if st.button("Connect LinkedIn Account", type="primary"):
                        # Save article so it survives the OAuth redirect
                        # Use session-specific cache keys to prevent cross-user conflicts
                        session_id = _get_session_id()
                        article_cache_key = f"pending_article_{session_id}"
                        source_cache_key = f"pending_source_{session_id}"

                        article = st.session_state.get("article", "").strip()
                        if article:
                            if IS_STREAMLIT_CLOUD:
                                _db_save_oauth_cache(article_cache_key, article)
                            else:
                                os.makedirs(SCRAPED_DIR, exist_ok=True)
                                with open(_SESSION_ARTICLE_CACHE, "w") as f:
                                    f.write(article)
                        source_payload = {
                            "episode": st.session_state.get("episode"),
                            "direct_audio_url": st.session_state.get("direct_audio_url", ""),
                            "source_mode": st.session_state.get("source_mode", ""),
                        }
                        try:
                            import json
                            source_json = json.dumps(source_payload)
                            if IS_STREAMLIT_CLOUD:
                                _db_save_oauth_cache(source_cache_key, source_json)
                            else:
                                os.makedirs(SCRAPED_DIR, exist_ok=True)
                                with open(_SESSION_SOURCE_CACHE, "w") as f:
                                    f.write(source_json)
                        except Exception as exc:
                            logger.warning("Failed to cache source data: %s", exc)

                        st.session_state._oauth_ready = True
                        st.rerun()
        else:
            st.success("LinkedIn connected!")
            if st.button("Disconnect LinkedIn"):
                for key in ["linkedin_token", "token_expires", "author_id"]:
                    st.session_state.pop(key, None)
                st.session_state.pop("_oauth_ready", None)
                st.rerun()

    with row1_right:
        st.subheader("Set Source")
        source_controls_enabled = token_active
        if not source_controls_enabled:
            st.info("Connect LinkedIn Account first to enable source setup.")

        # Podcast selector
        podcast_names = list(PODCAST_SOURCES.keys())
        selected_podcast = st.selectbox(
            "Select Podcast",
            podcast_names,
            index=0,
            key="selected_podcast",
            disabled=not source_controls_enabled,
        )
        selected_source = PODCAST_SOURCES[selected_podcast]
        st.session_state.selected_podcast_url = selected_source.get("episodes_url", "")

        if st.button("Fetch Latest Episode", type="primary", disabled=not source_controls_enabled):
            with st.spinner(f"Scraping {selected_podcast}..."):
                try:
                    episode = scrape_latest_episode(selected_podcast)
                    episode["podcast_name"] = selected_podcast
                    st.session_state.episode = episode
                    st.session_state.source_mode = "latest"
                    st.success(f"Found: **{episode['title']}**")
                    logger.info("Episode URL: %s", episode.get("url", ""))
                    logger.info("Audio URL: %s", episode.get("audio_url", ""))
                    st.info(f"Episode URL: {episode.get('url', 'N/A')}")
                    st.info(f"Audio URL: {episode.get('audio_url', 'N/A') or 'Not found'}")
                except Exception as e:
                    st.error(f"Failed to scrape episode: {e}")

        if source_controls_enabled and "episode" in st.session_state:
            ep = st.session_state.episode
            if ep.get("podcast_name"):
                st.markdown(f"**Podcast:** {ep['podcast_name']}")
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
                manual_url = st.text_input(
                    "Paste audio URL (mp3/m4a):",
                    key="manual_audio_url",
                    disabled=not source_controls_enabled,
                )
                if manual_url:
                    st.session_state.episode["audio_url"] = manual_url
                    st.session_state.source_mode = "url"

        col_url, col_btn = st.columns([3, 1])
        with col_url:
            direct_url = st.text_input(
                "Audio URL (mp3/m4a):",
                key="direct_audio_url",
                label_visibility="collapsed",
                placeholder="Paste audio URL here...",
                disabled=not source_controls_enabled,
            )
        with col_btn:
            if st.button("Set URL", disabled=(not source_controls_enabled or not direct_url)):
                st.session_state.episode = {
                    "title": "Manual Audio",
                    "date": "",
                    "url": "",
                    "audio_url": direct_url,
                    "description": "",
                }
                st.session_state.source_mode = "url"
                st.success("Audio URL set!")
                st.rerun()

    # ── Row 2 ──
    row2_left, row2_right = st.columns(2)
    with row2_right:
        st.subheader("Do All (Fetch → Transcribe → Generate → Publish)")
        save_audio_all = st.checkbox("Save audio to disk", value=True, key="doall_save_audio")
        include_image_all = False
        if GOOGLE_API_KEY:
            include_image_all = st.checkbox("Generate and include image", value=True, key="doall_include_image")
        else:
            st.caption("Set GOOGLE_API_KEY to enable image generation.")
        run_do_all = st.button("Do All", type="primary")

        if run_do_all:
            if not token_active:
                st.error("Connect LinkedIn first to publish.")
            else:
                temp_audio_dir = None
                try:
                    episode = st.session_state.get("episode") or {}
                    audio_url = episode.get("audio_url", "")
                    direct_url = st.session_state.get("direct_audio_url", "").strip()
                    source_mode = st.session_state.get("source_mode", "")
                    podcast_name = st.session_state.get("selected_podcast", list(PODCAST_SOURCES.keys())[0])

                    if source_mode == "url" and direct_url:
                        audio_url = direct_url
                        episode = {
                            "title": "Manual Audio",
                            "date": "",
                            "url": "",
                            "audio_url": direct_url,
                            "description": "",
                        }
                        st.session_state.episode = episode

                    if source_mode == "url" and not audio_url:
                        st.error("Audio URL not set. Use Set URL first.")
                        st.stop()

                    if source_mode != "url":
                        with st.spinner(f"Fetching latest episode from {podcast_name}..."):
                            episode = scrape_latest_episode(podcast_name)
                        episode["podcast_name"] = podcast_name
                        st.session_state.episode = episode
                        st.session_state.source_mode = "latest"
                        audio_url = episode.get("audio_url", "")

                    if not audio_url:
                        st.error("No audio URL found for the latest episode. Please paste one manually.")
                        st.stop()

                    if save_audio_all:
                        os.makedirs(SCRAPED_DIR, exist_ok=True)
                        audio_dir = SCRAPED_DIR
                    else:
                        temp_audio_dir = tempfile.mkdtemp()
                        audio_dir = temp_audio_dir

                    with st.spinner("Downloading audio..."):
                        filepath = download_audio(audio_url, audio_dir)

                    if save_audio_all:
                        ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                        ext = os.path.splitext(filepath)[1] or ".mp3"
                        saved_name = f"podcast_audio_{ts}{ext}"
                        saved_path = os.path.join(SCRAPED_DIR, saved_name)
                        if filepath != saved_path:
                            os.rename(filepath, saved_path)
                            filepath = saved_path

                    with st.spinner("Transcribing audio..."):
                        transcript = upload_and_transcribe(filepath)
                    st.session_state.transcript = transcript
                    st.session_state.transcript_editor = transcript

                    ep_title = episode.get("title", "")
                    with st.spinner("Generating LinkedIn article..."):
                        article = generate_linkedin_article(transcript, ep_title)
                    _set_current_article(article)

                    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                    filename = f"podcast_article_{ts}.txt"
                    if IS_STREAMLIT_CLOUD:
                        _db_save_article(filename, article, article_type="article")
                    else:
                        os.makedirs(SCRAPED_DIR, exist_ok=True)
                        save_path = os.path.join(SCRAPED_DIR, filename)
                        with open(save_path, "w") as f:
                            f.write(article)

                    image_payload = None
                    if include_image_all and GOOGLE_API_KEY:
                        with st.spinner("Generating image..."):
                            success, img_payload, prompt_used, error = generate_article_image(
                                article,
                                prompt_override=st.session_state.get("article_image_prompt", ""),
                            )
                        if success and img_payload:
                            image_payload = img_payload
                            st.session_state.article_image = img_payload
                            st.session_state.article_image_prompt = prompt_used
                        else:
                            st.warning(error or "Image generation failed; publishing without image.")

                    if len(article) > 3000:
                        st.error(f"Article is {len(article)} characters. LinkedIn's limit is 3000. Please shorten it.")
                    else:
                        with st.spinner("Publishing to LinkedIn..."):
                            success, result = post_to_linkedin(
                                article,
                                st.session_state.linkedin_token,
                                st.session_state.author_id,
                                image_payload=image_payload,
                            )
                        if success:
                            st.success("Published to LinkedIn!")
                            post_id = ""
                            if isinstance(result, dict):
                                warning = result.get("warning")
                                if warning:
                                    st.warning(warning)
                                debug_info = result.get("debug")
                                if debug_info:
                                    st.session_state.last_linkedin_post_debug = debug_info
                                    render_linkedin_post_debug(debug_info, key_prefix="do_all_publish")
                                post_id = result.get("id") or result.get("post_id") or ""
                            if post_id:
                                st.session_state.last_linkedin_post_urn = post_id
                                st.info(f"Post URN: `{post_id}`")
                                post_url = build_linkedin_post_url(post_id)
                                if post_url:
                                    st.session_state.last_linkedin_post_url = post_url
                                    st.text_input("Post URL (copy)", value=post_url)
                            st.balloons()
                        else:
                            st.error(f"Failed to publish: {result}")
                except Exception as e:
                    st.error(f"Do All failed: {e}")
                finally:
                    if temp_audio_dir:
                        import shutil
                        shutil.rmtree(temp_audio_dir, ignore_errors=True)

    with row2_left:
        # ── Step 2: Transcribe ──
        st.subheader("Transcribe Audio")
        # Check both episode audio_url and direct_audio_url input
        audio_url = st.session_state.get("episode", {}).get("audio_url", "")
        if not audio_url:
            audio_url = st.session_state.get("direct_audio_url", "")
        can_transcribe = bool(audio_url)
    
        # Load a previously saved transcript
        if IS_STREAMLIT_CLOUD:
            saved_transcripts = _db_list_articles(article_type="transcript", limit=20)
        else:
            saved_transcripts = sorted(
                [f for f in os.listdir(SCRAPED_DIR) if f.startswith("podcast_transcript_") and f.endswith(".txt")]
                if os.path.isdir(SCRAPED_DIR) else [],
                reverse=True,
            )
        if saved_transcripts:
            selected_transcript = st.selectbox("Load saved transcript", ["(none)"] + saved_transcripts, key="transcript_selector")
            if selected_transcript != "(none)":
                if st.button("Load Transcript"):
                    if IS_STREAMLIT_CLOUD:
                        content = _db_load_article(selected_transcript)
                        if content:
                            st.session_state.transcript = content
                            st.session_state.transcript_editor = content
                            st.success(f"Loaded {selected_transcript} from database")
                        else:
                            st.error("Failed to load transcript from database")
                    else:
                        with open(os.path.join(SCRAPED_DIR, selected_transcript)) as f:
                            content = f.read()
                        st.session_state.transcript = content
                        st.session_state.transcript_editor = content
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
                        st.session_state.transcript_editor = transcript
                        st.success("Transcription complete!")
    
                        # Clean up temp dir if not saving
                        if not save_audio:
                            import shutil
                            shutil.rmtree(audio_dir, ignore_errors=True)
                    except Exception as e:
                        st.error(f"Transcription failed: {e}")
    
        if "transcript" in st.session_state:
            if "transcript_editor" not in st.session_state:
                st.session_state.transcript_editor = st.session_state.transcript
            with st.expander("View Transcript", expanded=False):
                edited_transcript = st.text_area(
                    "Transcript (Editable)",
                    height=300,
                    key="transcript_editor",
                )
                st.session_state.transcript = edited_transcript
                st.caption(f"{len(edited_transcript):,} characters")
    
        # ── Step 3: Generate Article ──
        st.subheader("Generate LinkedIn Article")
        has_transcript = bool(st.session_state.get("transcript", "").strip())
    
        # Load a previously saved article
        if IS_STREAMLIT_CLOUD:
            saved_articles = _db_list_articles(article_type="article", limit=20)
        else:
            saved_articles = sorted(
                [f for f in os.listdir(SCRAPED_DIR) if f.startswith("podcast_article_") and f.endswith(".txt")]
                if os.path.isdir(SCRAPED_DIR) else [],
                reverse=True,
            )
        if saved_articles:
            selected_file = st.selectbox("Load saved article", ["(none)"] + saved_articles)
            if selected_file != "(none)":
                if st.button("Load"):
                    if IS_STREAMLIT_CLOUD:
                        content = _db_load_article(selected_file)
                        if content:
                            _set_current_article(content)
                            st.success(f"Loaded {selected_file} from database")
                        else:
                            st.error("Failed to load article from database")
                    else:
                        with open(os.path.join(SCRAPED_DIR, selected_file)) as f:
                            content = f.read()
                        _set_current_article(content)
                        st.success(f"Loaded {selected_file}")
                    st.rerun()
    
        if st.button("Generate Article", disabled=not has_transcript):
            try:
                ep_title = st.session_state.get("episode", {}).get("title", "")
                with st.spinner("Generating article with Claude..."):
                    article = generate_linkedin_article(st.session_state.transcript, ep_title)
                _set_current_article(article)
                # Auto-save article
                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"podcast_article_{ts}.txt"
                if IS_STREAMLIT_CLOUD:
                    if _db_save_article(filename, article, article_type="article"):
                        st.success(f"Article generated! Saved to database: {filename}")
                    else:
                        st.success("Article generated!")
                        st.warning("Failed to save to database")
                else:
                    os.makedirs(SCRAPED_DIR, exist_ok=True)
                    save_path = os.path.join(SCRAPED_DIR, filename)
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
        if not GOOGLE_API_KEY:
            st.info("Set GOOGLE_API_KEY to enable image generation.")
        else:
            has_article = bool(st.session_state.get("article", "").strip())
            if "article_image_prompt" not in st.session_state:
                st.session_state.article_image_prompt = ""
            default_prompt = _build_article_image_prompt(
                st.session_state.get("article", ""),
            )
            normalized_prompt = _normalize_article_image_prompt(st.session_state.article_image_prompt)
            if normalized_prompt != st.session_state.article_image_prompt:
                st.session_state.article_image_prompt = normalized_prompt
            prompt_value = st.text_area(
                "Image prompt",
                value=normalized_prompt or default_prompt,
                height=140,
            )
            if st.button("Reset prompt"):
                st.session_state.article_image_prompt = default_prompt
                st.rerun()
            st.session_state.article_image_prompt = prompt_value
            if st.button("Generate Article Image", disabled=not has_article):
                with st.spinner("Generating image from article..."):
                    success, image_payload, prompt_used, error = generate_article_image(
                        st.session_state.article,
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
                    caption="Current image",
                    width="stretch",
                )
                st.download_button(
                    label="Download image",
                    data=st.session_state.article_image["bytes"],
                    file_name="linkedin_article_image.png",
                    mime=st.session_state.article_image.get("mime_type", "image/png"),
                )
                img_col1, img_col2, img_col3 = st.columns(3)
                with img_col1:
                    if st.button("🔄 Generate Again"):
                        with st.spinner("Regenerating image..."):
                            success, image_payload, prompt_used, error = generate_article_image(
                                st.session_state.article,
                                prompt_override=st.session_state.get("article_image_prompt", ""),
                            )
                        if success and image_payload:
                            st.session_state.article_image = image_payload
                            st.session_state.article_image_prompt = prompt_used
                            st.success("Image regenerated!")
                            st.rerun()
                        else:
                            st.error(error or "Failed to regenerate image.")
                with img_col2:
                    uploaded_file = st.file_uploader(
                        "Upload image",
                        type=["png", "jpg", "jpeg", "webp"],
                        key="replace_image_uploader",
                        label_visibility="collapsed",
                    )
                    if uploaded_file is not None:
                        image_bytes = uploaded_file.read()
                        mime_type = uploaded_file.type or "image/png"
                        st.session_state.article_image = {
                            "bytes": image_bytes,
                            "mime_type": mime_type,
                            "alt_text": "Custom uploaded image",
                        }
                        st.success("Image uploaded!")
                        st.rerun()
                with img_col3:
                    if st.button("🗑️ Clear"):
                        st.session_state.pop("article_image", None)
                        st.session_state.pop("article_image_prompt", None)
                        st.rerun()
    
        # ── Step 4: Publish to LinkedIn ──
        st.subheader("Publish to LinkedIn")
    
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
                            warning = result.get("warning")
                            if warning:
                                st.warning(warning)
                            debug_info = result.get("debug")
                            if debug_info:
                                st.session_state.last_linkedin_post_debug = debug_info
                                render_linkedin_post_debug(debug_info, key_prefix="manual_publish")
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
