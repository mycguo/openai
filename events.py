import logging
import os
import re
import json
import asyncio
import hashlib
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from html import escape as html_escape
from typing import Dict, List, Optional, Tuple
from urllib.parse import parse_qsl, urlencode, urljoin, urlparse, urlunparse
from zoneinfo import ZoneInfo

import streamlit as st
import streamlit.components.v1 as st_components
from openai import OpenAI
from anthropic import Anthropic
from google import genai
from google.genai import types
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_browser_use_file_patch_applied = False
_luma_detail_rate_limited_until = 0.0
_luma_detail_rate_limit_lock = threading.Lock()
_luma_detail_rate_limit_log_until = 0.0


def render_copy_button(
    content: str,
    key_suffix: str,
    label: str = "📋 Copy",
    html_content: Optional[str] = None,
) -> None:
    if not content or not content.strip():
        st.caption("Copy unavailable (no text)")
        return

    safe_id = re.sub(r"[^0-9a-zA-Z_-]", "-", key_suffix)
    button_id = f"copy-btn-{safe_id}"
    escaped = json.dumps(content).replace("</", "<\\/")
    escaped_html = json.dumps(html_content or "").replace("</", "<\\/")
    html = f"""
        <style>
        #{button_id} {{
            width: 100%;
            padding: 0.6rem 1rem;
            border-radius: 0.5rem;
            border: 1px solid #2b6cb0;
            background: linear-gradient(90deg, #3182ce, #2c5282);
            color: white;
            font-weight: 600;
            cursor: pointer;
        }}
        #{button_id}:hover {{
            background: linear-gradient(90deg, #2c5282, #2a4365);
        }}
        </style>
        <button id="{button_id}">{label}</button>
        <script>
        (function() {{
            const btn = document.getElementById("{button_id}");
            if (!btn) {{
                return;
            }}
            const text = {escaped};
            const htmlContent = {escaped_html};
            const defaultLabel = "{label}";
            const successLabel = "✅ Copied!";
            const failureLabel = "⚠️ Copy failed";

            const fallbackCopy = () => {{
                const textarea = document.createElement('textarea');
                textarea.value = text;
                textarea.style.position = 'fixed';
                textarea.style.opacity = '0';
                textarea.style.left = '-1000px';
                textarea.style.top = '0';
                document.body.appendChild(textarea);
                textarea.focus();
                textarea.select();
                try {{
                    if (!document.execCommand('copy')) {{
                        btn.textContent = failureLabel;
                    }} else {{
                        btn.textContent = successLabel;
                    }}
                }} catch (err) {{
                    console.error('execCommand copy failed', err);
                    btn.textContent = failureLabel;
                }} finally {{
                    document.body.removeChild(textarea);
                    setTimeout(() => (btn.textContent = defaultLabel), 2000);
                }}
            }};

            const copyRichHtml = async () => {{
                if (!htmlContent || !navigator.clipboard || !navigator.clipboard.write || !window.ClipboardItem) {{
                    return false;
                }}
                try {{
                    const item = new ClipboardItem({{
                        "text/plain": new Blob([text], {{ type: "text/plain" }}),
                        "text/html": new Blob([htmlContent], {{ type: "text/html" }})
                    }});
                    await navigator.clipboard.write([item]);
                    return true;
                }} catch (err) {{
                    console.warn('Rich HTML clipboard copy failed', err);
                    return false;
                }}
            }};

            const handleClick = async () => {{
                if (await copyRichHtml()) {{
                    btn.textContent = successLabel;
                    setTimeout(() => (btn.textContent = defaultLabel), 2000);
                    return;
                }}
                if (navigator.clipboard && navigator.clipboard.writeText) {{
                    navigator.clipboard
                        .writeText(text)
                        .then(() => {{
                            btn.textContent = successLabel;
                            setTimeout(() => (btn.textContent = defaultLabel), 2000);
                        }})
                        .catch((err) => {{
                            console.warn('Clipboard API copy failed', err);
                            fallbackCopy();
                        }});
                }} else {{
                    fallbackCopy();
                }}
            }};

            btn.addEventListener('click', handleClick);
        }})();
        </script>
    """
    st_components.html(html, height=70)


def _disable_browser_use_file_saving() -> None:
    """Monkey patch browser_use FileSystem to avoid disk writes for extracted content."""
    global _browser_use_file_patch_applied
    if _browser_use_file_patch_applied:
        return

    try:
        from browser_use.filesystem import file_system as browser_file_system
    except Exception:  # noqa: BLE001
        return

    original_method = browser_file_system.FileSystem.save_extracted_content

    async def _log_only_save(self, content: str) -> str:  # type: ignore[override]
        entry_number = self.extracted_content_count
        logger.info(
            "Captured extracted content #%s (%s chars). Logging only, skipping disk write.",
            entry_number,
            len(content),
        )
        logger.debug("Extracted content #%s:\n%s", entry_number, content)
        self.extracted_content_count += 1
        return (
            f"Extracted content logged as entry #{entry_number}. "
            "Disk writes are disabled."
        )

    browser_file_system.FileSystem.save_extracted_content = _log_only_save  # type: ignore[assignment]
    browser_file_system._original_save_extracted_content = original_method  # type: ignore[attr-defined]
    _browser_use_file_patch_applied = True


# ─── Configuration ────────────────────────────────────────────────
OPENAI_API_KEY = (
    st.secrets.get("OPENAI_API_KEY")
    or os.getenv("OPENAI_API_KEY")
)
CONFIGURED_GPT_MODEL = (
    st.secrets.get("GPT_MODEL")
    or os.getenv("GPT_MODEL")
)
ANTHROPIC_API_KEY = (
    st.secrets.get("ANTHROPIC_API_KEY")
    or os.getenv("ANTHROPIC_API_KEY")
)
ANTHROPIC_SONNET_MODEL = (
    st.secrets.get("ANTHROPIC_SONNET_MODEL")
    or os.getenv("ANTHROPIC_SONNET_MODEL")
    or "claude-sonnet-4-5"
)
GOOGLE_API_KEY = (
    st.secrets.get("GOOGLE_API_KEY")
    or os.getenv("GOOGLE_API_KEY")
)
PREFERRED_GPT_MODEL = "gpt-5.5"
DEFAULT_GPT_MODELS: List[str] = [
    PREFERRED_GPT_MODEL,
    CONFIGURED_GPT_MODEL,
    "gpt-5",
    "gpt-5-turbo",
    "gpt-4o",
    "gpt-4-turbo",
]
NANO_BANANA_2_MODEL = "gemini-3.1-flash-image"
LUMA_BASE_URL = "https://lu.ma"
LUMA_REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) '
                  'AppleWebKit/537.36 (KHTML, like Gecko) '
                  'Chrome/120.0.0.0 Safari/537.36',
    'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
    'Accept-Language': 'en-US,en;q=0.9',
}
LUMA_REGION_SOURCES = {
    "Bay Area": {
        "source_name": "Lu.ma Bay Area",
        "urls": [
            f"{LUMA_BASE_URL}/genai-sf?k=c",
            f"{LUMA_BASE_URL}/sf",
        ],
        "caption": "Sources: genai-sf + sf",
    },
    "New York": {
        "source_name": "Lu.ma New York",
        "urls": [
            f"{LUMA_BASE_URL}/nyc",
        ],
        "caption": "Source: nyc",
    },
}
LUMA_DISCOVERY_PATHS = {"/sf", "/genai-sf", "/nyc"}
FOCUS_REGION_ORDER = ["Bay Area", "New York"]
CV_REGION_PAGE_URLS = {
    "Bay Area": "https://cerebralvalley.ai/events?locations=BAY_AREA",
    "New York": "https://cerebralvalley.ai/events?locations=NYC",
}
MEETUP_REGION_SOURCES = {
    "Bay Area": {
        "source_name": "Meetup Bay Area",
        "url": "https://www.meetup.com/find/?keywords=artificial%20intelligence&source=EVENTS&location=us--ca--San%20Francisco",
        "caption": "Source: Meetup AI events near San Francisco",
        "timezone": "America/Los_Angeles",
    },
    "New York": {
        "source_name": "Meetup New York",
        "url": "https://www.meetup.com/find/?keywords=artificial%20intelligence&source=EVENTS&location=us--ny--New%20York",
        "caption": "Source: Meetup AI events near New York",
        "timezone": "America/New_York",
    },
}
EVION_BASE_URL = "https://evion.app"
EVION_EVENTS_PAGE_URL = f"{EVION_BASE_URL}/events"
EVION_SUPABASE_URL = "https://adzesenszyhiexqnkfgh.supabase.co"
EVION_SUPABASE_ANON_KEY = (
    "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9."
    "eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImFkemVzZW5zenloaWV4cW5rZmdoIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NTkxNzI2NDEsImV4cCI6MjA3NDc0ODY0MX0."
    "Eg8mLCeFZrRSaSQrbcdMV0qWzzUtlq59UpXOxlm_iO8"
)
EVION_REQUEST_HEADERS = {
    "User-Agent": LUMA_REQUEST_HEADERS["User-Agent"],
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "apikey": EVION_SUPABASE_ANON_KEY,
    "Authorization": f"Bearer {EVION_SUPABASE_ANON_KEY}",
}
EVENT_SNAPSHOT_FORMAT = "ai_events_snapshot_v1"
CV_EVENTS_API_URL = "https://api.cerebralvalley.ai/v1/public/event/pull"
CV_REQUEST_HEADERS = {
    "User-Agent": LUMA_REQUEST_HEADERS["User-Agent"],
    "Accept": "application/json, text/plain, */*",
    "Accept-Language": "en-US,en;q=0.9",
    "Origin": "https://cerebralvalley.ai",
    "Referer": "https://cerebralvalley.ai/",
}
MEETUP_REQUEST_HEADERS = {
    "User-Agent": LUMA_REQUEST_HEADERS["User-Agent"],
    "Accept": LUMA_REQUEST_HEADERS["Accept"],
    "Accept-Language": LUMA_REQUEST_HEADERS["Accept-Language"],
}

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY
if ANTHROPIC_API_KEY:
    os.environ["ANTHROPIC_API_KEY"] = ANTHROPIC_API_KEY


def _gpt_model_candidates() -> List[str]:
    candidates: List[str] = []
    seen = set()
    for model_name in DEFAULT_GPT_MODELS:
        if not model_name:
            continue
        if model_name in seen:
            continue
        seen.add(model_name)
        candidates.append(model_name)
    return candidates


def _is_model_unavailable_error(error: Exception) -> bool:
    message = str(error).lower()
    # Check for various error conditions that indicate the model is not available or incompatible
    return ("not found" in message or
            "does not exist" in message or
            "not supported" in message or
            "unsupported_parameter" in message)


@st.cache_resource
def get_resolved_gpt_model() -> str:
    client = _create_openai_client()
    last_error: Optional[Exception] = None
    for candidate in _gpt_model_candidates():
        try:
            # Try to use the model with a simple test
            # GPT-5 and newer models use max_completion_tokens instead of max_tokens
            if candidate.startswith('gpt-5') or candidate.startswith('o1') or candidate.startswith('o3'):
                client.chat.completions.create(
                    model=candidate,
                    messages=[{"role": "user", "content": "test"}],
                    max_completion_tokens=1
                )
            else:
                client.chat.completions.create(
                    model=candidate,
                    messages=[{"role": "user", "content": "test"}],
                    max_tokens=1
                )
            logger.info("Using GPT model: %s", candidate)
            return candidate
        except Exception as exc:  # noqa: BLE001
            if _is_model_unavailable_error(exc):
                logger.warning("GPT model unavailable: %s (%s)", candidate, exc)
                last_error = exc
                continue
            # If it's a different error, the model exists but request failed for other reasons
            # Still return this model as it's available
            logger.info("Using GPT model: %s (validated)", candidate)
            return candidate

    raise RuntimeError(
        "No supported GPT model available. Set GPT_MODEL to a supported ID or check your OpenAI API access."
    ) from last_error


@st.cache_resource
def _create_openai_client() -> OpenAI:
    if not OPENAI_API_KEY:
        raise RuntimeError(
            "OpenAI API key is not configured. Set OPENAI_API_KEY in secrets or env."
        )
    return OpenAI(api_key=OPENAI_API_KEY)


@st.cache_resource
def _create_anthropic_client() -> Anthropic:
    if not ANTHROPIC_API_KEY:
        raise RuntimeError(
            "Anthropic API key is not configured. Set ANTHROPIC_API_KEY in secrets or env."
        )
    return Anthropic(api_key=ANTHROPIC_API_KEY)


def generate_with_gpt(prompt: str, temperature: float = 0.0, max_tokens: int = 2060, model_override: Optional[str] = None) -> str:
    """Call OpenAI GPT API and return the text output."""
    client = _create_openai_client()
    model_name = model_override or get_resolved_gpt_model()

    # GPT-5 and newer models use max_completion_tokens instead of max_tokens
    if (model_name.startswith('gpt-5') or model_name.startswith('o1') or
            model_name.startswith('o3') or model_name.startswith('gpt-4.5')):
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_completion_tokens=max_tokens,
        )
    else:
        response = client.chat.completions.create(
            model=model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=temperature,
            max_tokens=max_tokens,
        )

    if response.choices and len(response.choices) > 0:
        return response.choices[0].message.content.strip()

    raise RuntimeError("OpenAI response did not contain text output")


def generate_with_claude(prompt: str, temperature: float = 1.0, max_tokens: int = 3000) -> str:
    """Call Anthropic Claude Sonnet and return the text output."""
    client = _create_anthropic_client()
    response = client.messages.create(
        model=ANTHROPIC_SONNET_MODEL,
        max_tokens=max_tokens,
        extra_body={"temperature": temperature},
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": prompt,
                    }
                ],
            }
        ],
    )
    text_segments: List[str] = []
    for block in response.content:
        if getattr(block, "type", "") == "text":
            text_segments.append(block.text)
        elif isinstance(block, dict) and block.get("type") == "text":
            text_segments.append(block.get("text", ""))
    full_text = "".join(text_segments).strip()
    if full_text:
        return full_text
    raise RuntimeError("Anthropic response did not contain text output")


# ─── Main features ─────────────────────────────────────────────────
async def scrape_events(url="https://lu.ma/genai-sf?k=c", source_name="Lu.ma GenAI SF", days=8):
    """Use browser-use to scrape events from lu.ma/genai-sf"""
    from browser_use.llm.openai.chat import ChatOpenAI
    from browser_use.agent.service import Agent
    from browser_use.browser import BrowserProfile, BrowserSession
    import os

    _disable_browser_use_file_saving()

    if not OPENAI_API_KEY:
        raise RuntimeError("OpenAI API key not configured for scraping")

    # Ensure API key is in environment for downstream libraries
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # Configure Playwright to use system chromium on Streamlit Cloud
    os.environ["PLAYWRIGHT_BROWSERS_PATH"] = "0"
    os.environ["PLAYWRIGHT_SKIP_BROWSER_DOWNLOAD"] = "1"

    # Try to find system chromium executable (for Streamlit Cloud)
    chromium_paths = [
        "/usr/bin/chromium",
        "/usr/bin/chromium-browser",
        "/usr/bin/google-chrome",
        "/usr/bin/google-chrome-stable"
    ]

    chromium_executable = None
    for path in chromium_paths:
        if os.path.exists(path):
            chromium_executable = path
            os.environ["CHROME_PATH"] = path
            os.environ["PLAYWRIGHT_CHROMIUM_EXECUTABLE_PATH"] = path
            print(f"✅ Found system Chromium at: {path}")
            break

    if not chromium_executable:
        print("⚠️ System Chromium not found, using Playwright's bundled browser")
        print(f"Checked paths: {chromium_paths}")

    # Set up browser session
    browser_session = BrowserSession(
        browser_profile=BrowserProfile(
            keep_alive=False,
            headless=True,
            record_video_dir=None,
        )
    )

    # Set up the LLM using browser-use's ChatOpenAI with GPT
    resolved_model = get_resolved_gpt_model()
    llm = ChatOpenAI(
        model=resolved_model,
        api_key=OPENAI_API_KEY,
        temperature=0.2,
    )

    # System prompt for formatting
    system_prompt = """You are extracting event information. Format events as:

Event Name: [Name]
Date and Time: [Date and Time]
Location/Venue: [Venue/Address]
Brief Description: [Brief description including organizer/host]
Event URL: [ACTUAL URL]

CRITICAL URL EXTRACTION RULES:
- Each event card is clickable - you MUST find and extract the href attribute from the <a> tag that wraps the event
- ONLY treat <a> tags whose href contains "/event/" as event entries. Ignore navigation, sign-in, footer, or map attribution links.
- Use JavaScript evaluation or DOM queries to get the href: document.querySelector('a[href*="/event/"]').href
- ALWAYS call the run_javascript action to execute the provided snippet and capture the exact href/title pairs. Do NOT pass the snippet to extract_structured_data.
- After running run_javascript, inspect the returned JSON and log each href you plan to use. If you do not see href values, re-run run_javascript with a corrected selector before continuing.
- On Luma.com: URLs are in format https://lu.ma/event-slug-abc123
- The href might be relative (e.g., /event-slug) - prepend https://lu.ma if needed
- NEVER write "Link", "Not provided", or "URL extraction failed"
- You have access to browser automation - use it to inspect elements and get exact URLs
- SPECIAL CASE (https://cerebralvalley.ai/events): Event cards use `<div class="flex flex-col pb-[2rem]">` containers. Inside each, there is an `<a ... aria-label="Open event: ..." href="https://lu.ma/...">` wrapping the `<h3>` title. Select them via `document.querySelectorAll('div.flex.flex-col.pb-[2rem] a[aria-label^="Open event"]')`, and capture each anchor's href and inner text. These href values already include tracking parameters; copy them exactly.
"""

    # Task to scrape events
    task = f"""Go to {url} and extract AI/GenAI event information for the next {days} days.

STEP-BY-STEP PROCESS:

1. Load the page and wait for events to appear (wait 3 seconds after page load)

2. For EACH event card visible on the page, use JavaScript or DOM inspection to extract:

   CRITICAL - URL EXTRACTION METHOD:
   - Each event is wrapped in an <a> tag (anchor/link element)
   - Use JavaScript to get the href:
     * Find the event card container element
     * Get the parent <a> tag or find <a> tag within the card
     * Extract .href property (this gives full URL) or .getAttribute('href') (might be relative)
     * If relative (starts with /), prepend https://lu.ma

   Example JavaScript you can execute:
   ```javascript
   // Find only event detail links
   Array.from(document.querySelectorAll('a[href*="/event/"]')).map(a => ({{
     href: a.href.startsWith('http') ? a.href : `https://lu.ma${{a.getAttribute('href')}}`,
     title: (a.querySelector('[data-testid="event-card__title"]')?.textContent || a.textContent).trim()
   }}))
   ```
   Run the above EXACTLY via the run_javascript tool, then use the returned href values for every event you output.

   When scraping Cerebral Valley specifically, use:
   ```javascript
   Array.from(document.querySelectorAll('div.flex.flex-col.pb-[2rem] a[aria-label^="Open event"]')).map(a => ({{
     href: a.href,
     title: (a.querySelector('h3 span.inline')?.textContent || a.textContent).trim(),
     host: a.closest('div.flex.flex-col.pb-[2rem]')?.querySelector('div.flex.items-center.leading-[24px] p')?.textContent?.trim() || ''
   }}))
   ```
   This returns the exact href tied to the event title; attach that URL directly to the title when presenting results.

   Extract for each event:
   - Event Name (from the title/heading in the card)
   - Date and Time (visible on the card)
   - Location/Venue (location text on the card)
   - Brief Description (organizer, status like "Sold Out" or "Waitlist")
   - Event URL (use JavaScript/DOM inspection as described above)

3. After extracting visible events, scroll down ONCE to load more events

4. Extract URLs for the newly visible events using the same method

5. Scroll down ONE more time (maximum 2 scrolls total)

6. Extract URLs for any new events

7. STOP and return all collected events

CRITICAL REQUIREMENTS:
- You MUST use browser automation/JavaScript to get actual href attributes
- NEVER write "URL extraction failed" - use JavaScript evaluation to get URLs
- The URLs should all start with https://lu.ma/
- Before returning, double-check every event you output includes an "Event URL: https://..." line populated with the actual link you captured. If any event is missing a URL, inspect the DOM again until you have it.
- Maximum 2 scrolls, then STOP
- Focus on getting correct URLs - this is the most important part

"""

    try:
        # Start browser session
        await browser_session.start()

        # Create the agent with max_actions limit
        agent = Agent(
            task=task,
            llm=llm,
            browser_session=browser_session,
            system_message=system_prompt,
            max_actions=15  # Allow enough actions for: page load, wait, extract, scroll, extract, scroll, extract, return
        )

        # Run the agent
        result = await agent.run()

        # Clean up browser session
        await browser_session.kill()

        return result

    except Exception as e:
        st.error(f"Error scraping events: {str(e)}")
        import traceback
        traceback.print_exc()
        try:
            await browser_session.kill()
        except:
            pass
        return None


def scrape_luma_events(url="https://lu.ma/genai-sf?k=c", days=8):
    """Directly scrape lu.ma events without browser automation.

    Note: Luma.com shows events in chronological order. Since we can't reliably
    extract dates from static HTML without the full calendar rendering, we limit
    by taking only the first N events which are typically within the next few days.
    """
    normalized_url = _normalize_luma_url(url)
    logger.info("Fetching Luma events directly from %s (next %d days)", normalized_url, days)

    try:
        # Use headers to mimic a real browser
        response = requests.get(normalized_url, headers=LUMA_REQUEST_HEADERS, timeout=30)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch Luma events: %s", exc)
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    events = []

    # Find all links with event-like paths
    # Event links on lu.ma are simple paths like "/ra7ba3kr", "/ai-x-healthcare"
    all_links = soup.find_all('a', href=re.compile(r'^/[^/]+$'))

    for link in all_links:
        href = link.get('href', '').strip()
        if not href:
            continue

        parsed_href = urlparse(href if href.startswith('http') else urljoin(LUMA_BASE_URL, href))
        href_path = parsed_href.path.rstrip('/') or '/'

        # Filter out non-event links (navigation, etc.)
        if href_path in {'/', '/discover', '/signin'}:
            continue
        if href_path in LUMA_DISCOVERY_PATHS:
            continue

        # Build full URL
        if href.startswith('http'):
            full_url = _normalize_luma_url(href)
        else:
            full_url = urljoin(LUMA_BASE_URL, href)

        # Extract title - try aria-label first, then look for h3 in parent button
        title_text = link.get('aria-label', '').strip()

        if not title_text:
            # Try to find h3 within the same button container
            button = link.find_parent('button')
            if button:
                h3 = button.find('h3')
                if h3:
                    title_text = h3.get_text(" ", strip=True)

        # If still no title, use link text
        if not title_text:
            title_text = link.get_text(" ", strip=True)

        # Skip if we couldn't get a meaningful title
        if not title_text or len(title_text) < 3:
            continue

        logger.info("Luma event parsed: %s -> %s", title_text, full_url)

        events.append({
            'title': title_text,
            'url': full_url,
            'host': '',  # Luma doesn't expose host info easily in HTML
        })

    # Remove duplicates (same URL)
    seen_urls = set()
    unique_events = []
    for event in events:
        if event['url'] not in seen_urls:
            seen_urls.add(event['url'])
            unique_events.append(event)

    # Limit to approximately the requested time range
    # Allow up to ~20 events per day to avoid missing dense event days
    max_events = max(20, days * 20)
    limited_events = unique_events[:max_events]

    logger.info("Successfully extracted %d unique Luma events (limited from %d to ~%d days)",
                len(limited_events), len(unique_events), days)
    return limited_events


def _is_luma_url(url: str) -> bool:
    if not url:
        return False
    normalized = url.lower()
    return "luma.com" in normalized or "lu.ma" in normalized


def _normalize_luma_url(url: str) -> str:
    if not url:
        return url
    trimmed = url.strip()
    if not trimmed:
        return trimmed

    lowered = trimmed.lower()
    legacy_prefixes = [
        "https://luma.com",
        "http://luma.com",
        "https://www.luma.com",
        "http://www.luma.com",
    ]
    for prefix in legacy_prefixes:
        if lowered.startswith(prefix):
            return f"{LUMA_BASE_URL}{trimmed[len(prefix):]}"

    if lowered.startswith("https://lu.ma"):
        if not trimmed.startswith(LUMA_BASE_URL):
            return f"{LUMA_BASE_URL}{trimmed[len('https://lu.ma'):]}"
        return trimmed

    if lowered.startswith("http://lu.ma"):
        suffix = trimmed[len('http://lu.ma'):]
        return f"{LUMA_BASE_URL}{suffix}"

    if trimmed.startswith('/'):
        return f"{LUMA_BASE_URL}{trimmed}"

    return trimmed


def _dedupe_events_by_url(events: List[Dict]) -> List[Dict]:
    seen_urls = set()
    unique_events: List[Dict] = []
    for event in events:
        event_url = _ensure_text(event.get('url', '')).strip()
        dedupe_key = _canonicalize_event_url(event_url) or event_url
        if not dedupe_key or dedupe_key in seen_urls:
            continue
        seen_urls.add(dedupe_key)
        unique_events.append(event)
    return unique_events


def _get_luma_source_urls(url: str) -> Optional[List[str]]:
    if url == "LUMA_COMBINED":
        combined_urls: List[str] = []
        for config in LUMA_REGION_SOURCES.values():
            combined_urls.extend(config["urls"])
        return combined_urls
    if url == "LUMA_BAY_AREA":
        return list(LUMA_REGION_SOURCES["Bay Area"]["urls"])
    if url == "LUMA_NEW_YORK":
        return list(LUMA_REGION_SOURCES["New York"]["urls"])
    return None


def _collect_luma_events(urls: List[str], days: int) -> List[Dict]:
    events_list: List[Dict] = []
    for source_url in urls:
        scraped_events = scrape_luma_events(source_url, days)
        if not scraped_events:
            logger.info("No events found for Luma source %s", source_url)
            continue
        events_list.extend(scraped_events)
        logger.info("Added %d events from %s", len(scraped_events), source_url)

    unique_events = _dedupe_events_by_url(events_list)
    logger.info("Combined total: %d unique Luma events", len(unique_events))
    return unique_events


def _parse_iso_datetime(value: Optional[str]) -> Optional[datetime]:
    if not value:
        return None
    value = value.strip()
    if not value:
        return None
    normalized = value.replace('Z', '+00:00')
    try:
        return datetime.fromisoformat(normalized)
    except ValueError:
        return None


def _format_datetime_parts(dt_value: Optional[datetime]) -> Dict[str, str]:
    if not dt_value:
        return {"date_text": "", "time_text": ""}

    date_text = dt_value.strftime('%B %d, %Y')
    time_text = dt_value.strftime('%I:%M %p').lstrip('0')

    tz_name = dt_value.strftime('%Z')
    if tz_name:
        time_text = f"{time_text} {tz_name}".strip()
    else:
        offset = dt_value.strftime('%z')
        if offset:
            offset_formatted = f"GMT{offset[:3]}:{offset[3:]}"
            time_text = f"{time_text} {offset_formatted}".strip()

    return {
        "date_text": date_text,
        "time_text": time_text,
    }


def _cerebral_valley_start_datetime_utc() -> str:
    return (
        datetime.now(timezone.utc)
        .isoformat(timespec="milliseconds")
        .replace("+00:00", "Z")
    )


def _infer_cerebral_valley_timezone(location: str, venue: str = "") -> timezone | ZoneInfo:
    combined = f"{location} {venue}".lower()
    region_name = _classify_event_region(combined)
    if region_name in {"New York", "Boston / Cambridge"}:
        return ZoneInfo("America/New_York")
    if region_name in {"Bay Area", "Pacific Northwest"}:
        return ZoneInfo("America/Los_Angeles")
    if "london" in combined:
        return ZoneInfo("Europe/London")
    if "paris" in combined:
        return ZoneInfo("Europe/Paris")
    return timezone.utc


def _parse_cerebral_valley_api_datetime(
    value: Optional[str],
    location: str = "",
    venue: str = "",
) -> Optional[datetime]:
    if not value:
        return None

    normalized = value.strip().replace(" ", "T")
    if not normalized:
        return None

    try:
        parsed = datetime.fromisoformat(normalized)
    except ValueError:
        return None

    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)

    return parsed.astimezone(_infer_cerebral_valley_timezone(location, venue))


def _format_cerebral_valley_location(location: str, venue: str) -> str:
    location_text = _ensure_text(location).strip()
    venue_text = _ensure_text(venue).strip()
    if venue_text and location_text and venue_text.lower() != location_text.lower():
        return f"{venue_text}, {location_text}"
    return venue_text or location_text


def _merge_location_parts(primary: str, secondary: str) -> str:
    primary_text = _ensure_text(primary).strip()
    secondary_text = _ensure_text(secondary).strip()
    if primary_text and secondary_text and primary_text.lower() != secondary_text.lower():
        return f"{primary_text}, {secondary_text}"
    return primary_text or secondary_text


def _map_cerebral_valley_api_event(raw_event: Dict) -> Dict:
    title = _ensure_text(raw_event.get("name"), "Untitled Event").strip() or "Untitled Event"
    url = _ensure_text(raw_event.get("url")).strip()
    location = _ensure_text(raw_event.get("location")).strip()
    venue = _ensure_text(raw_event.get("venue")).strip()
    start_dt = _parse_cerebral_valley_api_datetime(raw_event.get("startDateTime"), location, venue)
    datetime_parts = _format_datetime_parts(start_dt)
    description = _ensure_text(
        raw_event.get("descriptionSummary") or raw_event.get("description")
    ).strip()

    mapped = {
        "title": title,
        "url": url,
        "host": "Cerebral Valley" if raw_event.get("CVEvent") else "",
        "location": _format_cerebral_valley_location(location, venue),
        "description": description,
        "date_text": datetime_parts["date_text"],
        "time_text": datetime_parts["time_text"],
        "start_iso": start_dt.isoformat() if start_dt else "",
    }

    if raw_event.get("id"):
        mapped["id"] = _ensure_text(raw_event["id"]).strip()

    return mapped


def _ensure_text(value, fallback: str = "") -> str:
    if isinstance(value, str):
        return value
    if value is None:
        return fallback
    if isinstance(value, (int, float)):
        return str(value)
    if isinstance(value, dict):
        extracted = (
            _extract_name(value)
            or _format_location_value(value)
        )
        if isinstance(extracted, str) and extracted:
            return extracted
        if extracted:
            return _ensure_text(extracted)
        return json.dumps(value, ensure_ascii=False)
    if isinstance(value, list):
        combined = ', '.join(
            part.strip()
            for part in (_ensure_text(item) for item in value)
            if part.strip()
        )
        return combined if combined else fallback
    return str(value)


def _format_location_value(location_value) -> str:
    if not location_value:
        return ""
    if isinstance(location_value, list):
        for item in location_value:
            formatted = _format_location_value(item)
            if formatted:
                return formatted
        return ""
    if isinstance(location_value, dict):
        name = _ensure_text(location_value.get('name') or location_value.get('legalName') or '')
        address = location_value.get('address')
        address_text = ""
        if isinstance(address, dict):
            components = [
                address.get('streetAddress'),
                address.get('addressLocality'),
                address.get('addressRegion'),
                address.get('postalCode'),
                address.get('addressCountry'),
            ]
            address_text = ', '.join(
                _ensure_text(part).strip() for part in components if _ensure_text(part).strip()
            )
        elif isinstance(address, str):
            address_text = address.strip()
        else:
            address_text = _ensure_text(address).strip()

        parts = [part for part in [name.strip(), address_text.strip()] if part]
        return ', '.join(parts)
    if isinstance(location_value, str):
        return location_value.strip()
    return ""


def _extract_name(value) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        for item in value:
            name = _extract_name(item)
            if name:
                return name
        return ""
    if isinstance(value, dict):
        return value.get('name') or value.get('legalName') or ''
    if isinstance(value, str):
        return value.strip()
    return ""


def _clean_description(text: Optional[str]) -> str:
    if not text:
        return ""
    cleaned = BeautifulSoup(text, 'html.parser').get_text(' ', strip=True)
    return re.sub(r'\s+', ' ', cleaned).strip()


def _collect_jsonld_events(payload) -> List[Dict]:
    events: List[Dict] = []

    def _walk(node):
        if isinstance(node, list):
            for item in node:
                _walk(item)
            return
        if not isinstance(node, dict):
            return

        node_type = node.get('@type')
        if isinstance(node_type, list):
            if any('Event' in str(entry) for entry in node_type):
                events.append(node)
        elif isinstance(node_type, str) and 'Event' in node_type:
            events.append(node)

        for key in ('@graph', 'graph', 'itemListElement', 'item', 'mainEntity'):
            if key in node:
                _walk(node[key])

    _walk(payload)
    return events


def _extract_details_from_jsonld(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, 'html.parser')
    scripts = soup.find_all('script', attrs={'type': 'application/ld+json'})

    for script in scripts:
        content = script.string or script.get_text()
        if not content:
            continue
        try:
            data = json.loads(content)
        except json.JSONDecodeError:
            continue

        for event_node in _collect_jsonld_events(data):
            start_value = event_node.get('startDate') or event_node.get('startTime')
            start_dt = _parse_iso_datetime(start_value)
            datetime_parts = _format_datetime_parts(start_dt)
            location_text = _format_location_value(event_node.get('location'))
            host_text = (
                _extract_name(event_node.get('organizer'))
                or _extract_name(event_node.get('performer'))
                or _extract_name(event_node.get('creator'))
            )
            description_text = _clean_description(event_node.get('description'))

            details = {
                'date_text': datetime_parts['date_text'],
                'time_text': datetime_parts['time_text'],
                'location': location_text,
                'host': host_text,
                'description': description_text,
            }

            if start_dt:
                details['start_iso'] = start_dt.isoformat()
            if not any(details.values()):
                continue
            return details

    return {}


def _clean_meetup_location_text(value: str) -> str:
    cleaned = _clean_location_for_display(value)
    cleaned = re.sub(
        r"\b(San Francisco|New York City|New York)\s+\1\b",
        r"\1",
        cleaned,
        flags=re.IGNORECASE,
    )
    return _clean_location_for_display(cleaned)


def _format_meetup_location(location_value) -> str:
    if isinstance(location_value, list):
        for item in location_value:
            formatted = _format_meetup_location(item)
            if formatted:
                return formatted
        return ""

    if not isinstance(location_value, dict):
        return _clean_meetup_location_text(_format_location_value(location_value))

    location_type = _ensure_text(location_value.get("@type")).lower()
    if "virtuallocation" in location_type:
        return "Online"

    name = _ensure_text(location_value.get("name")).strip()
    address = location_value.get("address")
    address_text = ""
    if isinstance(address, dict):
        street = _ensure_text(address.get("streetAddress")).strip()
        locality = _ensure_text(address.get("addressLocality")).strip()
        region = _ensure_text(address.get("addressRegion")).strip()

        address_parts: List[str] = []
        if street:
            address_parts.append(street)
            street_lower = street.lower()
            if locality and locality.lower() not in street_lower:
                address_parts.append(locality)
            if region and region.lower() not in street_lower:
                address_parts.append(region)
        else:
            address_parts.extend(part for part in [locality, region] if part)
        address_text = ", ".join(address_parts)
    else:
        address_text = _ensure_text(address).strip()

    if name and address_text:
        name_lower = name.lower()
        address_lower = address_text.lower()
        if name_lower == address_lower or name_lower in address_lower:
            return _clean_meetup_location_text(address_text)
        if address_lower in name_lower:
            return _clean_meetup_location_text(name)
        return _clean_meetup_location_text(f"{name}, {address_text}")

    return _clean_meetup_location_text(name or address_text)


def _clean_meetup_description(text: Optional[str]) -> str:
    cleaned = _clean_description(text)
    cleaned = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", cleaned)
    cleaned = re.sub(r"[*_`#>]+", "", cleaned)
    cleaned = re.sub(r"\\([,\\-])", r"\1", cleaned)
    return re.sub(r"\s+", " ", cleaned).strip()


def _map_meetup_jsonld_event(raw_event: Dict, region_name: str) -> Optional[Dict]:
    title = _ensure_text(raw_event.get("name"), "Untitled Event").strip() or "Untitled Event"
    url = _canonicalize_event_url(_ensure_text(raw_event.get("url")).strip())
    if not url:
        return None

    config = MEETUP_REGION_SOURCES.get(region_name, {})
    try:
        region_tz = ZoneInfo(_ensure_text(config.get("timezone")).strip())
    except Exception:  # noqa: BLE001
        region_tz = datetime.now().astimezone().tzinfo or timezone.utc

    start_dt = _parse_iso_datetime(_ensure_text(raw_event.get("startDate")).strip())
    if start_dt and start_dt.tzinfo is None:
        start_dt = start_dt.replace(tzinfo=region_tz)
    elif start_dt:
        start_dt = start_dt.astimezone(region_tz)

    datetime_parts = _format_datetime_parts(start_dt)
    location = _format_meetup_location(raw_event.get("location"))
    if not location and "online" in _ensure_text(raw_event.get("eventAttendanceMode")).lower():
        location = "Online"

    return {
        "id": f"meetup:{url}",
        "title": title,
        "url": url,
        "host": _extract_name(raw_event.get("organizer")) or "Meetup",
        "location": location,
        "description": _clean_meetup_description(raw_event.get("description")),
        "date_text": datetime_parts["date_text"],
        "time_text": datetime_parts["time_text"],
        "start_iso": start_dt.isoformat() if start_dt else "",
    }


def scrape_meetup_events(url: str, region_name: str) -> List[Dict]:
    logger.info("Fetching Meetup events from %s", url)
    response = requests.get(
        url,
        headers=MEETUP_REQUEST_HEADERS,
        timeout=30,
    )
    response.raise_for_status()

    soup = BeautifulSoup(response.text, "html.parser")
    collected_events: List[Dict] = []
    seen_keys = set()
    seen_fingerprints = set()

    for script in soup.find_all("script", attrs={"type": "application/ld+json"}):
        content = script.string or script.get_text()
        if not content:
            continue
        try:
            payload = json.loads(content)
        except json.JSONDecodeError:
            continue

        for raw_event in _collect_jsonld_events(payload):
            mapped = _map_meetup_jsonld_event(raw_event, region_name)
            if not mapped:
                continue
            if not mapped.get("date_text") and not mapped.get("start_iso"):
                continue
            dedupe_key = _canonicalize_event_url(_ensure_text(mapped.get("url")).strip()) or (
                f"{mapped.get('title')}|{mapped.get('start_iso')}"
            )
            fingerprint = "|".join(
                [
                    _ensure_text(mapped.get("title")).strip().lower(),
                    _ensure_text(mapped.get("start_iso")).strip(),
                    _ensure_text(mapped.get("location")).strip().lower(),
                ]
            )
            if dedupe_key in seen_keys or fingerprint in seen_fingerprints:
                continue
            seen_keys.add(dedupe_key)
            seen_fingerprints.add(fingerprint)
            collected_events.append(mapped)

    sorted_events = _sort_events_by_start(collected_events)
    logger.info("Extracted %d Meetup events from %s", len(sorted_events), url)
    return sorted_events


def _extract_details_fallback(html: str) -> Dict[str, str]:
    soup = BeautifulSoup(html, 'html.parser')
    details: Dict[str, str] = {}

    time_tag = soup.find('time')
    if time_tag:
        dt_attr = time_tag.get('datetime')
        parsed = _parse_iso_datetime(dt_attr)
        datetime_parts = _format_datetime_parts(parsed)
        details.update(datetime_parts)
        if parsed:
            details['start_iso'] = parsed.isoformat()

    if 'location' not in details or not details.get('location'):
        location_candidate = soup.find(attrs={'data-testid': re.compile('location', re.I)})
        if location_candidate:
            details['location'] = location_candidate.get_text(' ', strip=True)

    if 'description' not in details or not details.get('description'):
        meta_desc = soup.find('meta', attrs={'property': 'og:description'})
        if meta_desc and meta_desc.get('content'):
            details['description'] = meta_desc['content'].strip()

    return {k: v for k, v in details.items() if v}


@lru_cache(maxsize=256)
def _fetch_luma_event_details(url: str) -> Dict[str, str]:
    global _luma_detail_rate_limited_until
    global _luma_detail_rate_limit_log_until

    parsed_url = urlparse(_normalize_luma_url(url))
    fetch_url = urlunparse((
        parsed_url.scheme.lower(),
        parsed_url.netloc.lower(),
        parsed_url.path.rstrip("/") or "/",
        "",
        "",
        "",
    ))

    now = time.monotonic()
    with _luma_detail_rate_limit_lock:
        cooldown_remaining = _luma_detail_rate_limited_until - now
        should_log_cooldown = cooldown_remaining > 0 and now >= _luma_detail_rate_limit_log_until
        if should_log_cooldown:
            _luma_detail_rate_limit_log_until = now + 5.0

    if cooldown_remaining > 0:
        if should_log_cooldown:
            logger.info(
                "Skipping remaining Luma detail enrichment requests for %.1fs due to rate limiting",
                cooldown_remaining,
            )
        return {}

    max_attempts = 3
    for attempt in range(1, max_attempts + 1):
        try:
            response = requests.get(fetch_url, headers=LUMA_REQUEST_HEADERS, timeout=20)
            if response.status_code == 429:
                retry_after_header = response.headers.get("Retry-After", "").strip()
                wait_seconds = 10.0
                if retry_after_header:
                    try:
                        wait_seconds = max(1.0, min(float(retry_after_header), 30.0))
                    except ValueError:
                        pass
                with _luma_detail_rate_limit_lock:
                    _luma_detail_rate_limited_until = max(
                        _luma_detail_rate_limited_until,
                        time.monotonic() + wait_seconds,
                    )
                    _luma_detail_rate_limit_log_until = time.monotonic() + 5.0
                logger.warning(
                    "Rate limited fetching Luma event details; backing off for %.1fs after %s",
                    wait_seconds,
                    fetch_url,
                )
                return {}

            response.raise_for_status()
            html = response.text
            details = _extract_details_from_jsonld(html)
            if details:
                return details
            return _extract_details_fallback(html)
        except Exception as exc:  # noqa: BLE001
            status_code = getattr(getattr(exc, "response", None), "status_code", None)
            if status_code in {500, 502, 503, 504} and attempt < max_attempts:
                wait_seconds = 0.5 * attempt
                logger.info(
                    "Transient error fetching %s (status %s); retrying in %.1fs (attempt %s/%s)",
                    fetch_url,
                    status_code,
                    wait_seconds,
                    attempt,
                    max_attempts,
                )
                time.sleep(wait_seconds)
                continue
            logger.warning("Failed to fetch event details from %s: %s", fetch_url, exc)
            return {}

    return {}


def _enrich_events_with_details(events: List[Dict]) -> List[Dict]:
    detail_urls: List[str] = []
    normalized_urls: List[str] = []
    for event in events:
        url = event.get('url', '')
        normalized_url = _normalize_luma_url(url) if _is_luma_url(url) else url
        normalized_urls.append(normalized_url)
        if _is_luma_url(url) and normalized_url:
            detail_urls.append(normalized_url)

    details_by_url: Dict[str, Dict[str, str]] = {}
    unique_urls = sorted(set(detail_urls))
    if unique_urls:
        max_workers = min(2, len(unique_urls))
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_url = {
                executor.submit(_fetch_luma_event_details, url): url
                for url in unique_urls
            }
            for future in as_completed(future_to_url):
                url = future_to_url[future]
                try:
                    details_by_url[url] = future.result()
                except Exception as exc:  # noqa: BLE001
                    logger.warning("Failed to enrich event details for %s: %s", url, exc)
                    details_by_url[url] = {}

    enriched: List[Dict] = []
    for event, normalized_url in zip(events, normalized_urls):
        url = event.get('url', '')
        extra = details_by_url.get(normalized_url, {}) if _is_luma_url(url) else {}
        merged = event.copy()
        if normalized_url != url:
            merged['url'] = normalized_url
        for key, value in extra.items():
            if value:
                merged[key] = value
        enriched.append(merged)
    return enriched


def _fetch_cerebral_valley_api_payload(params: Dict[str, object]) -> Dict:
    response = requests.get(
        CV_EVENTS_API_URL,
        params=params,
        headers=CV_REQUEST_HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, dict):
        raise ValueError("Unexpected Cerebral Valley API response shape")
    return payload


def _scrape_cerebral_valley_via_api(days: int = 8) -> List[Dict]:
    start_datetime = _cerebral_valley_start_datetime_utc()
    logger.info(
        "Fetching Cerebral Valley events from public API starting at %s",
        start_datetime,
    )

    collected_events: List[Dict] = []
    seen_keys = set()

    def _add_events(raw_events: List[Dict]) -> None:
        for raw_event in raw_events:
            mapped = _map_cerebral_valley_api_event(raw_event)
            dedupe_key = (
                mapped.get("id")
                or mapped.get("url")
                or f"{mapped.get('title')}|{mapped.get('start_iso')}"
            )
            if dedupe_key in seen_keys:
                continue
            seen_keys.add(dedupe_key)
            collected_events.append(mapped)

    featured_payload = _fetch_cerebral_valley_api_payload(
        {
            "featured": "true",
            "approved": "true",
            "startDateTime": start_datetime,
        }
    )
    _add_events(featured_payload.get("events") or [])

    limit = 100
    offset = 0
    total_count: Optional[int] = None
    pages_fetched = 0
    max_pages = 10

    while pages_fetched < max_pages:
        approved_payload = _fetch_cerebral_valley_api_payload(
            {
                "approved": "true",
                "startDateTime": start_datetime,
                "limit": limit,
                "offset": offset,
            }
        )
        raw_events = approved_payload.get("events") or []
        _add_events(raw_events)
        pages_fetched += 1

        if total_count is None:
            total_count = approved_payload.get("totalCount")

        if not raw_events:
            break

        offset += limit
        if isinstance(total_count, int) and offset >= total_count:
            break

    logger.info(
        "Extracted %d Cerebral Valley events from public API",
        len(collected_events),
    )
    return collected_events


def _fetch_evion_json(path: str, params: Dict[str, object]) -> List[Dict]:
    response = requests.get(
        f"{EVION_SUPABASE_URL}{path}",
        params=params,
        headers=EVION_REQUEST_HEADERS,
        timeout=30,
    )
    response.raise_for_status()
    payload = response.json()
    if not isinstance(payload, list):
        raise ValueError(f"Unexpected Evion response shape for {path}")
    return payload


@lru_cache(maxsize=1)
def _fetch_evion_hidden_external_urls() -> List[str]:
    rows = _fetch_evion_json(
        "/rest/v1/hidden_external_events",
        {"select": "url"},
    )
    hidden_urls = []
    for row in rows:
        normalized = _canonicalize_event_url(_ensure_text(row.get("url")).strip())
        if normalized:
            hidden_urls.append(normalized)
    return hidden_urls


def _fetch_evion_public_events() -> List[Dict]:
    return _fetch_evion_json(
        "/rest/v1/events_public",
        {
            "select": ",".join(
                [
                    "id",
                    "title",
                    "description",
                    "event_date",
                    "event_start_date",
                    "event_end_date",
                    "event_time",
                    "event_start_time",
                    "event_end_time",
                    "event_timezone",
                    "timezone",
                    "location",
                    "venue",
                    "address",
                    "slug",
                    "category",
                    "tags",
                    "is_public",
                    "is_free",
                    "is_online",
                    "is_all_day",
                    "is_featured",
                    "price",
                    "admin_hidden",
                    "deleted_at",
                    "meeting_url",
                ]
            ),
            "is_public": "eq.true",
            "deleted_at": "is.null",
            "order": "event_date.desc",
        },
    )


def _fetch_evion_discovered_events() -> List[Dict]:
    start_floor = datetime.now().strftime("%Y-%m-%dT00:00:00")
    return _fetch_evion_json(
        "/rest/v1/discovered_events",
        {
            "select": ",".join(
                [
                    "id",
                    "url",
                    "title",
                    "description",
                    "category",
                    "industry",
                    "location",
                    "venue_name",
                    "organizer",
                    "source",
                    "source_badge",
                    "region",
                    "start_at",
                    "payload",
                ]
            ),
            "or": f"(start_at.is.null,start_at.gte.{start_floor})",
            "order": "start_at.asc",
            "limit": 3000,
        },
    )


def _is_evion_ai_relevant(raw_event: Dict) -> bool:
    normalized_categories = {
        _ensure_text(raw_event.get("category")).strip().lower(),
        _ensure_text(raw_event.get("industry")).strip().lower(),
    }
    normalized_categories.discard("")
    if normalized_categories.intersection({"ai", "genai", "ml", "machine learning", "artificial intelligence"}):
        return True

    payload = raw_event.get("payload") if isinstance(raw_event.get("payload"), dict) else {}
    tag_text = _ensure_text(raw_event.get("tags"))
    searchable_parts = [
        raw_event.get("title"),
        raw_event.get("description"),
        raw_event.get("category"),
        raw_event.get("industry"),
        raw_event.get("location"),
        raw_event.get("venue"),
        raw_event.get("venue_name"),
        raw_event.get("organizer"),
        raw_event.get("source_badge"),
        tag_text,
        payload.get("title"),
        payload.get("description"),
        payload.get("category"),
        payload.get("industry"),
        payload.get("labels"),
    ]
    searchable = " ".join(
        _ensure_text(part).strip().lower()
        for part in searchable_parts
        if _ensure_text(part).strip()
    )
    return bool(
        re.search(
            r"\b(ai|a\.i\.|llm|ml|genai|rag|gpt|agentic|artificial intelligence|machine learning|deep learning|computer vision|nlp|voice ai|multimodal|inference|embedding|openai|anthropic|claude)\b",
            searchable,
        )
    )


def _parse_evion_public_datetime(raw_event: Dict) -> Optional[datetime]:
    date_value = _ensure_text(raw_event.get("event_start_date") or raw_event.get("event_date")).strip()
    if not date_value:
        return None

    time_value = _ensure_text(raw_event.get("event_start_time") or raw_event.get("event_time")).strip() or "00:00:00"
    try:
        parsed = datetime.fromisoformat(f"{date_value}T{time_value}")
    except ValueError:
        try:
            parsed = datetime.fromisoformat(date_value)
        except ValueError:
            return None

    tz_name = _ensure_text(raw_event.get("event_timezone")).strip()
    location = _ensure_text(raw_event.get("location")).strip()
    venue = _ensure_text(raw_event.get("venue")).strip()
    if tz_name:
        try:
            return parsed.replace(tzinfo=ZoneInfo(tz_name))
        except Exception:  # noqa: BLE001
            pass

    short_tz = _ensure_text(raw_event.get("timezone")).strip().upper()
    short_tz_map = {
        "PT": "America/Los_Angeles",
        "PST": "America/Los_Angeles",
        "PDT": "America/Los_Angeles",
        "ET": "America/New_York",
        "EST": "America/New_York",
        "EDT": "America/New_York",
    }
    if short_tz in short_tz_map:
        return parsed.replace(tzinfo=ZoneInfo(short_tz_map[short_tz]))

    return parsed.replace(tzinfo=_infer_cerebral_valley_timezone(location, venue))


def _parse_evion_discovered_datetime(raw_event: Dict) -> Optional[datetime]:
    payload = raw_event.get("payload") if isinstance(raw_event.get("payload"), dict) else {}
    raw_start = _ensure_text(raw_event.get("start_at") or payload.get("start")).strip()
    parsed = _parse_iso_datetime(raw_start)
    if not parsed:
        return None

    location = _ensure_text(raw_event.get("location") or payload.get("location")).strip()
    venue = _ensure_text(raw_event.get("venue_name") or payload.get("venueName")).strip()
    timezone_hint = _infer_cerebral_valley_timezone(location, venue)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone_hint)
    return parsed.astimezone(timezone_hint)


def _map_evion_public_event(raw_event: Dict) -> Optional[Dict]:
    slug = _ensure_text(raw_event.get("slug")).strip()
    if not slug:
        return None

    title = _ensure_text(raw_event.get("title"), "Untitled Event").strip() or "Untitled Event"
    start_dt = _parse_evion_public_datetime(raw_event)
    datetime_parts = _format_datetime_parts(start_dt)
    if raw_event.get("is_all_day") or not _ensure_text(raw_event.get("event_start_time") or raw_event.get("event_time")).strip():
        datetime_parts["time_text"] = ""

    location = _merge_location_parts(
        _ensure_text(raw_event.get("venue")).strip(),
        _ensure_text(raw_event.get("location")).strip(),
    )
    description = _clean_description(_ensure_text(raw_event.get("description")))
    url = _ensure_text(raw_event.get("meeting_url")).strip() or f"{EVION_BASE_URL}/events/{slug}"

    return {
        "id": f"evion-public:{_ensure_text(raw_event.get('id') or slug).strip()}",
        "title": title,
        "url": url,
        "host": "Evion",
        "location": location,
        "description": description,
        "date_text": datetime_parts["date_text"],
        "time_text": datetime_parts["time_text"],
        "start_iso": start_dt.isoformat() if start_dt else "",
    }


def _map_evion_discovered_event(raw_event: Dict) -> Optional[Dict]:
    payload = raw_event.get("payload") if isinstance(raw_event.get("payload"), dict) else {}
    url = _ensure_text(raw_event.get("url") or payload.get("url")).strip()
    if not url:
        return None
    parsed_url = urlparse(url)
    if parsed_url.netloc.lower().endswith("google.com") and parsed_url.path == "/search":
        return None

    title = (
        _ensure_text(raw_event.get("title")).strip()
        or _ensure_text(payload.get("title")).strip()
        or "Untitled Event"
    )
    location = _merge_location_parts(
        _ensure_text(raw_event.get("venue_name") or payload.get("venueName")).strip(),
        _ensure_text(raw_event.get("location") or payload.get("location")).strip(),
    )
    start_dt = _parse_evion_discovered_datetime(raw_event)
    datetime_parts = _format_datetime_parts(start_dt)
    description = _clean_description(
        _ensure_text(raw_event.get("description") or payload.get("description"))
    )
    host = (
        _ensure_text(raw_event.get("organizer") or payload.get("organizer")).strip()
        or _ensure_text(raw_event.get("source_badge") or payload.get("sourceBadge")).strip()
        or "Evion"
    )

    return {
        "id": f"evion-discovered:{_ensure_text(raw_event.get('id') or url).strip()}",
        "title": title,
        "url": url,
        "host": host,
        "location": location,
        "description": description,
        "date_text": datetime_parts["date_text"],
        "time_text": datetime_parts["time_text"],
        "start_iso": start_dt.isoformat() if start_dt else "",
    }


def _event_datetime_for_sort(event: Dict) -> Optional[datetime]:
    start_iso = _ensure_text(event.get("start_iso")).strip()
    if start_iso:
        parsed = _parse_iso_datetime(start_iso)
        if parsed:
            local_tz = datetime.now().astimezone().tzinfo
            if parsed.tzinfo and local_tz:
                return parsed.astimezone(local_tz).replace(tzinfo=None)
            return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed

    date_text = _ensure_text(event.get("date_text")).strip()
    time_text = _ensure_text(event.get("time_text")).strip()
    if date_text and time_text:
        parsed = _parse_event_datetime_text(f"{date_text} {time_text}")
        if parsed:
            return parsed.replace(tzinfo=None) if parsed.tzinfo else parsed
    if date_text:
        for fmt in ("%B %d, %Y", "%b %d, %Y"):
            try:
                return datetime.strptime(date_text, fmt)
            except ValueError:
                continue
    return None


def _sort_events_by_start(events: List[Dict]) -> List[Dict]:
    def _sort_key(event: Dict) -> tuple:
        parsed = _event_datetime_for_sort(event)
        return (
            parsed is None,
            parsed or datetime.max,
            _ensure_text(event.get("title")).strip().lower(),
        )

    return sorted(events, key=_sort_key)


def generate_evion_region_events(days: int = 8):
    logger.info("Fetching Evion events from %s via Supabase API", EVION_EVENTS_PAGE_URL)
    hidden_urls = set(_fetch_evion_hidden_external_urls())
    public_events = _fetch_evion_public_events()
    discovered_events = _fetch_evion_discovered_events()

    collected_events: List[Dict] = []
    seen_keys = set()

    def _append_event(mapped_event: Optional[Dict]) -> None:
        if not mapped_event:
            return
        if not mapped_event.get("url"):
            return
        if not mapped_event.get("date_text") and not mapped_event.get("start_iso"):
            return
        canonical_url = _canonicalize_event_url(_ensure_text(mapped_event.get("url")).strip())
        dedupe_key = canonical_url or _ensure_text(mapped_event.get("id")).strip() or (
            f"{mapped_event.get('title')}|{mapped_event.get('start_iso')}"
        )
        if dedupe_key in seen_keys:
            return
        seen_keys.add(dedupe_key)
        mapped_event["url"] = canonical_url or _ensure_text(mapped_event.get("url")).strip()
        collected_events.append(mapped_event)

    for raw_event in public_events:
        if raw_event.get("admin_hidden"):
            continue
        if not _is_evion_ai_relevant(raw_event):
            continue
        _append_event(_map_evion_public_event(raw_event))

    for raw_event in discovered_events:
        if not _is_evion_ai_relevant(raw_event):
            continue
        url = _canonicalize_event_url(_ensure_text(raw_event.get("url")).strip())
        if not url or url in hidden_urls:
            continue
        _append_event(_map_evion_discovered_event(raw_event))

    detailed_events = _sort_events_by_start(_filter_events_for_date_range(collected_events, days))
    split_events = _split_focus_region_events(detailed_events)

    region_results: Dict[str, Dict[str, object]] = {}
    for region_name in FOCUS_REGION_ORDER:
        region_events = split_events.get(region_name, [])
        region_results[region_name] = {
            "formatted": _format_event_collection(
                region_events,
                f"Evion {region_name}",
                days,
                f"No {region_name} events found in Evion.",
            ),
            "has_events": bool(region_events),
            "caption": EVION_EVENTS_PAGE_URL,
        }

    combined_formatted = "\n".join(
        [
            _format_event_collection(
                detailed_events,
                "Evion",
                days,
                "No events found on evion.app/events",
            ),
            "",
            "=" * 50,
            f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
    )
    return bool(detailed_events), combined_formatted, region_results


def generate_meetup_region_events(days: int = 8):
    region_results: Dict[str, Dict[str, object]] = {}
    combined_events: List[Dict] = []

    for region_name in FOCUS_REGION_ORDER:
        config = MEETUP_REGION_SOURCES[region_name]
        raw_events = scrape_meetup_events(config["url"], region_name)
        detailed_events = _sort_events_by_start(_filter_events_for_date_range(raw_events, days))
        combined_events.extend(detailed_events)
        region_results[region_name] = {
            "formatted": _format_event_collection(
                detailed_events,
                config["source_name"],
                days,
                f"No {region_name} events found on Meetup.",
            ),
            "has_events": bool(detailed_events),
            "caption": config["caption"],
        }

    combined_formatted = "\n".join(
        [
            _format_event_collection(
                _sort_events_by_start(combined_events),
                "Meetup",
                days,
                "No events found on Meetup",
            ),
            "",
            "=" * 50,
            f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
    )
    return bool(combined_events), combined_formatted, region_results


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


async def _scrape_cerebral_valley_async(days=8):
    """Scrape cerebralvalley.ai/events using Playwright (JS-rendered page)."""
    from playwright.async_api import async_playwright

    target_url = "https://cerebralvalley.ai/events"
    logger.info("Fetching Cerebral Valley events from %s using Playwright", target_url)

    async with async_playwright() as p:
        # Try system chromium first, then fall back to Playwright-managed browser
        launch_kwargs = {"headless": True}
        for path in ["/usr/bin/chromium", "/usr/bin/chromium-browser",
                     "/usr/bin/google-chrome", "/usr/bin/google-chrome-stable"]:
            if os.path.exists(path):
                launch_kwargs["executable_path"] = path
                break

        browser = await p.chromium.launch(**launch_kwargs)
        page = await browser.new_page()
        await page.goto(target_url, wait_until="networkidle")
        await page.wait_for_timeout(3000)

        events_data = await page.evaluate("""
            () => {
                const links = Array.from(document.querySelectorAll('a[aria-label^="Open event"]'));
                return links.map(a => {
                    const title = a.getAttribute('aria-label')?.replace(/^Open event:\\s*/i, '').trim() || '';
                    const href = a.getAttribute('href') || '';
                    let host = '';
                    const parent = a.closest('div');
                    if (parent) {
                        const paragraphs = parent.querySelectorAll('p');
                        const locations = [];
                        paragraphs.forEach(p => {
                            const text = p.textContent?.trim() || '';
                            if (text && !/AM|PM|·/.test(text)) locations.push(text);
                        });
                        host = locations.slice(0, 2).join(' ');
                    }
                    return { title, url: href, host };
                });
            }
        """)

        await browser.close()
        logger.info("Extracted %d Cerebral Valley events", len(events_data))
        return events_data


def scrape_cerebral_valley_events(days=8):
    """Scrape Cerebral Valley events via public API, with Playwright as fallback."""
    try:
        api_events = _scrape_cerebral_valley_via_api(days)
        if api_events:
            return api_events
        logger.warning("Cerebral Valley API returned no events; falling back to Playwright")
    except Exception as exc:
        logger.warning("Cerebral Valley API scrape failed: %s", exc)

    try:
        _ensure_playwright_browsers()
        return asyncio.run(_scrape_cerebral_valley_async(days))
    except Exception as exc:
        logger.error("Failed to fetch Cerebral Valley events: %s", exc)
        import traceback
        traceback.print_exc()
        return []


def _extract_event_links(events_block: str):
    """Return (event_name, url) tuples from the combined events block."""
    if not events_block:
        return []

    # Try "Event URL:" first (new format), then fall back to "Sign-up URL:" (legacy)
    pattern = re.compile(r"\*\*(?P<name>[^*]+)\*\*.*?(?:Event URL|Sign-up URL):\s*(?P<url>https?://\S+)", re.DOTALL)
    matches = []
    for match in pattern.finditer(events_block):
        name = match.group("name").strip()
        url = match.group("url").strip().rstrip('.,)')
        if name and url:
            matches.append((name, url))

    # Fallback: look for less-structured "Event URL" or "Sign-up URL" lines and capture the preceding line as the name
    if not matches:
        lines = events_block.splitlines()
        for idx, line in enumerate(lines):
            if "Event URL:" in line or "Sign-up URL:" in line:
                url_match = re.search(r"(https?://\S+)", line)
                if not url_match:
                    continue
                url = url_match.group(1).rstrip('.,)')
                if url.startswith("http") and idx > 0:
                    name_line = lines[idx - 1].strip()
                    # Remove numbering like "1." and markdown bullets
                    name = re.sub(r"^[\s\d\.-]*", "", name_line)
                    name = name.replace("**", "").strip()
                    if name and url:
                        matches.append((name, url))

    # Deduplicate while preserving order
    seen = set()
    unique_matches = []
    for name, url in matches:
        key = (name, url)
        if key not in seen:
            seen.add(key)
            unique_matches.append((name, url))

    return unique_matches


def _build_fallback_essay(event_links: List[tuple], combined_text: str) -> Optional[str]:
    """Create a deterministic essay if the LLM returns nothing."""
    if event_links:
        intro = (
            "San Francisco's AI scene remains vibrant, with community meetups, demo days, and "
            "founder salons filling calendars nearly every night. Below are a few upcoming "
            "highlights drawn directly from the scraped events."
        )
        body_lines = []
        for idx, (name, url) in enumerate(event_links[:6], start=1):
            body_lines.append(
                f"{idx}. {name} ({url}) keeps builders and investors comparing playbooks, "
                "sharing product lessons, and meeting future collaborators."
            )
        body = '\n'.join(body_lines)
        closing = (
            "Seats typically vanish quickly for these gatherings, so confirm your RSVP early "
            "and bring a teammate to multiply the takeaways."
        )
        return f"{intro}\n\nEvent Highlights:\n{body}\n\n{closing}"

    if combined_text.strip():
        excerpt = combined_text.strip()
        if len(excerpt) > 1200:
            excerpt = excerpt[:1200].rsplit('\n', 1)[0]
        return (
            "Here is a condensed overview drawn from the scraped calendars when the AI "
            "essay service was unavailable:\n\n"
            f"{excerpt}\n\nStay tuned for fresh write-ups as soon as the generator is back."
        )

    return None


def generate_essay(combined_events_content=None):
    """Generate an essay based on combined events content and display it"""
    try:
        if combined_events_content:
            # Use provided combined events content
            selected = combined_events_content
            event_links = _extract_event_links(selected)
            links_guidance = ""
            if event_links:
                formatted_links = "\n".join(f"- {name}: {url}" for name, url in event_links)
                links_guidance = (
                    "Here is the list of events with their required URLs. Use these exact names and URLs and do not invent new ones.\n"
                    f"{formatted_links}\n\n"
                )

            prompt = (
                "Generate an engaging essay about the upcoming AI events using the provided information. "
                "Write in a way that encourages readers to attend these events and highlights the exciting opportunities in the AI community. "
                "Every time you mention an event by name, immediately include its event URL in parentheses right after the event name, e.g., 'AI Summit (https://lu.ma/ai-summit-2024)'. "
                "Do not reference an event without its URL, and only use URLs supplied below or in the source content.\n\n"
                f"{links_guidance}"
                f"Event source material:\n{selected}"
            )
        else:
            st.warning("No events data available. Please scrape events first.")
            return False, None

        # Generate essay using OpenAI GPT (some models require default temperature of 1.0)
        raw_result = None
        if ANTHROPIC_API_KEY:
            try:
                raw_result = generate_with_claude(
                    prompt,
                    temperature=1.0,
                    max_tokens=3000,
                )
            except Exception as claude_error:  # noqa: BLE001
                logger.warning("Anthropic Claude request failed: %s", claude_error)
                st.warning("Anthropic Sonnet 4.5 unavailable; falling back to GPT.")

        if raw_result is None:
            raw_result = generate_with_gpt(
                prompt,
                temperature=1.0,
                max_tokens=3000,
                model_override=PREFERRED_GPT_MODEL,
            )
        cleaned_result = raw_result.strip()
        logger.info("Essay generated with %d characters", len(cleaned_result))
        print(f"Essay preview (first 200 chars): {cleaned_result[:200]}")

        if not cleaned_result:
            logger.warning("Essay generation returned empty text; using fallback")
            fallback = _build_fallback_essay(event_links, selected)
            if fallback:
                return True, fallback
            return False, None

        return True, cleaned_result
    except Exception as e:
        logger.error("Essay generation failed: %s", e)
        fallback = _build_fallback_essay(event_links if 'event_links' in locals() else [], selected if 'selected' in locals() else '')
        if fallback:
            st.warning("LLM essay generation failed, showing fallback summary instead.")
            return True, fallback
        st.error(f"Error generating essay: {str(e)}")
        return False, None


def generate_event_image(combined_events_content=None):
    """Generate an image based on the events using Google's Nano Banana 2 model."""
    try:
        if not combined_events_content:
            return False, None, "No events data available. Please scrape events first."

        if not GOOGLE_API_KEY:
            return False, None, "Google API key is not configured. Set GOOGLE_API_KEY in secrets or env."

        prompt_for_image = (
            "Create a clean modern image for AI events, "
            "use small number of texts which relates to the events"
        )

        event_names = []
        lines = combined_events_content.split('\n')
        for line in lines:
            bold_matches = re.findall(r'\*\*([^*]+)\*\*', line)
            if bold_matches:
                event_names.extend(bold_matches)

        if event_names:
            prompt_for_image = (
                "Create a clean modern image for AI events relating to the events names, "
                "use small number of texts which relates to the events"
            )

        client = genai.Client(api_key=GOOGLE_API_KEY)
        response = client.models.generate_content(
            model=NANO_BANANA_2_MODEL,
            contents=prompt_for_image,
            config=types.GenerateContentConfig(
                response_modalities=["TEXT", "IMAGE"],
            ),
        )

        for part in response.parts:
            if part.inline_data is not None:
                image_data = part.inline_data.data
                mime_type = part.inline_data.mime_type or "image/png"
                return True, {"bytes": image_data, "mime_type": mime_type}, None

        error_msg = "Nano Banana 2 returned no image data"
        print(f"❌ {error_msg}")
        return False, None, error_msg

    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        print(f"❌ {error_msg}")
        return False, None, error_msg


def generate_events(url="https://lu.ma/genai-sf?k=c", source_name="Lu.ma GenAI SF", days=8):
    """Go to specified URL and get the events for the specified number of days"""
    try:
        if "cerebralvalley.ai" in url:
            events_list = scrape_cerebral_valley_events(days)
            if not events_list:
                st.error("Failed to extract Cerebral Valley events")
                return False, None
            formatted_events = format_cerebral_valley_list(events_list, source_name, days)
        elif url in {"LUMA_COMBINED", "LUMA_BAY_AREA", "LUMA_NEW_YORK"} or _is_luma_url(url):
            # Use direct HTTP scraping for Luma events
            source_urls = _get_luma_source_urls(url)
            if source_urls is not None:
                events_list = _collect_luma_events(source_urls, days)
            else:
                target_url = _normalize_luma_url(url)
                events_list = scrape_luma_events(target_url, days)

            if not events_list:
                st.error("Failed to extract Luma events")
                return False, None
            formatted_events = format_cerebral_valley_list(events_list, source_name, days)
        else:
            # Run the async scraping function for other sources
            events_data = asyncio.run(scrape_events(url, source_name, days))

            if not events_data:
                st.error(f"Failed to retrieve events from {source_name}")
                return False, None

            # Format the events for display
            formatted_events = format_events_for_doc(events_data, source_name, days)

        # Display results on the page
        st.subheader(f"📅 {source_name} Events")
        st.markdown("**Scraped Events:**")

        # Display as rendered markdown instead of plain text
        with st.container():
            st.markdown(formatted_events)

        # No longer writing to Google Doc
        print(f"✅ Successfully scraped {source_name} events.")

        return True, formatted_events
    except Exception as e:
        st.error(f"Error in generate_events: {str(e)}")
        return False, None


def format_events_for_doc(events_data, source_name="Events", days=8):
    """Format the scraped events into a readable document format"""
    try:
        # Get current date and next specified days
        today = datetime.now()
        end_date = today + timedelta(days=days)

        # Create header
        formatted_text = f"{source_name} Events - {today.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}\n\n"
        formatted_text += "=" * 50 + "\n\n"

        # Extract clean event information from agent result
        events_content = extract_events_from_agent_result(events_data)

        if events_content:
            formatted_text += events_content
        else:
            formatted_text += "No events found or error parsing events data.\n"

        formatted_text += "\n\n" + "=" * 50 + "\n"
        formatted_text += f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        return formatted_text
    except Exception as e:
        return f"Error formatting events: {str(e)}\n\nRaw data:\n{str(events_data)[:500]}..."




def extract_events_from_agent_result(agent_result):
    """Extract clean event information from browser-use agent result"""
    try:
        # Convert agent result to string for parsing
        result_str = str(agent_result)

        # Look for content between <result> tags or extract from extracted_content
        import re

        # Try to find content in <result> tags
        result_match = re.search(r'<result>(.*?)</result>', result_str, re.DOTALL)
        if result_match:
            content = result_match.group(1).strip()
            return clean_event_content(content)

        # Try to find extracted_content with events
        extracted_match = re.search(r"extracted_content='[^']*### AI/GenAI Events[^']*'", result_str)
        if extracted_match:
            content = extracted_match.group(0)
            # Extract the actual content
            content_match = re.search(r"extracted_content='([^']*)'", content)
            if content_match:
                return clean_event_content(content_match.group(1))

        # Try to find any content with "Event Name:" pattern
        event_pattern = r'Event Name:.*?(?=Event Name:|$)'
        events = re.findall(event_pattern, result_str, re.DOTALL)
        if events:
            return clean_event_content('\n\n'.join(events).strip())

        # If no structured format found, try to extract readable content
        # Look for lines that contain event-like information
        lines = result_str.split('\n')
        event_lines = []
        for line in lines:
            line = line.strip()
            if any(keyword in line.lower() for keyword in ['event', 'date:', 'location:', 'description:', 'link:']):
                if not line.startswith('ActionResult') and not 'extracted_content=' in line:
                    event_lines.append(line)

        if event_lines:
            return clean_event_content('\n'.join(event_lines))

        return "Unable to parse event information from agent result."

    except Exception as e:
        return f"Error extracting events: {str(e)}"


def clean_event_content(content):
    """Clean event data so titles carry their URLs."""
    import re

    # Normalize escape characters and split into meaningful lines
    content = content.replace('\\n', '\n').replace('\\t', '\t')
    raw_lines = [line.rstrip() for line in content.split('\n') if line.strip()]

    processed_lines: List[str] = []
    events: List[dict] = []

    def _normalize_url(url_text: str) -> Optional[str]:
        if not url_text:
            return None
        original = url_text
        url_text = url_text.strip().strip('.,)')
        if not url_text:
            logger.warning("Discarded empty URL after stripping: %s", original)
            return None
        if url_text.startswith('/'):
            normalized = _normalize_luma_url(url_text)
        elif url_text.startswith('lu.ma/'):
            normalized = _normalize_luma_url(f"https://{url_text}")
        elif url_text.startswith('luma.com/'):
            normalized = _normalize_luma_url(f"https://{url_text}")
        elif url_text.startswith('www.'):
            normalized = f"https://{url_text}"
        else:
            normalized = _normalize_luma_url(url_text)
        logger.info("URL normalized from %s to %s", original.strip(), normalized)
        return normalized

    def _record_event_label(before: str, title: str) -> None:
        events.append({
            'index': len(processed_lines),
            'type': 'label',
            'before': before.rstrip(),
            'title': title.strip(),
            'url': None,
        })

    def _record_event_bold(match) -> None:
        events.append({
            'index': len(processed_lines),
            'type': 'bold',
            'lead': match.group('lead') or '',
            'title': match.group('title').strip(),
            'trail': match.group('trail') or '',
            'url': None,
        })

    for raw_line in raw_lines:
        line = raw_line

        # Fix example.com or relative markdown links before parsing
        if 'example.com' in line or '[Event Link](/' in line or '**Link:**' in line or 'Link:' in line:
            line = fix_example_com_urls(line)

        colon_idx = line.find(':')
        if colon_idx != -1:
            before = line[:colon_idx]
            after = line[colon_idx + 1 :]
            before_lower = before.lower()

            if any(key in before_lower for key in ['event url', 'url', 'link']):
                url_match = re.search(r'(https?://[^\s)]+)', after)
                url_value = url_match.group(1) if url_match else after.strip()
                url_value = _normalize_url(url_value)
                if url_value and events:
                    events[-1]['url'] = url_value
                    logger.info(
                        "Captured Event URL for title '%s' from line: %s",
                        events[-1]['title'],
                        raw_line,
                    )
                    continue  # Skip standalone URL line

            if 'event name' in before_lower or before_lower.strip().startswith('event '):
                _record_event_label(before, after)
                processed_lines.append(line)
                continue

        bold_match = re.match(r'^(?P<lead>\s*(?:[-*]\s+|\d+\.\s+)?)\*\*(?P<title>[^*]+)\*\*(?P<trail>.*)$', line)
        if bold_match:
            _record_event_bold(bold_match)
            processed_lines.append(line)
            continue

        processed_lines.append(line)

    # Apply collected URLs to their respective title lines
    for event in events:
        url = event.get('url')
        if not url:
            continue

        if event['type'] == 'label':
            linked = f"[{event['title']}]({url})"
            processed_lines[event['index']] = f"{event['before']}: {linked}"
            logger.info(
                "Applied URL %s to label event title '%s'",
                url,
                event['title'],
            )
        else:
            linked = f"[{event['title']}]({url})"
            processed_lines[event['index']] = f"{event['lead']}**{linked}**{event['trail']}"
            logger.info(
                "Applied URL %s to numbered/bold event title '%s'",
                url,
                event['title'],
            )

    return '\n'.join(processed_lines)


def _format_event_entry_lines(event: Dict) -> List[str]:
    lines: List[str] = []
    title = _ensure_text(event.get('title', 'Untitled Event'), 'Untitled Event').strip()
    url = _ensure_text(event.get('url', '')).strip()
    if url:
        lines.append(f"- **[{title}]({url})**")
    else:
        lines.append(f"- **{title}**")

    date_text = _ensure_text(event.get('date_text', '')).strip()
    time_text = _ensure_text(event.get('time_text', '')).strip()
    datetime_parts = ' '.join(part for part in [date_text, time_text] if part)
    if datetime_parts:
        lines.append(f"  Date and Time: {datetime_parts}")

    location = _ensure_text(event.get('location', '')).strip()
    if location:
        lines.append(f"  Location/Venue: {location}")

    host = _ensure_text(event.get('host', '')).strip()
    if host:
        lines.append(f"  Host: {host}")

    description = _ensure_text(event.get('description', '')).strip()
    if description:
        if len(description) > 300:
            description = description[:297].rstrip() + '...'
        lines.append(f"  Brief Description: {description}")

    return lines


def _filter_events_for_date_range(events: List[Dict], days: int) -> List[Dict]:
    today = datetime.now()
    end_date = today + timedelta(days=days)
    today_floor = today.replace(hour=0, minute=0, second=0, microsecond=0)
    local_tz = datetime.now().astimezone().tzinfo

    def _normalize_event_dt(value: Optional[datetime]) -> Optional[datetime]:
        if not value:
            return None
        if value.tzinfo is None or local_tz is None:
            return value.replace(tzinfo=None)
        return value.astimezone(local_tz).replace(tzinfo=None)

    filtered_events: List[Dict] = []
    for event in events:
        event_dt: Optional[datetime] = None
        start_iso = event.get('start_iso')
        if isinstance(start_iso, str) and start_iso:
            event_dt = _parse_iso_datetime(start_iso)
        if not event_dt:
            date_text = _ensure_text(event.get('date_text', '')).strip()
            if date_text:
                for fmt in ('%B %d, %Y', '%b %d, %Y'):
                    try:
                        event_dt = datetime.strptime(date_text, fmt)
                        break
                    except ValueError:
                        continue

        event_dt = _normalize_event_dt(event_dt)

        if event_dt and event_dt > end_date:
            logger.info(
                "Skipping event %s (date %s beyond %s days)",
                event.get('title'),
                event_dt,
                days,
            )
            continue
        if event_dt and event_dt < today_floor:
            logger.info(
                "Skipping outdated event %s (date %s before today)",
                event.get('title'),
                event_dt,
            )
            continue

        filtered_events.append(event)

    return filtered_events


def _format_event_collection(
    events: List[Dict],
    source_name: str,
    days: int,
    empty_message: str,
) -> str:
    today = datetime.now()
    end_date = today + timedelta(days=days)
    lines = [
        f"{source_name} Events - {today.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}",
        "=" * 50,
        "",
    ]

    if not events:
        lines.append(empty_message)
        return '\n'.join(lines)

    filtered_events = _filter_events_for_date_range(events, days)
    if not filtered_events:
        lines.append("No events within the requested date range")
        return '\n'.join(lines)

    for idx, event in enumerate(filtered_events):
        lines.extend(_format_event_entry_lines(event))
        if idx < len(filtered_events) - 1:
            lines.append("")

    return '\n'.join(lines)


def format_cerebral_valley_list(events, source_name="Cerebral Valley", days=8):
    formatted = _format_event_collection(
        _enrich_events_with_details(events) if events else [],
        source_name,
        days,
        "No events found on cerebralvalley.ai/events",
    )
    return '\n'.join(
        [
            formatted,
            "",
            "=" * 50,
            f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
    )


def _split_focus_region_events(events: List[Dict]) -> Dict[str, List[Dict]]:
    grouped: Dict[str, List[Dict]] = {region: [] for region in FOCUS_REGION_ORDER}
    for event in events:
        location = _clean_location_for_display(_ensure_text(event.get('location', '')))
        region = _classify_event_region(location)
        if region in grouped:
            grouped[region].append(event)
    return grouped


def generate_luma_region_events(days: int = 8):
    raw_events = _collect_luma_events(_get_luma_source_urls("LUMA_COMBINED") or [], days)
    detailed_events = _enrich_events_with_details(raw_events) if raw_events else []
    split_events = _split_focus_region_events(detailed_events)

    region_results: Dict[str, Dict[str, object]] = {}
    for region_name in FOCUS_REGION_ORDER:
        config = LUMA_REGION_SOURCES[region_name]
        region_events = split_events.get(region_name, [])
        region_results[region_name] = {
            "formatted": _format_event_collection(
                region_events,
                config["source_name"],
                days,
                f"No {region_name} events found on Lu.ma.",
            ),
            "has_events": bool(region_events),
            "caption": config["caption"],
        }

    combined_formatted = _format_event_collection(
        detailed_events,
        "Lu.ma Events",
        days,
        "No events found on Lu.ma",
    )
    return bool(detailed_events), combined_formatted, region_results


def generate_cerebral_valley_region_events(days: int = 8):
    raw_events = scrape_cerebral_valley_events(days)
    detailed_events = _enrich_events_with_details(raw_events) if raw_events else []
    split_events = _split_focus_region_events(detailed_events)

    region_results: Dict[str, Dict[str, object]] = {}
    for region_name in FOCUS_REGION_ORDER:
        region_events = split_events.get(region_name, [])
        region_results[region_name] = {
            "formatted": _format_event_collection(
                region_events,
                f"Cerebral Valley {region_name}",
                days,
                f"No {region_name} events found in Cerebral Valley.",
            ),
            "has_events": bool(region_events),
            "caption": CV_REGION_PAGE_URLS[region_name],
        }

    combined_formatted = '\n'.join(
        [
            _format_event_collection(
                detailed_events,
                "Cerebral Valley",
                days,
                "No events found on cerebralvalley.ai/events",
            ),
            "",
            "=" * 50,
            f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        ]
    )
    return bool(raw_events), combined_formatted, region_results


def _render_focus_region_tabs(region_results: Dict[str, Dict[str, object]], key_prefix: str) -> None:
    tabs = st.tabs(FOCUS_REGION_ORDER)
    for index, region_name in enumerate(FOCUS_REGION_ORDER):
        result = region_results.get(region_name, {})
        formatted = _ensure_text(result.get("formatted", ""))
        caption = _ensure_text(result.get("caption", ""))
        has_events = bool(result.get("has_events"))

        with tabs[index]:
            st.markdown(f"**{region_name}**")
            if caption:
                st.caption(caption)
            if has_events:
                render_copy_button(formatted, f"{key_prefix}-{region_name.lower().replace(' ', '-')}")
            with st.expander(f"View {region_name} events", expanded=True):
                st.markdown(formatted)


def _build_region_combined_events(source_region_results: Dict[str, Dict[str, Dict[str, object]]]) -> Dict[str, str]:
    combined_by_region: Dict[str, str] = {}
    for region_name in FOCUS_REGION_ORDER:
        valid_events = []
        for source_label, region_results in source_region_results.items():
            result = (region_results or {}).get(region_name) or {}
            if not result.get("has_events"):
                continue
            formatted = _ensure_text(result.get("formatted", "")).strip()
            if formatted:
                valid_events.append((source_label, formatted))
        if valid_events:
            combined_by_region[region_name] = parse_and_format_combined_events(valid_events)
    return combined_by_region


def _refresh_region_combined_state() -> None:
    source_region_results: Dict[str, Dict[str, Dict[str, object]]] = {}
    if st.session_state.get("luma_region_results"):
        source_region_results["Lu.ma Events"] = st.session_state.luma_region_results
    if st.session_state.get("cv_region_results"):
        source_region_results["Cerebral Valley Events"] = st.session_state.cv_region_results
    if st.session_state.get("evion_region_results"):
        source_region_results["Evion Events"] = st.session_state.evion_region_results
    if st.session_state.get("meetup_region_results"):
        source_region_results["Meetup Events"] = st.session_state.meetup_region_results

    combined_by_region = _build_region_combined_events(source_region_results)
    if not combined_by_region:
        st.session_state.pop("region_combined_events", None)
        return

    st.session_state.region_combined_events = combined_by_region
    current_region = st.session_state.get("selected_event_region")
    if current_region not in combined_by_region:
        current_region = next((region for region in FOCUS_REGION_ORDER if region in combined_by_region), None)
        if current_region:
            st.session_state.selected_event_region = current_region
    if current_region:
        st.session_state.combined_events = combined_by_region[current_region]


def _get_selected_events_content(selectbox_key: str, label: str = "Region") -> Optional[str]:
    combined_by_region = st.session_state.get("region_combined_events") or {}
    valid_regions = [region for region in FOCUS_REGION_ORDER if region in combined_by_region]
    if valid_regions:
        current_region = st.session_state.get("selected_event_region")
        if current_region not in valid_regions:
            current_region = valid_regions[0]
        selected_region = st.selectbox(
            label,
            options=valid_regions,
            index=valid_regions.index(current_region),
            key=selectbox_key,
        )
        st.session_state.selected_event_region = selected_region
        st.session_state.combined_events = combined_by_region[selected_region]
        return combined_by_region[selected_region]
    return st.session_state.get("combined_events")


def _clear_event_workspace_state() -> None:
    keys_to_clear = [
        "luma_region_results",
        "cv_region_results",
        "evion_region_results",
        "meetup_region_results",
        "region_combined_events",
        "combined_events",
        "selected_event_region",
        "organized_events",
        "organized_events_mode",
        "organized_events_cache",
        "loaded_events_source",
    ]
    for key in keys_to_clear:
        st.session_state.pop(key, None)


def _build_events_snapshot_payload() -> Optional[Dict[str, object]]:
    luma_region_results = st.session_state.get("luma_region_results") or {}
    cv_region_results = st.session_state.get("cv_region_results") or {}
    evion_region_results = st.session_state.get("evion_region_results") or {}
    meetup_region_results = st.session_state.get("meetup_region_results") or {}
    region_combined_events = st.session_state.get("region_combined_events") or {}
    combined_events = _ensure_text(st.session_state.get("combined_events", ""))

    if not any([
        luma_region_results,
        cv_region_results,
        evion_region_results,
        meetup_region_results,
        region_combined_events,
        combined_events.strip(),
    ]):
        return None

    payload: Dict[str, object] = {
        "format": EVENT_SNAPSHOT_FORMAT,
        "saved_at": datetime.now().isoformat(timespec="seconds"),
        "selected_event_region": st.session_state.get("selected_event_region"),
        "luma_region_results": luma_region_results,
        "cv_region_results": cv_region_results,
        "evion_region_results": evion_region_results,
        "meetup_region_results": meetup_region_results,
        "region_combined_events": region_combined_events,
        "combined_events": combined_events,
        "organized_events": _ensure_text(st.session_state.get("organized_events", "")),
        "organized_events_mode": _ensure_text(st.session_state.get("organized_events_mode", "")),
    }
    return payload


def _is_events_snapshot_payload(payload: object) -> bool:
    return isinstance(payload, dict) and payload.get("format") == EVENT_SNAPSHOT_FORMAT


def _restore_events_snapshot_payload(payload: Dict[str, object]) -> bool:
    if not _is_events_snapshot_payload(payload):
        return False

    _clear_event_workspace_state()

    selected_region = payload.get("selected_event_region")
    if selected_region in FOCUS_REGION_ORDER:
        st.session_state.selected_event_region = selected_region

    luma_region_results = payload.get("luma_region_results")
    if isinstance(luma_region_results, dict):
        st.session_state.luma_region_results = luma_region_results

    cv_region_results = payload.get("cv_region_results")
    if isinstance(cv_region_results, dict):
        st.session_state.cv_region_results = cv_region_results

    evion_region_results = payload.get("evion_region_results")
    if isinstance(evion_region_results, dict):
        st.session_state.evion_region_results = evion_region_results

    meetup_region_results = payload.get("meetup_region_results")
    if isinstance(meetup_region_results, dict):
        st.session_state.meetup_region_results = meetup_region_results

    if (
        st.session_state.get("luma_region_results")
        or st.session_state.get("cv_region_results")
        or st.session_state.get("evion_region_results")
        or st.session_state.get("meetup_region_results")
    ):
        _refresh_region_combined_state()

    region_combined_events = payload.get("region_combined_events")
    if isinstance(region_combined_events, dict) and not st.session_state.get("region_combined_events"):
        st.session_state.region_combined_events = region_combined_events

    combined_events = _ensure_text(payload.get("combined_events", ""))
    if combined_events:
        st.session_state.combined_events = combined_events

    if st.session_state.get("region_combined_events"):
        selected = st.session_state.get("selected_event_region")
        if selected in st.session_state.region_combined_events:
            st.session_state.combined_events = st.session_state.region_combined_events[selected]

    organized_events = _ensure_text(payload.get("organized_events", ""))
    if organized_events:
        st.session_state.organized_events = organized_events

    organized_events_mode = _ensure_text(payload.get("organized_events_mode", ""))
    if organized_events_mode:
        st.session_state.organized_events_mode = organized_events_mode

    return bool(
        st.session_state.get("luma_region_results")
        or st.session_state.get("cv_region_results")
        or st.session_state.get("evion_region_results")
        or st.session_state.get("meetup_region_results")
        or st.session_state.get("region_combined_events")
        or st.session_state.get("combined_events")
    )


def _describe_snapshot_payload(payload: Dict[str, object]) -> str:
    lines: List[str] = []
    saved_at = _ensure_text(payload.get("saved_at", ""))
    if saved_at:
        lines.append(f"Saved at: {saved_at}")

    for region_name in FOCUS_REGION_ORDER:
        sources: List[str] = []
        luma_result = ((payload.get("luma_region_results") or {}) if isinstance(payload.get("luma_region_results"), dict) else {}).get(region_name, {})
        cv_result = ((payload.get("cv_region_results") or {}) if isinstance(payload.get("cv_region_results"), dict) else {}).get(region_name, {})
        evion_result = ((payload.get("evion_region_results") or {}) if isinstance(payload.get("evion_region_results"), dict) else {}).get(region_name, {})
        meetup_result = ((payload.get("meetup_region_results") or {}) if isinstance(payload.get("meetup_region_results"), dict) else {}).get(region_name, {})
        if isinstance(luma_result, dict) and luma_result.get("has_events"):
            sources.append("Lu.ma")
        if isinstance(cv_result, dict) and cv_result.get("has_events"):
            sources.append("Cerebral Valley")
        if isinstance(evion_result, dict) and evion_result.get("has_events"):
            sources.append("Evion")
        if isinstance(meetup_result, dict) and meetup_result.get("has_events"):
            sources.append("Meetup")
        if sources:
            lines.append(f"{region_name}: {', '.join(sources)}")

    if not lines:
        lines.append("Snapshot contains saved event text.")
    return "\n".join(lines)


def _render_region_summary_tabs(
    luma_region_results: Optional[Dict[str, Dict[str, object]]],
    cv_region_results: Optional[Dict[str, Dict[str, object]]],
    evion_region_results: Optional[Dict[str, Dict[str, object]]],
    meetup_region_results: Optional[Dict[str, Dict[str, object]]],
    key_prefix: str,
) -> None:
    combined_by_region = _build_region_combined_events(
        {
            "Lu.ma Events": luma_region_results or {},
            "Cerebral Valley Events": cv_region_results or {},
            "Evion Events": evion_region_results or {},
            "Meetup Events": meetup_region_results or {},
        }
    )
    tabs = st.tabs(FOCUS_REGION_ORDER)
    for index, region_name in enumerate(FOCUS_REGION_ORDER):
        with tabs[index]:
            st.markdown(f"**{region_name}**")

            luma_result = (luma_region_results or {}).get(region_name)
            if luma_result:
                st.markdown("Lu.ma")
                if luma_result.get("has_events"):
                    render_copy_button(
                        _ensure_text(luma_result.get("formatted", "")),
                        f"{key_prefix}-luma-{region_name.lower().replace(' ', '-')}",
                    )
                st.markdown(_ensure_text(luma_result.get("formatted", "")))

            cv_result = (cv_region_results or {}).get(region_name)
            if cv_result:
                st.markdown("Cerebral Valley")
                if cv_result.get("has_events"):
                    render_copy_button(
                        _ensure_text(cv_result.get("formatted", "")),
                        f"{key_prefix}-cv-{region_name.lower().replace(' ', '-')}",
                    )
                st.markdown(_ensure_text(cv_result.get("formatted", "")))

            evion_result = (evion_region_results or {}).get(region_name)
            if evion_result:
                st.markdown("Evion")
                if evion_result.get("has_events"):
                    render_copy_button(
                        _ensure_text(evion_result.get("formatted", "")),
                        f"{key_prefix}-evion-{region_name.lower().replace(' ', '-')}",
                    )
                st.markdown(_ensure_text(evion_result.get("formatted", "")))

            meetup_result = (meetup_region_results or {}).get(region_name)
            if meetup_result:
                st.markdown("Meetup")
                if meetup_result.get("has_events"):
                    render_copy_button(
                        _ensure_text(meetup_result.get("formatted", "")),
                        f"{key_prefix}-meetup-{region_name.lower().replace(' ', '-')}",
                    )
                st.markdown(_ensure_text(meetup_result.get("formatted", "")))

            combined_text = combined_by_region.get(region_name)
            if combined_text:
                col_header1, col_header2 = st.columns([3, 1])
                with col_header1:
                    st.markdown(f"**Combined {region_name} (All Sources)**")
                with col_header2:
                    render_copy_button(
                        combined_text,
                        f"{key_prefix}-combined-{region_name.lower().replace(' ', '-')}",
                    )
                with st.expander(f"View combined {region_name} events", expanded=False):
                    st.markdown(combined_text)
            else:
                st.info(f"No {region_name} events available across current sources.")


def _scrape_luma_region_to_state(region_name: str, days: int) -> bool:
    config = LUMA_REGION_SOURCES[region_name]
    raw_events = _collect_luma_events(config["urls"], days)
    detailed_events = _enrich_events_with_details(raw_events) if raw_events else []
    region_results = dict(st.session_state.get("luma_region_results") or {})
    region_results[region_name] = {
        "formatted": _format_event_collection(
            detailed_events,
            config["source_name"],
            days,
            f"No {region_name} events found on Lu.ma.",
        ),
        "has_events": bool(detailed_events),
        "caption": config["caption"],
    }
    st.session_state.luma_region_results = region_results
    st.session_state.selected_event_region = region_name
    _refresh_region_combined_state()
    return bool(detailed_events)


def _scrape_cerebral_valley_region_to_state(region_name: str, days: int) -> bool:
    raw_events = scrape_cerebral_valley_events(days)
    detailed_events = _enrich_events_with_details(raw_events) if raw_events else []
    split_events = _split_focus_region_events(detailed_events)
    region_events = split_events.get(region_name, [])
    region_results = dict(st.session_state.get("cv_region_results") or {})
    region_results[region_name] = {
        "formatted": _format_event_collection(
            region_events,
            f"Cerebral Valley {region_name}",
            days,
            f"No {region_name} events found in Cerebral Valley.",
        ),
        "has_events": bool(region_events),
        "caption": CV_REGION_PAGE_URLS[region_name],
    }
    st.session_state.cv_region_results = region_results
    st.session_state.selected_event_region = region_name
    _refresh_region_combined_state()
    return bool(region_events)


def _scrape_evion_region_to_state(region_name: str, days: int) -> bool:
    _, _, source_region_results = generate_evion_region_events(days)
    result = (source_region_results or {}).get(region_name) or {}
    region_results = dict(st.session_state.get("evion_region_results") or {})
    region_results[region_name] = result
    st.session_state.evion_region_results = region_results
    st.session_state.selected_event_region = region_name
    _refresh_region_combined_state()
    return bool(result.get("has_events"))


def _scrape_meetup_region_to_state(region_name: str, days: int) -> bool:
    config = MEETUP_REGION_SOURCES[region_name]
    raw_events = scrape_meetup_events(config["url"], region_name)
    detailed_events = _sort_events_by_start(_filter_events_for_date_range(raw_events, days))
    region_results = dict(st.session_state.get("meetup_region_results") or {})
    region_results[region_name] = {
        "formatted": _format_event_collection(
            detailed_events,
            config["source_name"],
            days,
            f"No {region_name} events found on Meetup.",
        ),
        "has_events": bool(detailed_events),
        "caption": config["caption"],
    }
    st.session_state.meetup_region_results = region_results
    st.session_state.selected_event_region = region_name
    _refresh_region_combined_state()
    return bool(detailed_events)


def _render_region_source_section(
    source_label: str,
    result: Optional[Dict[str, object]],
    key_prefix: str,
    empty_message: str,
) -> None:
    st.markdown(f"**{source_label}**")
    if not result:
        st.info(empty_message)
        return

    caption = _ensure_text(result.get("caption", ""))
    formatted = _ensure_text(result.get("formatted", ""))
    has_events = bool(result.get("has_events"))

    if caption:
        st.caption(caption)
    if has_events:
        render_copy_button(formatted, key_prefix)
    with st.expander(f"View {source_label}", expanded=False):
        st.markdown(formatted)


def _render_region_tab_content(region_name: str, days_to_scrape: int) -> None:
    luma_region_results = st.session_state.get("luma_region_results") or {}
    cv_region_results = st.session_state.get("cv_region_results") or {}
    evion_region_results = st.session_state.get("evion_region_results") or {}
    meetup_region_results = st.session_state.get("meetup_region_results") or {}

    action_col1, action_col2, action_col3, action_col4, action_col5 = st.columns([1, 1, 1, 1, 1])
    with action_col1:
        if st.button(f"Scrape Lu.ma {region_name}", key=f"luma_button_{region_name}", width="stretch"):
            with st.spinner(f"Scraping Lu.ma {region_name} events..."):
                success = _scrape_luma_region_to_state(region_name, days_to_scrape)
                result = (st.session_state.get("luma_region_results") or {}).get(region_name)
                if success and result:
                    saved = _auto_save_results(
                        _ensure_text(result.get("formatted", "")),
                        f"luma_{region_name.lower().replace(' ', '_')}_events",
                    )
                    if saved:
                        st.success(f"✅ Lu.ma {region_name} events saved to `{os.path.basename(saved)}`")
                    else:
                        st.success(f"✅ Lu.ma {region_name} events scraped successfully!")
                else:
                    st.error(f"❌ Failed to scrape Lu.ma {region_name} events")

    with action_col2:
        if st.button(
            f"Scrape Cerebral Valley {region_name}",
            key=f"cv_button_{region_name}",
            width="stretch",
        ):
            with st.spinner(f"Scraping Cerebral Valley {region_name} events..."):
                success = _scrape_cerebral_valley_region_to_state(region_name, days_to_scrape)
                result = (st.session_state.get("cv_region_results") or {}).get(region_name)
                if success and result:
                    saved = _auto_save_results(
                        _ensure_text(result.get("formatted", "")),
                        f"cerebral_valley_{region_name.lower().replace(' ', '_')}_events",
                    )
                    if saved:
                        st.success(
                            f"✅ Cerebral Valley {region_name} events saved to `{os.path.basename(saved)}`"
                        )
                    else:
                        st.success(f"✅ Cerebral Valley {region_name} events scraped successfully!")
                else:
                    st.error(f"❌ Failed to scrape Cerebral Valley {region_name} events")

    with action_col3:
        if st.button(f"Scrape Evion {region_name}", key=f"evion_button_{region_name}", width="stretch"):
            with st.spinner(f"Scraping Evion {region_name} events..."):
                success = _scrape_evion_region_to_state(region_name, days_to_scrape)
                result = (st.session_state.get("evion_region_results") or {}).get(region_name)
                if result and result.get("has_events"):
                    saved = _auto_save_results(
                        _ensure_text(result.get("formatted", "")),
                        f"evion_{region_name.lower().replace(' ', '_')}_events",
                    )
                    if saved:
                        st.success(f"✅ Evion {region_name} events saved to `{os.path.basename(saved)}`")
                    else:
                        st.success(f"✅ Evion {region_name} events scraped successfully!")
                elif result:
                    st.info(
                        f"ℹ️ Evion scraped successfully, but no {region_name} events were present in the current Evion feed."
                    )
                    st.caption(EVION_EVENTS_PAGE_URL)
                elif not success:
                    st.error(f"❌ Failed to scrape Evion {region_name} events")
                else:
                    st.error(f"❌ Failed to scrape Evion {region_name} events")

    with action_col4:
        if st.button(f"Scrape Meetup {region_name}", key=f"meetup_button_{region_name}", width="stretch"):
            with st.spinner(f"Scraping Meetup {region_name} events..."):
                success = _scrape_meetup_region_to_state(region_name, days_to_scrape)
                result = (st.session_state.get("meetup_region_results") or {}).get(region_name)
                if success and result:
                    saved = _auto_save_results(
                        _ensure_text(result.get("formatted", "")),
                        f"meetup_{region_name.lower().replace(' ', '_')}_events",
                    )
                    if saved:
                        st.success(f"✅ Meetup {region_name} events saved to `{os.path.basename(saved)}`")
                    else:
                        st.success(f"✅ Meetup {region_name} events scraped successfully!")
                else:
                    st.error(f"❌ Failed to scrape Meetup {region_name} events")

    with action_col5:
        if st.button(f"Scrape All {region_name}", key=f"all_button_{region_name}", type="primary", width="stretch"):
            luma_ok = False
            cv_ok = False
            evion_ok = False
            meetup_ok = False
            with st.spinner(f"Scraping all {region_name} event sources..."):
                luma_ok = _scrape_luma_region_to_state(region_name, days_to_scrape)
                cv_ok = _scrape_cerebral_valley_region_to_state(region_name, days_to_scrape)
                evion_ok = _scrape_evion_region_to_state(region_name, days_to_scrape)
                meetup_ok = _scrape_meetup_region_to_state(region_name, days_to_scrape)
            if luma_ok or cv_ok or evion_ok or meetup_ok:
                combined_region_text = (st.session_state.get("region_combined_events") or {}).get(region_name)
                if combined_region_text:
                    saved = _auto_save_results(
                        combined_region_text,
                        f"combined_{region_name.lower().replace(' ', '_')}_events",
                    )
                    if saved:
                        st.info(f"💾 {region_name} combined events saved to `{os.path.basename(saved)}`")
                st.success(f"✅ {region_name} sources updated")
            else:
                st.warning(f"⚠️ No {region_name} events found from the loaded sources")

    st.divider()
    source_tab_luma, source_tab_cv, source_tab_evion, source_tab_meetup, source_tab_combined = st.tabs(
        ["Lu.ma", "Cerebral Valley", "Evion", "Meetup", "Combined"]
    )
    with source_tab_luma:
        _render_region_source_section(
            "Lu.ma",
            luma_region_results.get(region_name),
            f"luma-tab-{region_name.lower().replace(' ', '-')}",
            f"No Lu.ma {region_name} events loaded yet.",
        )
    with source_tab_cv:
        _render_region_source_section(
            "Cerebral Valley",
            cv_region_results.get(region_name),
            f"cv-tab-{region_name.lower().replace(' ', '-')}",
            f"No Cerebral Valley {region_name} events loaded yet.",
        )
    with source_tab_evion:
        _render_region_source_section(
            "Evion",
            evion_region_results.get(region_name),
            f"evion-tab-{region_name.lower().replace(' ', '-')}",
            f"No Evion {region_name} events loaded yet.",
        )
    with source_tab_meetup:
        _render_region_source_section(
            "Meetup",
            meetup_region_results.get(region_name),
            f"meetup-tab-{region_name.lower().replace(' ', '-')}",
            f"No Meetup {region_name} events loaded yet.",
        )
    with source_tab_combined:
        combined_text = (st.session_state.get("region_combined_events") or {}).get(region_name)
        st.markdown("**Combined Region Events**")
        if combined_text:
            col_header1, col_header2 = st.columns([3, 1])
            with col_header1:
                st.caption(f"All {region_name} events across loaded sources")
            with col_header2:
                render_copy_button(combined_text, f"combined-tab-{region_name.lower().replace(' ', '-')}")
            with st.expander(f"View combined {region_name} events", expanded=True):
                st.markdown(combined_text)
        else:
            st.info(f"No combined {region_name} events available yet.")


def fix_relative_urls(link_line):
    """Convert relative URLs to full lu.ma URLs"""
    import re

    # Pattern to match various URL formats in Link: lines
    url_patterns = [
        r'Link\*?\s*:\s*\[.*?\]\s*\(\s*(https://example\.com/[^\s\)]+)\s*\)',  # Markdown with spaces: [link] ( https://example.com/xyz )
        r'Link\*?\s*:\s*\[.*?\]\((https://example\.com/[^\)]+)\)',  # Standard markdown: [text](https://example.com/xyz)
        r'Link\*?\s*:\s*(https://example\.com/[^\s]+)',  # Direct example.com URLs
        r'Link\*?\s*:\s*(/[^\s]+)',  # Relative path like /event-name
        r'Link\*?\s*:\s*([a-zA-Z0-9-]+)(?:\s|$)',  # Just the event slug
    ]

    for pattern in url_patterns:
        match = re.search(pattern, link_line)
        if match:
            url_part = match.group(1).strip()  # Remove any extra whitespace
            # Convert to full lu.ma URL
            if url_part.startswith('/'):
                # Relative path
                full_url = _normalize_luma_url(url_part)
            elif 'example.com' in url_part:
                # Replace example.com with lu.ma
                event_id = url_part.split('/')[-1]
                full_url = f"{LUMA_BASE_URL}/{event_id}"
            elif not url_part.startswith('http'):
                # Just a slug
                full_url = f"{LUMA_BASE_URL}/{url_part}"
            else:
                # Already a full URL, but check if it needs hostname replacement
                if 'example.com' in url_part:
                    event_id = url_part.split('/')[-1]
                    full_url = f"{LUMA_BASE_URL}/{event_id}"
                else:
                    full_url = _normalize_luma_url(url_part)

            return f"Link: {full_url}"

    # If no pattern matched, return as is
    return link_line


def parse_and_format_combined_events(all_events):
    """Combine event source strings sequentially with clear source headers."""

    if not all_events:
        return "No event sources provided to combine."

    sections = []
    for idx, entry in enumerate(all_events):
        if isinstance(entry, tuple):
            label, content = entry[0], entry[1]
        else:
            label, content = (f"Source {idx + 1}", entry)

        if not content:
            continue

        label = label or f"Source {idx + 1}"
        cleaned = str(content).strip()
        if not cleaned:
            continue

        divider = "=" * len(label)
        sections.append(f"{label}\n{divider}\n\n{cleaned}")

    if not sections:
        return "No combined events available."

    header = "Combined Events Overview"
    header_line = "=" * len(header)
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    body = "\n\n---\n\n".join(sections)
    return f"{header}\n{header_line}\nGenerated: {timestamp}\n\n{body}"


def _canonicalize_event_url(url: str) -> str:
    if not url:
        return ""

    parsed = urlparse(url.strip())
    if not parsed.scheme or not parsed.netloc:
        return url.strip()

    filtered_query = [
        (key, value)
        for key, value in parse_qsl(parsed.query, keep_blank_values=True)
        if not key.lower().startswith("utm_") and key.lower() not in {"_bhlid", "tab"}
    ]
    path = parsed.path.rstrip("/") or "/"
    return urlunparse((
        parsed.scheme.lower(),
        parsed.netloc.lower(),
        path,
        "",
        urlencode(filtered_query),
        "",
    ))


def _parse_event_datetime_text(value: str) -> Optional[datetime]:
    if not value:
        return None

    cleaned = " ".join(value.strip().split())
    match = re.match(
        r"^(?P<date>[A-Za-z]+ \d{2}, \d{4}) (?P<time>\d{1,2}:\d{2} [AP]M)(?: (?P<tz>.+))?$",
        cleaned,
    )
    if not match:
        return None

    dt_value = datetime.strptime(
        f"{match.group('date')} {match.group('time')}",
        "%B %d, %Y %I:%M %p",
    )
    tz_text = (match.group("tz") or "").strip().upper()
    if not tz_text:
        return dt_value

    if tz_text in {"UTC", "GMT"}:
        return dt_value.replace(tzinfo=timezone.utc)

    if tz_text in {"PT", "PST"}:
        return dt_value.replace(tzinfo=timezone(timedelta(hours=-8)))
    if tz_text == "PDT":
        return dt_value.replace(tzinfo=timezone(timedelta(hours=-7)))
    if tz_text in {"ET", "EST"}:
        return dt_value.replace(tzinfo=timezone(timedelta(hours=-5)))
    if tz_text == "EDT":
        return dt_value.replace(tzinfo=timezone(timedelta(hours=-4)))

    offset_match = re.match(r"^(?:UTC|GMT)(?P<sign>[+-])(?P<hours>\d{2}):(?P<minutes>\d{2})$", tz_text)
    if offset_match:
        sign = 1 if offset_match.group("sign") == "+" else -1
        hours = int(offset_match.group("hours"))
        minutes = int(offset_match.group("minutes"))
        offset = timedelta(hours=hours, minutes=minutes) * sign
        return dt_value.replace(tzinfo=timezone(offset))

    return dt_value


def _format_organized_day_heading(dt_value: datetime) -> str:
    return dt_value.strftime("%A, %B %d").replace(" 0", " ")


def _format_organized_time(dt_value: datetime, raw_date_time: str) -> str:
    time_text = dt_value.strftime("%I:%M %p").lstrip("0")
    offset = dt_value.utcoffset()
    if offset is not None:
        offset_minutes = int(offset.total_seconds() // 60)
        if offset_minutes in {-480, -420}:
            return f"{time_text} PT"
        if offset_minutes in {-300, -240}:
            return f"{time_text} ET"
        if offset_minutes == 0:
            return f"{time_text} UTC"

    raw_parts = raw_date_time.strip().split()
    if raw_parts:
        last_token = raw_parts[-1].upper()
        if last_token not in {"AM", "PM"}:
            return f"{time_text} {last_token}"
    return time_text


def _clean_location_for_display(location: str) -> str:
    cleaned = " ".join(_ensure_text(location).split()).strip()
    if not cleaned:
        return "Location TBD"

    if cleaned.lower() in {"online", "online event", "virtual", "virtual event"}:
        return "Online"

    cleaned = cleaned.replace("Register to See Address", "Register for address")
    parts = [part.strip() for part in cleaned.split(",") if part.strip()]
    deduped_parts: List[str] = []
    seen = set()
    for part in parts:
        key = part.lower()
        if key in seen:
            continue
        seen.add(key)
        deduped_parts.append(part)
    return ", ".join(deduped_parts) if deduped_parts else cleaned


def _classify_event_region(location: str) -> str:
    lowered = location.lower()
    if any(keyword in lowered for keyword in ["online", "virtual", "zoom", "google meet"]):
        return "Online"
    if any(keyword in lowered for keyword in ["seattle", "washington", "portland", "oregon", "bellevue"]):
        return "Pacific Northwest"
    if any(keyword in lowered for keyword in ["new york", "nyc", "brooklyn", "manhattan", "queens", "bronx"]):
        return "New York"
    if any(keyword in lowered for keyword in ["boston", "cambridge", "somerville"]):
        return "Boston / Cambridge"
    if any(
        keyword in lowered
        for keyword in [
            "san francisco",
            "south san francisco",
            "berkeley",
            "oakland",
            "palo alto",
            "mountain view",
            "san jose",
            "menlo park",
            "fremont",
            "los gatos",
            "san carlos",
            "stanford",
            "sunnyvale",
            "santa clara",
            "alameda",
            "san ramon",
            "foster city",
            "milpitas",
            "millbrae",
            "redwood city",
            "san mateo",
            "burlingame",
            "cupertino",
            "embarcadero",
        ]
    ):
        return "Bay Area"
    return "Other"


def _shorten_event_description(text: str, max_length: int = 140) -> str:
    cleaned = re.sub(r"\s+", " ", _ensure_text(text)).strip()
    if not cleaned:
        return "Details available on the RSVP page."
    if len(cleaned) <= max_length:
        return cleaned

    sentence = re.split(r"(?<=[.!?])\s+", cleaned, maxsplit=1)[0].strip()
    if sentence and len(sentence) <= max_length:
        return sentence
    return cleaned[: max_length - 3].rstrip(" ,;:") + "..."


def _extract_organize_candidate_events(combined_events_content: str) -> List[Dict[str, str]]:
    if not combined_events_content:
        return []

    event_pattern = re.compile(r"^\s*-\s+\*\*\[(?P<title>.+?)\]\((?P<url>https?://[^)]+)\)\*\*\s*$")
    ignored_titles = {"submit event", "pricing", "create event"}
    ignored_paths = {"/signin", "/pricing", "/create", "/sf", "/genai-sf"}

    parsed_events: List[Dict[str, str]] = []
    current_event: Optional[Dict[str, str]] = None

    for raw_line in combined_events_content.splitlines():
        line = raw_line.rstrip()
        match = event_pattern.match(line)
        if match:
            if current_event:
                parsed_events.append(current_event)
            current_event = {
                "title": match.group("title").strip(),
                "rsvp_link": match.group("url").strip(),
                "date_time_raw": "",
                "location": "",
                "host": "",
                "description": "",
            }
            continue

        if not current_event:
            continue

        stripped = line.strip()
        if stripped.startswith("Date and Time:"):
            current_event["date_time_raw"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Location/Venue:"):
            current_event["location"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Host:"):
            current_event["host"] = stripped.split(":", 1)[1].strip()
        elif stripped.startswith("Brief Description:"):
            current_event["description"] = stripped.split(":", 1)[1].strip()

    if current_event:
        parsed_events.append(current_event)

    organized_events: List[Dict[str, str]] = []
    seen_urls = set()
    for event in parsed_events:
        title = event["title"].strip()
        rsvp_link = event["rsvp_link"].strip()
        if not title or not rsvp_link:
            continue

        canonical_url = _canonicalize_event_url(rsvp_link)
        parsed_url = urlparse(canonical_url)
        if title.lower() in ignored_titles or parsed_url.path.rstrip("/") in ignored_paths:
            continue

        parsed_dt = _parse_event_datetime_text(event["date_time_raw"])
        if not parsed_dt:
            continue

        dedupe_key = canonical_url or rsvp_link
        if dedupe_key in seen_urls:
            continue
        seen_urls.add(dedupe_key)

        location = _clean_location_for_display(event["location"])
        organized_events.append({
            "title": title,
            "rsvp_link": rsvp_link,
            "date_heading": _format_organized_day_heading(parsed_dt),
            "time_display": _format_organized_time(parsed_dt, event["date_time_raw"]),
            "location": location,
            "host": _ensure_text(event.get("host", "")).strip(),
            "suggested_region": _classify_event_region(location),
            "description": _shorten_event_description(event["description"]),
            "sort_key": parsed_dt.isoformat(),
        })

    organized_events.sort(key=lambda item: item["sort_key"])
    return organized_events


def _build_organized_events_fallback(events: List[Dict[str, str]]) -> str:
    if not events:
        return "No events with date, time, and RSVP link were available to organize."

    return _build_organized_events_text(events, use_markdown_links=True)


def _build_organized_event_sections(
    events: List[Dict[str, str]],
) -> Tuple[List[Tuple[str, List[Tuple[str, List[Dict[str, str]]]]]], bool]:
    grouped: Dict[str, List[Dict[str, str]]] = {}
    for event in events:
        day_heading = event["date_heading"]
        grouped.setdefault(day_heading, []).append(event)

    day_order = sorted(grouped.keys(), key=lambda heading: min(item["sort_key"] for item in grouped[heading]))
    ordered_sections: List[Tuple[str, List[Tuple[str, List[Dict[str, str]]]]]] = []
    for day_heading in day_order:
        ordered_sections.append(
            (day_heading, [("", sorted(grouped[day_heading], key=lambda item: item["sort_key"]))])
        )

    return ordered_sections, False


def _format_linkedin_event_line(
    event: Dict[str, str],
    *,
    use_markdown_links: bool,
    include_plain_rsvp_link: bool = False,
) -> str:
    if use_markdown_links:
        title_text = f"[{event['title']}]({event['rsvp_link']})"
    else:
        title_text = event["title"]

    line = f"{event['time_display']} - {title_text}"

    metadata_parts: List[str] = []
    location = _ensure_text(event.get("location", "")).strip()
    if location and location != "Location TBD":
        metadata_parts.append(location)

    host = _ensure_text(event.get("host", "")).strip()
    if host:
        metadata_parts.append(f"Host: {host}")

    description = _ensure_text(event.get("description", "")).strip()
    trailing_parts = metadata_parts[:]
    if description:
        trailing_parts.append(description)
    if trailing_parts:
        line = f"{line} - {' - '.join(trailing_parts)}"
    if include_plain_rsvp_link and not use_markdown_links:
        line = f"{line} - RSVP: {event['rsvp_link']}"
    return line


def _build_organized_events_text(
    events: List[Dict[str, str]],
    *,
    use_markdown_links: bool,
    include_plain_rsvp_link: bool = False,
) -> str:
    if not events:
        return ""

    ordered_sections, show_region_headings = _build_organized_event_sections(events)
    lines: List[str] = []
    for day_heading, ordered_regions in ordered_sections:
        if lines:
            lines.append("")
        lines.append(day_heading)
        lines.append("")

        for region, region_events in ordered_regions:
            if show_region_headings:
                lines.append(region)
                lines.append("")
            for event in region_events:
                lines.append(
                    _format_linkedin_event_line(
                        event,
                        use_markdown_links=use_markdown_links,
                        include_plain_rsvp_link=include_plain_rsvp_link,
                    )
                )
            lines.append("")

    return "\n".join(lines).strip()


def _build_organized_events_html(events: List[Dict[str, str]]) -> str:
    if not events:
        return ""

    ordered_sections, show_region_headings = _build_organized_event_sections(events)
    html_lines: List[str] = ["<div>"]
    first_day = True
    for day_heading, ordered_regions in ordered_sections:
        if not first_day:
            html_lines.append("<p><br></p>")
        first_day = False
        html_lines.append(f"<p><strong>{html_escape(day_heading)}</strong></p>")

        for region, region_events in ordered_regions:
            if show_region_headings:
                html_lines.append(f"<p><strong>{html_escape(region)}</strong></p>")
            html_lines.append("<ul>")
            for event in region_events:
                time_text = html_escape(_ensure_text(event.get("time_display", "")).strip())
                title_text = html_escape(_ensure_text(event.get("title", "")).strip())
                rsvp_link = html_escape(_ensure_text(event.get("rsvp_link", "")).strip(), quote=True)

                metadata_parts: List[str] = []
                location = _ensure_text(event.get("location", "")).strip()
                if location and location != "Location TBD":
                    metadata_parts.append(html_escape(location))

                host = _ensure_text(event.get("host", "")).strip()
                if host:
                    metadata_parts.append(f"Host: {html_escape(host)}")

                description = _ensure_text(event.get("description", "")).strip()
                if description:
                    metadata_parts.append(html_escape(description))

                line = f"{time_text} - <a href=\"{rsvp_link}\">{title_text}</a>"
                if metadata_parts:
                    line = f"{line} - {' - '.join(metadata_parts)}"
                html_lines.append(f"<li>{line}</li>")
            html_lines.append("</ul>")

    html_lines.append("</div>")
    return "\n".join(html_lines)


def _build_organized_events_preview_html(html_content: str) -> str:
    safe_html = html_content or "<p>No preview available.</p>"
    return f"""
        <style>
        body {{
            margin: 0;
            font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
            color: #1f2328;
            background: white;
        }}
        .linkedin-preview {{
            max-width: 720px;
            margin: 0 auto;
            padding: 0.25rem 0.75rem 0.75rem;
            font-size: 18px;
            line-height: 1.55;
        }}
        .linkedin-preview p {{
            margin: 0 0 0.9rem 0;
        }}
        .linkedin-preview ul {{
            margin: 0.15rem 0 1rem 1.4rem;
            padding-left: 1rem;
        }}
        .linkedin-preview li {{
            margin: 0 0 1rem 0;
        }}
        .linkedin-preview a {{
            color: #0a66c2;
            text-decoration: none;
            font-weight: 600;
        }}
        </style>
        <div class="linkedin-preview">{safe_html}</div>
    """


def _organized_events_cache_key(combined_events_content: str, use_gpt_polish: bool) -> str:
    payload = f"linkedin_v4::{int(use_gpt_polish)}::{combined_events_content}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def generate_organized_events(combined_events_content=None, use_gpt_polish: bool = False):
    """Organize combined event text into a grouped daily agenda."""
    try:
        if not combined_events_content:
            st.warning("No events data available. Please scrape events first.")
            return False, None

        cache_key = _organized_events_cache_key(combined_events_content, use_gpt_polish)
        cache = st.session_state.setdefault("organized_events_cache", {})
        cached_output = cache.get(cache_key)
        if cached_output:
            return True, cached_output

        events = _extract_organize_candidate_events(combined_events_content)
        if not events:
            st.warning("No events with RSVP links and parseable date/time were found.")
            return False, None

        fallback_output = _build_organized_events_fallback(events)
        if not use_gpt_polish:
            cache[cache_key] = fallback_output
            return True, fallback_output

        prompt = (
            "You are organizing upcoming AI events into a concise daily agenda.\n"
            "Use ONLY the supplied JSON records. Do not invent any events or details.\n"
            "Requirements:\n"
            "- Skip any item without an RSVP link.\n"
            "- Keep events sorted chronologically.\n"
            "- Group by day using the provided `date_heading` value.\n"
            "- Do not group by region, location, or online/offline status within a day.\n"
            "- Do not add headings like `Online`, `Bay Area`, `New York`, or `Other`.\n"
            "- For each event, output exactly one markdown line in this format:\n"
            "  `TIME - [TITLE](RSVP_URL) - LOCATION - Host: NAME short description`\n"
            "- Omit the `Host: NAME` segment when no host is provided.\n"
            "- Keep the short description to one sentence.\n"
            "- Preserve the RSVP link exactly.\n"
            "- Do not emit separate `RSVP:` or `About:` lines.\n"
            "- Do not add commentary or code fences.\n"
            "- Separate day sections with blank lines.\n\n"
            f"Event JSON:\n{json.dumps(events, ensure_ascii=False, indent=2)}"
        )

        organized_output = generate_with_gpt(
            prompt,
            temperature=0.1,
            max_tokens=5000,
            model_override=PREFERRED_GPT_MODEL,
        ).strip()

        if not organized_output or not re.search(r"\[[^\]]+\]\(https?://", organized_output):
            logger.warning("Organized events output invalid or empty; using fallback formatter.")
            cache[cache_key] = fallback_output
            return True, fallback_output

        cache[cache_key] = organized_output
        return True, organized_output
    except Exception as exc:
        logger.error("Organize events failed: %s", exc)
        fallback_events = _extract_organize_candidate_events(combined_events_content or "")
        if fallback_events:
            fallback_output = _build_organized_events_fallback(fallback_events)
            if use_gpt_polish:
                st.warning("LLM organize step failed, showing deterministic fallback instead.")
            cache = st.session_state.setdefault("organized_events_cache", {})
            cache[_organized_events_cache_key(combined_events_content or "", use_gpt_polish)] = fallback_output
            return True, fallback_output
        st.error(f"Error organizing events: {exc}")
        return False, None


def fix_example_com_urls(line, base_url=LUMA_BASE_URL):
    """Replace example.com URLs and relative URLs with proper base URLs"""
    import re

    # Determine base domain from context
    if 'cerebralvalley' in line.lower():
        base_url = "https://cerebralvalley.ai/"

    # Pattern 1: Markdown links with example.com
    example_pattern = r'\[([^\]]+)\]\s*\((https://example\.com/[^\)]+)\)'

    def replace_example_url(match):
        link_text = match.group(1)
        url = match.group(2)
        event_id = url.split('/')[-1].strip()
        # For lu.ma, use just the ID; for others, might need /events/ prefix
        if base_url == LUMA_BASE_URL:
            return f'[{link_text}]({base_url}/{event_id})'
        else:
            return f'[{link_text}]({base_url}/events/{event_id})'

    fixed_line = re.sub(example_pattern, replace_example_url, line)

    # Pattern 2: Markdown links with relative paths (starting with /)
    relative_pattern = r'\[([^\]]+)\]\s*\((/[^\)]+)\)'

    def replace_relative_url(match):
        link_text = match.group(1)
        path = match.group(2)
        # For lu.ma, remove leading slash; for others, keep the path structure
        if base_url == LUMA_BASE_URL:
            event_id = path.lstrip('/')
            return f'[{link_text}]({base_url}/{event_id})'
        else:
            return f'[{link_text}]({base_url}{path})'

    fixed_line = re.sub(relative_pattern, replace_relative_url, fixed_line)

    # Pattern 3: Plain example.com URLs
    plain_example = r'https://example\.com/([^\s\)]+)'
    if base_url == LUMA_BASE_URL:
        fixed_line = re.sub(plain_example, LUMA_BASE_URL + r'/\1', fixed_line)
    else:
        fixed_line = re.sub(plain_example, base_url + r'/events/\1', fixed_line)

    # Pattern 4: Plain relative paths in Link: lines
    if 'Link:' in fixed_line or '**Link:**' in fixed_line:
        plain_relative = r'(\*\*Link:\*\*|\bLink:)\s*(/[^\s]+)'
        if base_url == LUMA_BASE_URL:
            fixed_line = re.sub(plain_relative, r'\1 ' + LUMA_BASE_URL + r'\2', fixed_line)
        else:
            fixed_line = re.sub(plain_relative, r'\1 ' + base_url + r'\2', fixed_line)

    return fixed_line



def _auto_save_results(content: str, source_name: str) -> Optional[str]:
    """Save scraped results to a timestamped file in the scraped_results directory."""
    if not content or not content.strip():
        return None
    try:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scraped_results")
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        safe_name = re.sub(r"[^a-zA-Z0-9_-]", "_", source_name).strip("_").lower()
        filename = f"{safe_name}_{timestamp}.txt"
        filepath = os.path.join(save_dir, filename)
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)
        logger.info("Auto-saved scraped results to %s", filepath)
        return filepath
    except Exception as exc:
        logger.warning("Failed to auto-save results: %s", exc)
        return None


def _save_events_snapshot(payload: Dict[str, object]) -> Optional[str]:
    if not payload:
        return None
    try:
        save_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "scraped_results")
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filepath = os.path.join(save_dir, f"events_snapshot_{timestamp}.json")
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        logger.info("Saved events snapshot to %s", filepath)
        return filepath
    except Exception as exc:
        logger.warning("Failed to save events snapshot: %s", exc)
        return None


def main():
    st.title("AI Events Scraper")
    st.header("Automatically extract and display AI events")

    st.divider()
    days_to_scrape = int(st.session_state.get("days_to_scrape", 8))

    st.subheader("🎯 Event Sources")
    st.caption("Bay Area and New York are separate workspaces. Each region tab has independent source tabs for Lu.ma, Cerebral Valley, Evion, Meetup, and a combined view.")
    region_tab_bay, region_tab_ny = st.tabs(FOCUS_REGION_ORDER)

    with region_tab_bay:
        _render_region_tab_content("Bay Area", days_to_scrape)

    with region_tab_ny:
        _render_region_tab_content("New York", days_to_scrape)

    with st.expander("💾 Save Events", expanded=False):
        st.write("Save the current event workspace as a JSON snapshot that can be loaded later.")
        snapshot_payload = _build_events_snapshot_payload()
        if snapshot_payload:
            snapshot_text = json.dumps(snapshot_payload, ensure_ascii=False, indent=2)
            snapshot_filename = (
                f"events_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            )

            st.text_area(
                "Snapshot summary",
                value=_describe_snapshot_payload(snapshot_payload),
                height=100,
                key="events_snapshot_summary",
                label_visibility="visible",
                disabled=True,
            )

            col_save1, col_save2 = st.columns(2)
            with col_save1:
                st.download_button(
                    "Download Events Snapshot",
                    data=snapshot_text.encode("utf-8"),
                    file_name=snapshot_filename,
                    mime="application/json",
                    key="download_events_snapshot",
                    type="primary",
                    help="Download a JSON snapshot that can be reloaded later from the Load Events section.",
                )
            with col_save2:
                if st.button("Save Events to Workspace", key="save_events_snapshot_button", width="stretch"):
                    saved = _save_events_snapshot(snapshot_payload)
                    if saved:
                        st.success(f"✅ Events snapshot saved to `{os.path.basename(saved)}`")
                    else:
                        st.error("❌ Failed to save events snapshot")
        else:
            st.info("No event workspace state is available to save yet.")

    with st.expander("💽 Load Events", expanded=False):
        st.write("Upload a saved events snapshot or a plain text events file to skip scraping.")
        uploaded_events_file = st.file_uploader(
            "Load saved events file",
            type=["txt", "json"],
            key="upload_saved_events",
            help="Upload a file previously saved from this app to reuse event data."
        )
        if uploaded_events_file is not None:
            try:
                uploaded_content = uploaded_events_file.getvalue().decode("utf-8")
                if uploaded_content.strip():
                    loaded_snapshot = None
                    try:
                        parsed_json = json.loads(uploaded_content)
                        if _is_events_snapshot_payload(parsed_json):
                            loaded_snapshot = parsed_json
                    except json.JSONDecodeError:
                        loaded_snapshot = None

                    if loaded_snapshot:
                        restored = _restore_events_snapshot_payload(loaded_snapshot)
                        if restored:
                            st.session_state.loaded_events_source = uploaded_events_file.name or "Uploaded snapshot"
                            st.success("Events snapshot loaded successfully.")
                            st.text_area(
                                "Loaded snapshot summary",
                                value=_describe_snapshot_payload(loaded_snapshot),
                                height=100,
                                key="loaded_events_snapshot_summary",
                                label_visibility="visible",
                                disabled=True,
                            )
                        else:
                            st.error("Unable to restore the uploaded events snapshot.")
                    else:
                        _clear_event_workspace_state()
                        st.session_state.combined_events = uploaded_content
                        st.session_state.loaded_events_source = uploaded_events_file.name or "Uploaded text file"
                        st.success("Saved events text loaded successfully. You can generate an essay without scraping.")

                        with st.expander("Preview Loaded Events", expanded=False):
                            col_header1, col_header2 = st.columns([3, 1])
                            with col_header1:
                                st.markdown("**Loaded Events (Uploaded Preview)**")
                            with col_header2:
                                render_copy_button(uploaded_content, "loaded-preview")

                            st.markdown(uploaded_content)

                            st.text_area(
                                "📋 Preview (first 200 characters)",
                                value=(
                                    uploaded_content[:200]
                                    + ("\n\n... (open expander for full text) ..." if len(uploaded_content) > 200 else "")
                                ),
                                height=100,
                                key="loaded_events_preview",
                                label_visibility="visible",
                                disabled=True
                            )

                            with st.expander("✏️ View / edit full loaded text", expanded=False):
                                st.text_area(
                                    "Loaded events text",
                                    value=uploaded_content,
                                    height=300,
                                    key="loaded_events_text",
                                    label_visibility="collapsed"
                                )
                else:
                    st.warning("Uploaded file is empty.")
            except Exception as exc:
                st.error(f"Unable to read uploaded file: {exc}")

    with st.expander("🗂 Organize Events", expanded=False):
        st.write("Group scraped events by day and region with RSVP links and short descriptions.")
        organized_text = st.session_state.get("organized_events")
        selected_events_content = _get_selected_events_content("organize_region_select", "Event section")
        use_gpt_polish = st.checkbox(
            f"Use {PREFERRED_GPT_MODEL.upper()} polish (slower)",
            value=False,
            key="organize_events_use_gpt",
            help="Fast mode uses the deterministic formatter. Enable this only if you want GPT to rewrite the presentation.",
        )

        if selected_events_content:
            if st.button("Organize Events", key="organize_events_button", type="primary"):
                spinner_text = f"Organizing events with {PREFERRED_GPT_MODEL.upper()}..." if use_gpt_polish else "Organizing events..."
                with st.spinner(spinner_text):
                    success, result = generate_organized_events(
                        selected_events_content,
                        use_gpt_polish=use_gpt_polish,
                    )
                    if success and result:
                        organized_text = result
                        st.session_state.organized_events = result
                        st.session_state.organized_events_mode = (
                            f"{PREFERRED_GPT_MODEL.upper()} polish" if use_gpt_polish else "Fast deterministic"
                        )
                        st.success("✅ Events organized successfully!")
                        saved = _auto_save_results(result, "organized_events")
                        if saved:
                            st.info(f"💾 Results auto-saved to `{os.path.basename(saved)}`")
                    else:
                        st.error("❌ Failed to organize events")
        else:
            st.info("📋 Please scrape or load events first.")
            st.button("Organize Events", key="organize_events_button", disabled=True)

        if organized_text:
            organized_preview_events = _extract_organize_candidate_events(selected_events_content or "")
            organized_html = _build_organized_events_html(organized_preview_events)
            organized_plain_text = _build_organized_events_text(
                organized_preview_events,
                use_markdown_links=False,
                include_plain_rsvp_link=True,
            )
            st.session_state["organized_events_text"] = organized_text
            st.session_state["organized_events_html_source"] = organized_html

            col_header1, col_header2 = st.columns([3, 1])
            with col_header1:
                st.markdown("**Organized Events**")
                mode_label = st.session_state.get("organized_events_mode")
                if mode_label:
                    st.caption(mode_label)
            with col_header2:
                render_copy_button(
                    organized_plain_text or organized_text,
                    "organized-events-linkedin",
                    label="📋 Copy for LinkedIn",
                    html_content=organized_html or None,
                )

            st.markdown("**LinkedIn Preview**")
            if organized_html:
                st_components.html(
                    _build_organized_events_preview_html(organized_html),
                    height=560,
                    scrolling=True,
                )
                st.caption("This copy button uses rich HTML so pasting into LinkedIn keeps bullets and links when the browser allows it.")
            else:
                st.markdown(organized_text)

            with st.expander("💻 View LinkedIn HTML", expanded=False):
                st.text_area(
                    "LinkedIn HTML",
                    key="organized_events_html_source",
                    height=320,
                    label_visibility="visible",
                )

            with st.expander("✏️ View / edit raw organized text", expanded=False):
                st.text_area(
                    "Organized events text",
                    height=420,
                    key="organized_events_text",
                    label_visibility="visible",
                )

    with st.expander("📝 Essay Generation", expanded=False):
        st.write("Generate an essay based on the scraped events")
        button_essay = False
        selected_events_content = _get_selected_events_content("essay_region_select", "Event section")
        if selected_events_content:
            button_essay = st.button("Generate Essay from Scraped Events", key="essay_button")
            if button_essay:
                with st.spinner("Generating essay from scraped events..."):
                    success, essay_text = generate_essay(selected_events_content)
                    if success:
                        st.session_state.generated_essay = essay_text
                        st.success("✅ Essay generated successfully!")
                        st.markdown("**Essay based on scraped events:**")
                        st.markdown(essay_text)
                        st.text_area(
                            "Generated essay (copy or edit):",
                            value=essay_text,
                            height=400,
                            key="generated_essay_live",
                            label_visibility="visible"
                        )
                    else:
                        st.error("❌ Failed to generate essay")
        else:
            st.info("📋 Please scrape events first using 'Scrape All Sources' to generate an essay.")
            st.button("Generate Essay from Scraped Events", key="essay_button", disabled=True)

        if st.session_state.get('generated_essay') and not button_essay:
            st.markdown("**Essay based on scraped events:**")
            st.markdown(st.session_state.generated_essay)
            st.text_area(
                "Generated essay (copy or edit):",
                value=st.session_state.generated_essay,
                height=400,
                key="generated_essay_saved",
                label_visibility="visible"
            )

    with st.expander("🎨 Image Generation", expanded=False):
        st.write("Generate a promotional image based on the scraped events using Google's Nano Banana 2 model")

        selected_events_content = _get_selected_events_content("image_region_select", "Event section")
        if selected_events_content:
            col_img1, col_img2 = st.columns([1, 1])

            with col_img1:
                button_image = st.button("Generate Event Image", key="image_button", type="primary")

            with col_img2:
                if 'generated_image' in st.session_state:
                    st.success("✅ Image generated! Displayed below.")

            if button_image:
                with st.spinner("Generating image from events..."):
                    success, image_payload, error = generate_event_image(selected_events_content)

                    if success and image_payload:
                        st.session_state.generated_image = image_payload
                        st.success("✅ Image generated successfully!")
                        st.image(
                            image_payload["bytes"],
                            caption="AI Events Promotional Image",
                            width="stretch",
                        )
                        st.download_button(
                            "Download Image",
                            data=image_payload["bytes"],
                            file_name="ai_events_image.png",
                            mime=image_payload.get("mime_type", "image/png"),
                            help="Download the generated image",
                        )
                    else:
                        error_msg = error or "Failed to generate image"
                        st.error(f"❌ {error_msg}")
                        with st.expander("🔍 Error Details", expanded=False):
                            st.code(error_msg, language="text")
                            st.markdown("""
                            **Troubleshooting:**
                            - Make sure your Google API key is configured (GOOGLE_API_KEY)
                            - Check if you have sufficient API quota/credits
                            - Verify that Nano Banana 2 (Gemini image) is available for your account
                            - Check Google AI API status for outages
                        """)
                        st.info("💡 Tip: Ensure your Google API key has access to Nano Banana 2 image generation.")
        else:
            st.info("📋 Please scrape events first using 'Scrape All Sources' to generate an image.")
            st.button("Generate Event Image", key="image_button", disabled=True)
    
    # Display previously generated image if available
    if 'generated_image' in st.session_state and st.session_state.generated_image:
        st.divider()
        st.subheader("🖼️ Previously Generated Image")
        st.image(
            st.session_state.generated_image["bytes"], 
            caption="AI Events Promotional Image", 
            width="stretch"
        )

    st.divider()
    st.subheader("⚙️ Configuration")
    st.number_input(
        "Days to scrape",
        min_value=1,
        max_value=30,
        value=days_to_scrape,
        key="days_to_scrape",
        help="Number of days ahead to scrape events for"
    )


if __name__ == "__main__":
    main()
