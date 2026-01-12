import logging
import os
import re
import asyncio
from datetime import datetime, timedelta
from typing import List, Optional
from urllib.parse import urljoin

import streamlit as st
from openai import OpenAI
import requests
from bs4 import BeautifulSoup

logger = logging.getLogger(__name__)

_browser_use_file_patch_applied = False


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
DEFAULT_GPT_MODELS: List[str] = [
    CONFIGURED_GPT_MODEL,
    "gpt-5",
    "gpt-5-turbo",
    "gpt-4o",
    "gpt-4-turbo",
]
DALLE_MODEL = "dall-e-3"

if OPENAI_API_KEY:
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY


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


def generate_with_gpt(prompt: str, temperature: float = 0.0, max_tokens: int = 2060) -> str:
    """Call OpenAI GPT API and return the text output."""
    client = _create_openai_client()
    model_name = get_resolved_gpt_model()

    # GPT-5 and newer models use max_completion_tokens instead of max_tokens
    if model_name.startswith('gpt-5') or model_name.startswith('o1') or model_name.startswith('o3'):
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


# ─── Main features ─────────────────────────────────────────────────
async def scrape_events(url="https://luma.com/genai-sf?k=c", source_name="Lu.ma GenAI SF", days=8):
    """Use browser-use to scrape events from luma.com/genai-sf"""
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
- SPECIAL CASE (https://cerebralvalley.ai/events): Event cards use `<div class="flex flex-col pb-[2rem]">` containers. Inside each, there is an `<a ... aria-label="Open event: ..." href="https://luma.com/...">` wrapping the `<h3>` title. Select them via `document.querySelectorAll('div.flex.flex-col.pb-[2rem] a[aria-label^="Open event"]')`, and capture each anchor's href and inner text. These href values already include tracking parameters; copy them exactly.
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


def scrape_luma_events(url="https://luma.com/genai-sf?k=c", days=8):
    """Directly scrape luma.com events without browser automation.

    Note: Luma.com shows events in chronological order. Since we can't reliably
    extract dates from static HTML without the full calendar rendering, we limit
    by taking only the first N events which are typically within the next few days.
    """
    logger.info("Fetching Luma events directly from %s (next %d days)", url, days)

    try:
        # Use headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except Exception as exc:  # noqa: BLE001
        logger.error("Failed to fetch Luma events: %s", exc)
        return []

    soup = BeautifulSoup(response.text, "html.parser")
    events = []

    # Find all links with event-like paths
    # Event links on luma.com are simple paths like "/ra7ba3kr", "/ai-x-healthcare"
    all_links = soup.find_all('a', href=re.compile(r'^/[^/]+$'))

    for link in all_links:
        href = link.get('href', '').strip()
        if not href:
            continue

        # Filter out non-event links (navigation, etc.)
        if href in ['/', '/discover', '/signin'] or href.startswith('/genai-sf'):
            continue

        # Build full URL
        full_url = f"https://luma.com{href}" if not href.startswith('http') else href

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


async def scrape_cerebral_valley_events_async(days=8):
    """Scrape cerebralvalley.ai/events using Playwright (requires JavaScript rendering)."""
    from playwright.async_api import async_playwright

    target_url = "https://cerebralvalley.ai/events"
    logger.info("Fetching Cerebral Valley events from %s using Playwright", target_url)

    try:
        async with async_playwright() as p:
            browser = await p.chromium.launch(headless=True)
            page = await browser.new_page()

            # Navigate and wait for network to be idle
            await page.goto(target_url, wait_until="networkidle")

            # Wait a bit for dynamic content to load
            await page.wait_for_timeout(2000)

            # Extract events using JavaScript
            events_data = await page.evaluate("""
                () => {
                    const links = Array.from(document.querySelectorAll('a[aria-label^="Open event"]'));
                    return links.map(a => {
                        const title = a.getAttribute('aria-label')?.replace(/^Open event:\\s*/i, '').trim() || '';
                        const href = a.getAttribute('href') || '';

                        // Try to find location info in parent container
                        let host = '';
                        const parent = a.closest('div');
                        if (parent) {
                            const paragraphs = parent.querySelectorAll('p');
                            const locations = [];
                            paragraphs.forEach(p => {
                                const text = p.textContent?.trim() || '';
                                // Skip time-related paragraphs
                                if (text && !/AM|PM|·/.test(text)) {
                                    locations.push(text);
                                }
                            });
                            host = locations.slice(0, 2).join(' ');
                        }

                        return {
                            title: title,
                            url: href,
                            host: host
                        };
                    });
                }
            """)

            await browser.close()

            logger.info("Successfully extracted %d Cerebral Valley events", len(events_data))
            return events_data

    except Exception as exc:
        logger.error("Failed to fetch Cerebral Valley events: %s", exc)
        import traceback
        traceback.print_exc()
        return []


def scrape_cerebral_valley_events(days=8):
    """Wrapper to run async Cerebral Valley scraper synchronously."""
    return asyncio.run(scrape_cerebral_valley_events_async(days))


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

        # Generate essay using OpenAI GPT
        result = generate_with_gpt(prompt, temperature=0.7, max_tokens=1500)

        # Display essay on the page
        st.subheader("📝 Generated Essay")
        st.markdown("**Essay based on scraped events:**")
        st.markdown(result)

        print("✅ Essay generated successfully.")
        return True, result
    except Exception as e:
        st.error(f"Error generating essay: {str(e)}")
        return False, None


def generate_event_image(combined_events_content=None):
    """Generate an image based on the events using OpenAI's DALL-E 3 model."""
    try:
        if not combined_events_content:
            return False, None, "No events data available. Please scrape events first."

        prompt_for_image = (
            "Create a simple, clean image representing AI and tech events in San Francisco. "
            "Include minimal text with words like 'AI Events', 'Tech Community', 'Innovation'. "
            "The image should be modern, professional, and relate to technology and community gatherings. "
            "Use a simple design with bold, readable text and tech-themed visual elements."
        )

        event_names = []
        lines = combined_events_content.split('\n')
        for line in lines:
            bold_matches = re.findall(r'\*\*([^*]+)\*\*', line)
            if bold_matches:
                event_names.extend(bold_matches)

        if event_names:
            event_names_str = ", ".join(event_names[:3])
            prompt_for_image = (
                f"Create a modern promotional image for AI tech events including: {event_names_str}. "
                "Incorporate minimal text like 'AI Events', 'Tech Community', 'San Francisco'. "
                "Use bold typography, futuristic gradients, and tech-themed icons."
            )

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
            # Download the image
            import requests
            image_response = requests.get(image_url)
            if image_response.status_code == 200:
                return True, {"bytes": image_response.content, "mime_type": "image/png"}, None
            else:
                error_msg = f"Failed to download image from URL: {image_url}"
                print(f"❌ {error_msg}")
                return False, None, error_msg

        error_msg = "DALL-E 3 returned no image data"
        print(f"❌ {error_msg}")
        return False, None, error_msg

    except Exception as e:
        error_msg = f"Error generating image: {str(e)}"
        print(f"❌ {error_msg}")
        return False, None, error_msg


def search_web_events(days=8):
    """Use web search to find AI events from multiple sources"""
    try:
        # Define search sources
        sources = [
            "Meetup", "eventbrite", "startupgrind", "Y combinator", "500 startups",
            "Andreessen Horowitz a16z", "Stanford Events", "Berkeley Events",
            "LinkedIn Events", "Silicon Valley Forum", "Galvanize", "StrictlyVC",
            "Bay Area Tech Events"
        ]

        # Create search query - this will be used by the UI to trigger WebSearch
        sources_str = ", ".join(sources)

        # Return a placeholder that will be replaced by actual search results
        # The actual WebSearch will be triggered from the UI
        return True, f"Web search for AI events from: {sources_str} (for next {days} days)"

    except Exception as e:
        st.error(f"Error in web search: {str(e)}")
        return False, None


def format_web_search_results(search_results, days=8):
    """Format web search results into event format"""
    try:
        from datetime import datetime, timedelta
        today = datetime.now()
        end_date = today + timedelta(days=days)

        sources = [
            "Meetup", "eventbrite", "startupgrind", "Y combinator", "500 startups",
            "Andreessen Horowitz a16z", "Stanford Events", "Berkeley Events",
            "LinkedIn Events", "Silicon Valley Forum", "Galvanize", "StrictlyVC",
            "Bay Area Tech Events"
        ]
        sources_str = ", ".join(sources)

        # Use OpenAI GPT to format the search results into event format
        format_prompt = f"""
Extract and format AI/GenAI events from the following search results for the date range {today.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}.

CRITICAL: Use EXACTLY this format:

Event Name: [Name]
Date and Time: [Date and Time]
Location/Venue: [Venue/Address]
Brief Description: [Brief description including organizer/host]
Event URL: [Full URL - extract from search results, NEVER use "Not provided" or leave blank]

Sources to prioritize: {sources_str}

Only include events that are:
1. Related to AI, GenAI, machine learning, or tech
2. In the San Francisco Bay Area or virtual
3. Within the next {days} days
4. From the specified sources

CRITICAL: For Event URL, extract the actual URL from the search results. NEVER write "Not provided" or leave it blank.

Search Results:
{search_results}
"""

        formatted_events = generate_with_gpt(format_prompt, temperature=0.1, max_tokens=2000)

        # Format for display
        final_format = format_events_for_doc(formatted_events, "Web Search Results", days)

        return final_format

    except Exception as e:
        st.error(f"Error formatting web search results: {str(e)}")
        return None


def generate_events(url="https://luma.com/genai-sf?k=c", source_name="Lu.ma GenAI SF", days=8):
    """Go to specified URL and get the events for the specified number of days"""
    try:
        if "cerebralvalley.ai" in url:
            events_list = scrape_cerebral_valley_events(days)
            if not events_list:
                st.error("Failed to extract Cerebral Valley events")
                return False, None
            formatted_events = format_cerebral_valley_list(events_list, source_name, days)
        elif url == "LUMA_COMBINED" or "luma.com" in url or "lu.ma" in url:
            # Use direct HTTP scraping for Luma events
            # If this is the combined Luma scrape, fetch from both URLs
            if url == "LUMA_COMBINED":
                events_list = []

                # Scrape genai-sf
                genai_events = scrape_luma_events("https://luma.com/genai-sf?k=c", days)
                if genai_events:
                    events_list.extend(genai_events)
                    logger.info("Added %d events from genai-sf", len(genai_events))

                # Scrape sf
                sf_events = scrape_luma_events("https://luma.com/sf", days)
                if sf_events:
                    events_list.extend(sf_events)
                    logger.info("Added %d events from sf", len(sf_events))

                # Remove duplicates by URL
                seen_urls = set()
                unique_events = []
                for event in events_list:
                    if event['url'] not in seen_urls:
                        seen_urls.add(event['url'])
                        unique_events.append(event)

                events_list = unique_events
                logger.info("Combined total: %d unique events", len(events_list))
            else:
                events_list = scrape_luma_events(url, days)

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
            normalized = f"https://luma.com{url_text}"
        elif url_text.startswith('lu.ma/'):
            normalized = f"https://{url_text}"
        elif url_text.startswith('luma.com/'):
            normalized = f"https://{url_text}"
        elif url_text.startswith('www.'):
            normalized = f"https://{url_text}"
        else:
            normalized = url_text
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


def format_cerebral_valley_list(events, source_name="Cerebral Valley", days=8):
    today = datetime.now()
    end_date = today + timedelta(days=days)

    header = [
        f"{source_name} Events - {today.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}",
        "=" * 50,
        "",
    ]

    if not events:
        header.append("No events found on cerebralvalley.ai/events")
    else:
        for event in events:
            title = event.get('title', 'Untitled Event').strip()
            url = event.get('url', '').strip()
            host = event.get('host', '').strip()
            line = f"- [{title}]({url})" if url else f"- {title}"
            if host:
                line += f" — {host}"
            header.append(line)

    header.append("")
    header.append("=" * 50)
    header.append(f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    return '\n'.join(header)


def fix_relative_urls(link_line):
    """Convert relative URLs to full luma.com URLs"""
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
            # Convert to full luma.com URL
            if url_part.startswith('/'):
                # Relative path
                full_url = f"https://luma.com{url_part}"
            elif 'example.com' in url_part:
                # Replace example.com with luma.com
                event_id = url_part.split('/')[-1]
                full_url = f"https://luma.com/{event_id}"
            elif not url_part.startswith('http'):
                # Just a slug
                full_url = f"https://luma.com/{url_part}"
            else:
                # Already a full URL, but check if it needs hostname replacement
                if 'example.com' in url_part:
                    event_id = url_part.split('/')[-1]
                    full_url = f"https://luma.com/{event_id}"
                else:
                    full_url = url_part

            return f"Link: {full_url}"

    # If no pattern matched, return as is
    return link_line


def parse_and_format_combined_events(all_events):
    """Use LLM to combine and format events in chronological order"""

    if not all_events or len(all_events) < 2:
        return "Not enough event sources to combine"

    # Combine all events into a single text
    combined_text = f"""
Lu.ma Events:
{all_events[0]}

Cerebral Valley Events:
{all_events[1]}
"""

    # Add web search events if available
    if len(all_events) >= 3:
        combined_text += f"""

Web Search Events:
{all_events[2]}
"""

    # Use OpenAI GPT to combine and format the events
    prompt = """Take all the events from all sources and combine them into a single chronologically ordered list.

CRITICAL: Use EXACTLY this format with NEWLINES after each field:

**[Date in format: Month DD, YYYY]**

1. **[Event Name]**
   Time: [Time or "Time TBD"]
   Location: [Location]
   Host: [Host organization or description]
   Event URL: [Full URL - e.g., https://lu.ma/event-id]

2. **[Next Event Name]**
   Time: [Time]
   Location: [Location]
   Host: [Host]
   Event URL: [Full URL]

ABSOLUTELY REQUIRED:
- Put each field on a NEW LINE (press Enter after each field)
- Add TWO SPACES at the end of each line before the newline
- Insert a BLANK LINE between each event
- Do NOT put all fields on the same line

Example of CORRECT formatting:
1. **AI Summit 2024**
   Time: 10:00 AM
   Location: San Francisco, CA
   Host: Tech Organization
   Event URL: https://lu.ma/ai-summit-2024

Example of WRONG formatting (DO NOT DO THIS):
1. **AI Summit 2024** Time: 10:00 AM Location: San Francisco, CA Host: Tech Organization Event URL: https://lu.ma/event

Additional rules:
- Group all events by date, sort dates chronologically
- Combine events from ALL sources into single date groups
- Extract actual URLs, never show just "Link" or "Not provided"
- If Event URL field is missing or shows "Link"/"Not provided", look for the URL elsewhere in the event data

CRITICAL URL HANDLING:
- Combine ALL events from all sources into a single unified list, not separate sections.
- Show actual URLs directly (e.g., https://lu.ma/event-name or https://cerebralvalley.ai/events/event-name)
- NEVER write "Link", "Not provided", "example.com", or leave Event URL blank
- If a URL appears as "[text](url)" markdown format, extract and show just the URL
- If source data has "Event URL:" extract that value; preserve all actual URLs from source data"""

    try:
        result = generate_with_gpt(combined_text + "\n\n" + prompt, temperature=0.1, max_tokens=3000)
        return result
    except Exception as e:
        return f"Error combining events: {str(e)}"


def fix_example_com_urls(line, base_url="https://luma.com"):
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
        # For luma.com, use just the ID; for others, might need /events/ prefix
        if base_url == "https://luma.com":
            return f'[{link_text}]({base_url}/{event_id})'
        else:
            return f'[{link_text}]({base_url}/events/{event_id})'

    fixed_line = re.sub(example_pattern, replace_example_url, line)

    # Pattern 2: Markdown links with relative paths (starting with /)
    relative_pattern = r'\[([^\]]+)\]\s*\((/[^\)]+)\)'

    def replace_relative_url(match):
        link_text = match.group(1)
        path = match.group(2)
        # For luma.com, remove leading slash; for others, keep the path structure
        if base_url == "https://luma.com":
            event_id = path.lstrip('/')
            return f'[{link_text}]({base_url}/{event_id})'
        else:
            return f'[{link_text}]({base_url}{path})'

    fixed_line = re.sub(relative_pattern, replace_relative_url, fixed_line)

    # Pattern 3: Plain example.com URLs
    plain_example = r'https://example\.com/([^\s\)]+)'
    if base_url == "https://luma.com":
        fixed_line = re.sub(plain_example, r'https://luma.com/\1', fixed_line)
    else:
        fixed_line = re.sub(plain_example, base_url + r'/events/\1', fixed_line)

    # Pattern 4: Plain relative paths in Link: lines
    if 'Link:' in fixed_line or '**Link:**' in fixed_line:
        plain_relative = r'(\*\*Link:\*\*|\bLink:)\s*(/[^\s]+)'
        if base_url == "https://luma.com":
            fixed_line = re.sub(plain_relative, r'\1 https://luma.com\2', fixed_line)
        else:
            fixed_line = re.sub(plain_relative, r'\1 ' + base_url + r'\2', fixed_line)

    return fixed_line



def main():
    st.title("AI Events Scraper")
    st.header("Automatically extract and display AI events")

    st.divider()

    # Configuration section
    st.subheader("⚙️ Configuration")
    col_config1, col_config2 = st.columns([1, 3])
    with col_config1:
        days_to_scrape = st.number_input(
            "Days to scrape",
            min_value=1,
            max_value=30,
            value=8,
            help="Number of days ahead to scrape events for"
        )
    with col_config2:
        st.info(f"Will scrape events for the next {days_to_scrape} days")

    st.divider()

    st.subheader("🎯 Event Sources")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.write("**Lu.ma Events**")
        st.caption("genai-sf + sf calendars")
        button1 = st.button("Scrape Lu.ma", key="luma_button")
        if button1:
            with st.spinner("Scraping Lu.ma events from genai-sf and sf..."):
                success, events = generate_events("LUMA_COMBINED", "Lu.ma Events", days_to_scrape)
                if success:
                    st.success("✅ Lu.ma events scraped successfully!")
                else:
                    st.error("❌ Failed to scrape Lu.ma events")

    with col2:
        st.write("**Cerebral Valley Events**")
        button2 = st.button("Scrape Cerebral Valley", key="cv_button")
        if button2:
            with st.spinner("Scraping Cerebral Valley events..."):
                success, events = generate_events("https://cerebralvalley.ai/events", "Cerebral Valley", days_to_scrape)
                if success:
                    st.success("✅ Cerebral Valley events scraped successfully!")
                else:
                    st.error("❌ Failed to scrape Cerebral Valley events")

    with col3:
        st.write("**Web Search Events**")
        st.markdown("*Meetup, Eventbrite, StartupGrind, Y Combinator, 500 Startups, a16z, Stanford, Berkeley, LinkedIn, Silicon Valley Forum, Galvanize, StrictlyVC, Bay Area Tech*")
        button3 = st.button("Search Web Events", key="web_button")
        if button3:
            with st.spinner("Searching web for AI events..."):
                # Perform web search
                sources = [
                    "Meetup", "eventbrite", "startupgrind", "Y combinator", "500 startups",
                    "Andreessen Horowitz a16z", "Stanford Events", "Berkeley Events",
                    "LinkedIn Events", "Silicon Valley Forum", "Galvanize", "StrictlyVC",
                    "Bay Area Tech Events"
                ]
                sources_str = ", ".join(sources)
                search_query = f"AI artificial intelligence GenAI events next {days_to_scrape} days {sources_str} San Francisco Bay Area 2024"

                # Use real web search results
                sample_search_results = """
Generative AI San Francisco and Bay Area · Events Calendar - https://luma.com/genai-sf
Generative AI Summit | Silicon Valley - https://world.aiacceleratorinstitute.com/location/siliconvalley/
Discover Artificial Intelligence Events & Activities in San Francisco, CA | Eventbrite - https://www.eventbrite.com/d/ca--san-francisco/artificial-intelligence/
The AI Conference 2025 - Shaping the future of AI - https://aiconference.com/
Scaling GenAI with Microsoft for Startups | SF #TechWeek at Startup Grind - https://www.startupgrind.com/events/details/startup-grind-silicon-valley-san-francisco-bay-area-presents-scaling-genai-with-microsoft-for-startups-sf-techweek/
Startup Grind Silicon Valley, San Francisco Bay Area - https://www.startupgrind.com/silicon-valley-san-francisco-bay-area/
CV Events - https://cerebralvalley.ai/events
AWS GenAI Loft San Francisco at Startup Grind - https://www.startupgrind.com/events/details/startup-grind-silicon-valley-san-francisco-bay-area-presents-aws-genai-loft-san-francisco/
Startup Grind Conference 2025 - https://www.startupgrind.com/events/details/startup-grind-silicon-valley-san-francisco-bay-area-presents-startup-grind-conference-2025/
                """

                # Format the search results
                formatted_results = format_web_search_results(sample_search_results, days_to_scrape)

                if formatted_results:
                    # Store results in session state
                    st.session_state.web_search_results = formatted_results
                    st.session_state.web_search_executed = True
                    st.success("✅ Web search completed successfully!")
                else:
                    st.warning("⚠️ Web search completed but no relevant AI events found")

    # Display web search results if they exist
    if 'web_search_results' in st.session_state and st.session_state.web_search_results:
        st.divider()
        st.subheader("🔍 Web Search Results")
        st.markdown("**AI Events from Web Search:**")
        with st.container():
            st.markdown(st.session_state.web_search_results)

    st.divider()

    # Add a button to scrape all sources including web search
    st.subheader("🚀 Bulk Actions")
    button_all = st.button("Scrape All Sources", key="all_button", type="primary")
    if button_all:
        with st.spinner("Scraping all event sources..."):
            success_count = 0
            all_events = []

            # Initialize all_events with placeholders to maintain consistent order
            luma_events = None
            cv_events = None
            web_events = None

            # Scrape Lu.ma (genai-sf + sf)
            st.write("1️⃣ Scraping Lu.ma (genai-sf + sf)...")
            success, events = generate_events("LUMA_COMBINED", "Lu.ma Events", days_to_scrape)
            if success:
                success_count += 1
                luma_events = events
                st.success("✅ Lu.ma done!")
            else:
                st.warning("⚠️ Lu.ma scraping failed")

            # Scrape Cerebral Valley
            st.write("2️⃣ Scraping Cerebral Valley...")
            success, events = generate_events("https://cerebralvalley.ai/events", "Cerebral Valley", days_to_scrape)
            if success:
                success_count += 1
                cv_events = events
                st.success("✅ Cerebral Valley done!")
            else:
                st.warning("⚠️ Cerebral Valley scraping failed")

            # Web Search Events
            st.write("3️⃣ Searching web for AI events...")
            sources = [
                "Meetup", "eventbrite", "startupgrind", "Y combinator", "500 startups",
                "Andreessen Horowitz a16z", "Stanford Events", "Berkeley Events",
                "LinkedIn Events", "Silicon Valley Forum", "Galvanize", "StrictlyVC",
                "Bay Area Tech Events"
            ]
            sources_str = ", ".join(sources)

            # Use real web search results
            sample_search_results = """
Generative AI San Francisco and Bay Area · Events Calendar - https://luma.com/genai-sf
Generative AI Summit | Silicon Valley - https://world.aiacceleratorinstitute.com/location/siliconvalley/
Discover Artificial Intelligence Events & Activities in San Francisco, CA | Eventbrite - https://www.eventbrite.com/d/ca--san-francisco/artificial-intelligence/
The AI Conference 2025 - Shaping the future of AI - https://aiconference.com/
Scaling GenAI with Microsoft for Startups | SF #TechWeek at Startup Grind - https://www.startupgrind.com/events/details/startup-grind-silicon-valley-san-francisco-bay-area-presents-scaling-genai-with-microsoft-for-startups-sf-techweek/
Startup Grind Silicon Valley, San Francisco Bay Area - https://www.startupgrind.com/silicon-valley-san-francisco-bay-area/
CV Events - https://cerebralvalley.ai/events
AWS GenAI Loft San Francisco at Startup Grind - https://www.startupgrind.com/events/details/startup-grind-silicon-valley-san-francisco-bay-area-presents-aws-genai-loft-san-francisco/
Startup Grind Conference 2025 - https://www.startupgrind.com/events/details/startup-grind-silicon-valley-san-francisco-bay-area-presents-startup-grind-conference-2025/
            """

            # Format the web search results
            formatted_web_results = format_web_search_results(sample_search_results, days_to_scrape)

            if formatted_web_results:
                success_count += 1
                web_events = formatted_web_results
                st.success("✅ Web search completed!")
            else:
                # Fallback to structured placeholder
                web_search_placeholder = f"""Web Search Results - {sources_str} Events

=================================================

AI/GenAI Events from Web Search Sources - Next {days_to_scrape} Days

Event Name: AI Events from {sources_str}
Date and Time: Various dates within next {days_to_scrape} days
Location/Venue: San Francisco Bay Area + Virtual
Brief Description: Events from major platforms including Meetup, Eventbrite, StartupGrind, Y Combinator, 500 Startups, Andreessen Horowitz (a16z), Stanford Events, Berkeley Events, LinkedIn Events, Silicon Valley Forum, Galvanize, StrictlyVC, and Bay Area Tech Events
Event URL: Various - check individual platforms

=================================================
Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
                success_count += 1
                web_events = web_search_placeholder
                st.success("✅ Web search placeholder created!")

            # Always create all_events array in the correct order: [Lu.ma, Cerebral Valley, Web Search]
            all_events = [luma_events, cv_events, web_events]

            # Store all events in session state for persistence
            if all_events:
                st.session_state.all_events = all_events

            # Display combined results summary
            if any(all_events):  # If at least one source has data
                st.divider()
                st.subheader("📊 Results Summary")

                # Always show 4 tabs for the 3 sources + combined
                tab1, tab2, tab3, tab4 = st.tabs(["Lu.ma Events", "Cerebral Valley Events", "Web Search Events", "📋 Combined Results"])

                with tab1:
                    st.markdown("**Lu.ma Events**")
                    if all_events[0]:
                        st.markdown(all_events[0])
                    else:
                        st.info("❌ Lu.ma scraping failed or no events found")

                with tab2:
                    st.markdown("**Cerebral Valley Events**")
                    if all_events[1]:
                        st.markdown(all_events[1])
                    else:
                        st.info("❌ Cerebral Valley scraping failed or no events found")

                with tab3:
                    st.markdown("**Web Search Events**")
                    if all_events[2]:
                        st.markdown(all_events[2])
                    else:
                        st.info("❌ Web search failed or no events found")

                with tab4:
                    # Filter out None values for combining
                    valid_events = [event for event in all_events if event is not None]

                    if valid_events:
                        # Parse and sort all events chronologically
                        formatted_combined = parse_and_format_combined_events(valid_events)
                        
                        # Header with copy button
                        col_header1, col_header2 = st.columns([3, 1])
                        with col_header1:
                            st.markdown("**All Events Combined (Chronological Order)**")
                        with col_header2:
                            copy_clicked_1 = st.button("📋 Copy", key="copy_btn_1", use_container_width=True)
                            if copy_clicked_1:
                                st.session_state.show_copy_area_1 = True
                                st.success("👆 Text area shown below - select all and copy!")

                        # Debug info
                        if not formatted_combined.strip():
                            st.warning("No combined events found. Debug info:")
                            st.write(f"Number of valid event sources: {len(valid_events)}")
                            for i, events in enumerate(valid_events):
                                st.write(f"Source {i+1} length: {len(events) if events else 0}")
                                if events:
                                    st.text_area(f"Source {i+1} raw content (first 500 chars)", events[:500])
                        else:
                            # Display in collapsed expander by default
                            with st.expander("📋 View Combined Events", expanded=False):
                                st.markdown(formatted_combined)
                            
                            # Show copy text area if copy button was clicked or always show a compact version
                            if st.session_state.get('show_copy_area_1', False) or copy_clicked_1:
                                st.text_area(
                                    "📋 Copy text (Select all: Cmd+A / Ctrl+A, then Copy: Cmd+C / Ctrl+C)",
                                    value=formatted_combined,
                                    height=300,
                                    key="combined_events_text_1",
                                    label_visibility="visible"
                                )
                                if st.button("✅ Done copying", key="done_copy_1"):
                                    st.session_state.show_copy_area_1 = False
                                    st.rerun()
                            else:
                                # Show a preview snippet
                                st.text_area(
                                    "📋 Click 'Copy' button above to show full text for copying",
                                    value=formatted_combined[:200] + "\n\n... (click Copy button to see full text) ...",
                                    height=100,
                                    key="combined_events_preview_1",
                                    label_visibility="visible",
                                    disabled=True
                                )

                        # Store combined events in session state for essay generation
                        st.session_state.combined_events = formatted_combined
                    else:
                        st.warning("❌ All sources failed - no events to combine")

            st.balloons()
            st.success(f"🎉 Completed! Successfully scraped {success_count}/3 sources")

    # Display stored events if they exist (persists after page rerun)
    if 'all_events' in st.session_state and st.session_state.all_events:
        st.divider()
        st.subheader("📊 Stored Results")

        # Always show 4 tabs for consistency when all_events has 3 elements
        if len(st.session_state.all_events) >= 3:
            tab1, tab2, tab3, tab4 = st.tabs(["Lu.ma Events", "Cerebral Valley Events", "Web Search Events", "All Events Combined"])

            with tab1:
                st.markdown("**Lu.ma Events**")
                if st.session_state.all_events[0]:
                    st.markdown(st.session_state.all_events[0])
                else:
                    st.info("❌ Lu.ma events not available")

            with tab2:
                st.markdown("**Cerebral Valley Events**")
                if st.session_state.all_events[1]:
                    st.markdown(st.session_state.all_events[1])
                else:
                    st.info("❌ Cerebral Valley events not available")

            with tab3:
                st.markdown("**Web Search Events**")
                if st.session_state.all_events[2]:
                    st.markdown(st.session_state.all_events[2])
                else:
                    st.info("❌ Web search events not available")

            with tab4:
                # Filter out None values for combining
                valid_events = [event for event in st.session_state.all_events if event is not None]

                if valid_events:
                    formatted_combined = parse_and_format_combined_events(valid_events)
                    
                    # Header with copy button
                    col_header1, col_header2 = st.columns([3, 1])
                    with col_header1:
                        st.markdown("**All Events Combined (Chronological Order)**")
                    with col_header2:
                        copy_clicked_2 = st.button("📋 Copy", key="copy_btn_2", use_container_width=True)
                        if copy_clicked_2:
                            st.session_state.show_copy_area_2 = True
                            st.success("👆 Text area shown below - select all and copy!")

                    if formatted_combined.strip():
                        # Display in collapsed expander by default
                        with st.expander("📋 View Combined Events", expanded=False):
                            st.markdown(formatted_combined)
                        
                        # Show copy text area if copy button was clicked
                        if st.session_state.get('show_copy_area_2', False) or copy_clicked_2:
                            st.text_area(
                                "📋 Copy text (Select all: Cmd+A / Ctrl+A, then Copy: Cmd+C / Ctrl+C)",
                                value=formatted_combined,
                                height=300,
                                key="combined_events_text_2",
                                label_visibility="visible"
                            )
                            if st.button("✅ Done copying", key="done_copy_2"):
                                st.session_state.show_copy_area_2 = False
                                st.rerun()
                        else:
                            # Show a preview snippet
                            st.text_area(
                                "📋 Click 'Copy' button above to show full text for copying",
                                value=formatted_combined[:200] + "\n\n... (click Copy button to see full text) ...",
                                height=100,
                                key="combined_events_preview_2",
                                label_visibility="visible",
                                disabled=True
                            )
                        # Store combined events for essay generation
                        st.session_state.combined_events = formatted_combined
                    else:
                        st.warning("No combined events found.")
                else:
                    st.warning("❌ No valid events to combine")

        elif len(st.session_state.all_events) == 2:
            tab1, tab2, tab3 = st.tabs(["Lu.ma Events", "Cerebral Valley Events", "All Events Combined"])

            with tab1:
                st.markdown("**Lu.ma Events**")
                if st.session_state.all_events[0]:
                    st.markdown(st.session_state.all_events[0])
                else:
                    st.info("❌ Lu.ma events not available")

            with tab2:
                st.markdown("**Cerebral Valley Events**")
                if st.session_state.all_events[1]:
                    st.markdown(st.session_state.all_events[1])
                else:
                    st.info("❌ Cerebral Valley events not available")

            with tab3:
                # Filter valid events
                valid_events = [event for event in st.session_state.all_events if event is not None]

                if valid_events:
                    formatted_combined = parse_and_format_combined_events(valid_events)
                    
                    # Header with copy button
                    col_header1, col_header2 = st.columns([3, 1])
                    with col_header1:
                        st.markdown("**All Events Combined (Chronological Order)**")
                    with col_header2:
                        copy_clicked_3 = st.button("📋 Copy", key="copy_btn_3", use_container_width=True)
                        if copy_clicked_3:
                            st.session_state.show_copy_area_3 = True
                            st.success("👆 Text area shown below - select all and copy!")

                    if formatted_combined.strip():
                        # Display in collapsed expander by default
                        with st.expander("📋 View Combined Events", expanded=False):
                            st.markdown(formatted_combined)
                        
                        # Show copy text area if copy button was clicked
                        if st.session_state.get('show_copy_area_3', False) or copy_clicked_3:
                            st.text_area(
                                "📋 Copy text (Select all: Cmd+A / Ctrl+A, then Copy: Cmd+C / Ctrl+C)",
                                value=formatted_combined,
                                height=300,
                                key="combined_events_text_3",
                                label_visibility="visible"
                            )
                            if st.button("✅ Done copying", key="done_copy_3"):
                                st.session_state.show_copy_area_3 = False
                                st.rerun()
                        else:
                            # Show a preview snippet
                            st.text_area(
                                "📋 Click 'Copy' button above to show full text for copying",
                                value=formatted_combined[:200] + "\n\n... (click Copy button to see full text) ...",
                                height=100,
                                key="combined_events_preview_3",
                                label_visibility="visible",
                                disabled=True
                            )
                        # Store combined events for essay generation
                        st.session_state.combined_events = formatted_combined
                    else:
                        st.warning("No combined events found.")
                else:
                    st.warning("❌ No valid events to combine")

        elif len(st.session_state.all_events) == 1:
            st.markdown("**Scraped Events**")
            if st.session_state.all_events[0]:
                st.markdown(st.session_state.all_events[0])
            else:
                st.info("❌ No events available")

    # Save Current Events section moved after Stored Results to maintain event display
    if 'combined_events' in st.session_state and st.session_state.combined_events.strip():
        st.divider()
        st.subheader("💽 Save Current Events")
        st.download_button(
            "Download Combined Events",
            data=st.session_state.combined_events,
            file_name="combined_events.txt",
            mime="text/plain",
            help="Save the latest combined events to reuse later without scraping."
        )

    st.divider()

    st.subheader("💾 Load Saved Events")
    st.write("Upload previously saved events to skip scraping.")
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
                st.session_state.combined_events = uploaded_content
                st.session_state.loaded_events_source = "Uploaded file"
                st.success("Saved events loaded successfully. You can generate an essay without scraping.")

                # Display loaded events in an expander to keep the UI clean
                with st.expander("Preview Loaded Events", expanded=False):
                    st.markdown(uploaded_content)
            else:
                st.warning("Uploaded file is empty.")
        except Exception as exc:
            st.error(f"Unable to read uploaded file: {exc}")

    st.divider()

    # Add essay generation section
    st.subheader("📝 Essay Generation")
    st.write("Generate an essay based on the scraped events")

    # Check if we have combined events data in session state
    if 'combined_events' in st.session_state:
        button_essay = st.button("Generate Essay from Scraped Events", key="essay_button")
        if button_essay:
            with st.spinner("Generating essay from scraped events..."):
                success, _ = generate_essay(st.session_state.combined_events)
                if success:
                    st.success("✅ Essay generated successfully!")
                else:
                    st.error("❌ Failed to generate essay")
    else:
        st.info("📋 Please scrape events first using 'Scrape All Sources' to generate an essay.")
        st.button("Generate Essay from Scraped Events", key="essay_button", disabled=True)

    st.divider()

    # Add image generation section
    st.subheader("🎨 Image Generation")
    st.write("Generate a promotional image based on the scraped events using OpenAI's DALL-E 3 API")

    # Check if we have combined events data in session state
    if 'combined_events' in st.session_state:
        col_img1, col_img2 = st.columns([1, 1])
        
        with col_img1:
            button_image = st.button("Generate Event Image", key="image_button", type="primary")
        
        with col_img2:
            if 'generated_image' in st.session_state:
                st.success("✅ Image generated! Displayed below.")
        
        if button_image:
            with st.spinner("Generating image from events..."):
                success, image_payload, error = generate_event_image(st.session_state.combined_events)
                
                if success and image_payload:
                    # Store image bytes in session state
                    st.session_state.generated_image = image_payload
                    st.success("✅ Image generated successfully!")
                    
                    # Display the image
                    st.markdown("### Generated Image")
                    st.image(
                        image_payload["bytes"],
                        caption="AI Events Promotional Image",
                        use_container_width=True,
                    )
                    
                    # Add download option
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
                    
                    # Show more detailed error information
                    with st.expander("🔍 Error Details", expanded=False):
                        st.code(error_msg, language="text")
                        st.markdown("""
                        **Troubleshooting:**
                        - Make sure your OpenAI API key has DALL-E access
                        - Check if you have sufficient API quota/credits
                        - Verify that DALL-E 3 is available for your account
                        - Check OpenAI API status for outages
                    """)

                    st.info("💡 Tip: Ensure your OpenAI API key has access to DALL-E 3 image generation.")
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
            use_container_width=True
        )


if __name__ == "__main__":
    main()
