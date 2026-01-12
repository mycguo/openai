#!/usr/bin/env python3
"""Test script for Luma event scraping."""

import logging
import re
from urllib.parse import urljoin

import requests
from bs4 import BeautifulSoup

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def scrape_luma_events(url="https://luma.com/genai-sf?k=c", days=8):
    """Directly scrape luma.com events without browser automation."""
    logger.info("Fetching Luma events directly from %s", url)

    try:
        # Use headers to mimic a real browser
        headers = {
            'User-Agent': 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36',
            'Accept': 'text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8',
            'Accept-Language': 'en-US,en;q=0.9',
        }
        response = requests.get(url, headers=headers, timeout=30)
        response.raise_for_status()
    except Exception as exc:
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
            'host': '',
        })

    # Remove duplicates (same URL)
    seen_urls = set()
    unique_events = []
    for event in events:
        if event['url'] not in seen_urls:
            seen_urls.add(event['url'])
            unique_events.append(event)

    logger.info("Successfully extracted %d unique Luma events", len(unique_events))
    return unique_events


if __name__ == "__main__":
    print("Testing Luma event scraping...")
    print("=" * 80)

    events = scrape_luma_events()

    print(f"\n✅ Found {len(events)} events:\n")

    for i, event in enumerate(events, 1):
        print(f"{i}. {event['title']}")
        print(f"   URL: {event['url']}")
        print()

    if len(events) == 0:
        print("❌ No events found - something went wrong!")
    else:
        print(f"✅ Successfully extracted {len(events)} events with URLs!")
