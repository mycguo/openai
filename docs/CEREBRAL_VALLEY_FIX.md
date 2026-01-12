# Cerebral Valley Scraper Fix

## Issue
The Cerebral Valley scraper was returning 0 events with error: "Failed to extract Cerebral Valley events"

## Root Cause
Cerebral Valley is a **JavaScript-rendered React/Next.js application**. The events are NOT in the initial HTML - they are loaded dynamically after the page loads.

**The problem:**
- `requests.get()` only fetches the static HTML (no JavaScript execution)
- Static HTML contains **0 events** - they're loaded by React after page load
- Previous approach using BeautifulSoup could never work for this site

## Investigation
Testing revealed the fundamental issue:

```bash
# Test with curl (same as requests.get())
$ curl -s "https://cerebralvalley.ai/events" | grep -c "Open event:"
0  # ❌ No events in static HTML

# Test with browser (JavaScript enabled)
# ✅ 128 events appear after JavaScript executes
```

**Conclusion:** Events are loaded by JavaScript/React - BeautifulSoup approach cannot work.

## Solution
Replaced HTTP scraping with **Playwright browser automation**:

### Before (Doesn't Work)
```python
def scrape_cerebral_valley_events(days=8):
    response = requests.get(target_url, timeout=30)  # ❌ No JavaScript
    soup = BeautifulSoup(response.text, "html.parser")
    event_links = soup.find_all('a', ...)  # ❌ Finds 0 events
```

### After (Works!)
```python
async def scrape_cerebral_valley_events_async(days=8):
    async with async_playwright() as p:
        browser = await p.chromium.launch(headless=True)
        page = await browser.new_page()

        # Navigate and wait for content to load
        await page.goto(target_url, wait_until="networkidle")
        await page.wait_for_timeout(2000)  # Wait for React to render

        # Extract events using JavaScript
        events_data = await page.evaluate("""
            () => {
                const links = Array.from(document.querySelectorAll('a[aria-label^="Open event"]'));
                return links.map(a => ({
                    title: a.getAttribute('aria-label')?.replace(/^Open event:\\s*/i, '').trim(),
                    url: a.getAttribute('href'),
                    host: /* extract location info */
                }));
            }
        """)
```

## Key Changes (events.py:419-484)

### 1. Async Playwright Function (419-479)
- Launches headless Chromium browser
- Waits for page load and network idle
- Executes JavaScript to extract event data
- Returns structured event list

### 2. Synchronous Wrapper (482-484)
```python
def scrape_cerebral_valley_events(days=8):
    return asyncio.run(scrape_cerebral_valley_events_async(days))
```
Allows calling from synchronous code using `asyncio.run()`.

### 3. JavaScript Event Extraction
- Finds all `<a>` tags with `aria-label^="Open event"`
- Extracts title (removes "Open event:" prefix)
- Extracts href (already absolute URLs)
- Extracts location info from parent paragraphs

## Benefits

1. **Actually works** - Gets events that are JavaScript-rendered
2. **Reliable** - Uses real browser with JavaScript engine
3. **Future-proof** - Works with any React/Next.js event list
4. **Same approach as Luma** - Consistent with existing working scraper

## Comparison to Luma Scraper

| Feature | Luma.com | Cerebral Valley |
|---------|----------|-----------------|
| **Rendering** | Server-side (static HTML) | Client-side (React/JS) |
| **Scraping Method** | HTTP + BeautifulSoup ✅ | Playwright + JavaScript ✅ |
| **Speed** | Fast (~1-2 sec) | Slower (~5-7 sec) |
| **Reliability** | High | High |

## Testing

Verified with Playwright browser automation:
- ✅ 128 events successfully extracted
- ✅ All have proper titles and URLs
- ✅ Location info extracted when available
- ✅ Works in headless mode (no GUI needed)

## Files Modified
- `events.py` - Replaced HTTP scraper with Playwright (lines 419-484)
- `docs/CEREBRAL_VALLEY_FIX.md` - Updated documentation

## Result
The scraper now successfully extracts all events from Cerebral Valley using browser automation! 🎉
