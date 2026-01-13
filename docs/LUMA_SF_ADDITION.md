# Added https://lu.ma/sf to Luma Event Sources

## Summary
Added a second Lu.ma calendar (https://lu.ma/sf) to the existing scraper while keeping the 3-source structure (Lu.ma, Cerebral Valley, Web Search).

## Implementation

### How It Works
The Lu.ma scraper now fetches events from **two calendars** and combines them:
1. `https://lu.ma/genai-sf?k=c` (original GenAI SF calendar)
2. `https://lu.ma/sf` (new SF calendar)

Events from both sources are merged and deduplicated by URL, so the same event won't appear twice.

### Code Changes

#### Modified: `generate_events()` function (events.py:695-719)

Added special handling for `url == "LUMA_COMBINED"`:
```python
if url == "LUMA_COMBINED":
    events_list = []

    # Scrape genai-sf
    genai_events = scrape_luma_events("https://lu.ma/genai-sf?k=c", days)
    if genai_events:
        events_list.extend(genai_events)

    # Scrape sf
    sf_events = scrape_luma_events("https://lu.ma/sf", days)
    if sf_events:
        events_list.extend(sf_events)

    # Remove duplicates by URL
    seen_urls = set()
    unique_events = []
    for event in events_list:
        if event['url'] not in seen_urls:
            seen_urls.add(event['url'])
            unique_events.append(event)

    events_list = unique_events
```

#### Updated: UI Button (events.py:1180-1189)
Changed from "Lu.ma GenAI SF" to "Lu.ma Events" with caption indicating both sources:
```python
st.write("**Lu.ma Events**")
st.caption("genai-sf + sf calendars")
button1 = st.button("Scrape Lu.ma", key="luma_button")
if button1:
    with st.spinner("Scraping Lu.ma events from genai-sf and sf..."):
        success, events = generate_events("LUMA_COMBINED", "Lu.ma Events", days_to_scrape)
```

#### Updated: Bulk Scraping (events.py:1266-1273)
Changed to use "LUMA_COMBINED" trigger:
```python
st.write("1️⃣ Scraping Lu.ma (genai-sf + sf)...")
success, events = generate_events("LUMA_COMBINED", "Lu.ma Events", days_to_scrape)
```

## User-Facing Changes

### Before
- **Lu.ma GenAI SF Events** - Only scraped genai-sf calendar
- Button: "Scrape Lu.ma GenAI SF"

### After
- **Lu.ma Events** (genai-sf + sf calendars)
- Button: "Scrape Lu.ma"
- Now fetches events from both calendars automatically
- Deduplicates any events that appear on both calendars

## Benefits

1. **More comprehensive coverage** - Captures events from both SF calendars
2. **Still 3 sources** - Maintains the existing UI structure
3. **Automatic deduplication** - Same event won't appear twice
4. **Same performance** - Both are direct HTTP scrapes (fast)

## Testing

To test:
```bash
streamlit run events.py
```

Click "Scrape Lu.ma" and verify:
- Events from both genai-sf and sf appear
- No duplicate events
- All URLs are valid (https://lu.ma/...)

## Files Modified
- `events.py` - Updated generate_events() and UI sections
- `LUMA_SF_ADDITION.md` - This documentation (NEW)
