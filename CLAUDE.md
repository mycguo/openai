# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is an AI Events Scraper and Essay Generator built with Streamlit. The application scrapes AI/GenAI events from multiple sources (Lu.ma and Cerebral Valley), displays them in a chronological format, and generates engaging essays based on the scraped content.

## Core Architecture

### Main Applications
- **`events.py`** - Primary Streamlit application with full event scraping and essay generation
- **`app.py`** - Legacy Google Docs integration app (generates LinkedIn posts and essays from Google Docs)

### Key Components

1. **Event Scraping Pipeline**
   - Uses `browser-use` library with Playwright for web scraping
   - Async scraping with `browser_use.agent.service.Agent`
   - Configurable headless browser sessions
   - Supports multiple event sources (Lu.ma, Cerebral Valley)

2. **Event Processing & Formatting**
   - `parse_and_format_combined_events()` - Parses raw scraped content into structured events
   - `parse_date_for_sorting()` - Handles various date formats (relative dates, month/day formats)
   - Chronological sorting and grouping by date
   - URL fixing for relative paths and example.com placeholders

3. **UI Structure (events.py)**
   - Individual source scraping (Lu.ma, Cerebral Valley)
   - Bulk scraping with tabbed results display
   - Combined results with chronological ordering
   - Essay generation based on scraped events
   - Session state management for data persistence

4. **OpenAI Integration**
   - Text generation via OpenAI API using `gpt-3.5-turbo`
   - Essay generation with customizable prompts
   - Temperature and max_tokens configuration

## Development Commands

### Running the Applications
```bash
# Run main events scraper application
streamlit run events.py

# Run legacy Google Docs application
streamlit run app.py

# Test event parsing functionality
python test_event_parser.py
```

### Environment Setup
```bash
# Install dependencies
pip install -r requirements.txt

# Install Playwright browsers (required for browser-use)
python -m playwright install chromium
```

## Configuration Requirements

### Streamlit Secrets
The application requires these secrets in `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "your-openai-api-key"

[gcp_service_account]
# Google service account JSON for Google Docs integration (legacy app.py)
```

### Browser-use Integration
- Uses `browser_use` library for AI-powered web scraping
- Configures headless Chromium sessions via Playwright
- Implements action limits (`max_actions=20`) to prevent infinite loops
- System prompts guide event extraction format

## Key Technical Patterns

### Async Event Scraping
```python
async def scrape_events(url, source_name):
    # Browser session setup
    browser_session = BrowserSession(browser_profile=BrowserProfile(...))

    # AI agent with task-specific prompts
    agent = Agent(task=task, llm=llm, browser_session=browser_session)

    # Execute and cleanup
    result = await agent.run()
    await browser_session.kill()
```

### Event Data Flow
1. Raw scraped content → `extract_events_from_agent_result()`
2. Structured parsing → `parse_events_from_content()`
3. Date parsing & sorting → `parse_date_for_sorting()`
4. Chronological formatting → `parse_and_format_combined_events()`
5. UI display with session state persistence

### URL Processing Pipeline
- Detects and fixes relative URLs (`/event-id` → `https://lu.ma/event-id`)
- Handles placeholder URLs (`example.com` → proper domain)
- Source-aware URL construction (Lu.ma vs Cerebral Valley)
- Markdown link parsing for `[text](url)` format

## Event Source Configuration

### Supported Sources
- **Lu.ma GenAI SF**: `https://lu.ma/genai-sf?k=c`
- **Cerebral Valley**: `https://cerebralvalley.ai/events`

### Adding New Sources
1. Update `scrape_events()` function with new URL
2. Add source detection in `parse_events_from_content()`
3. Configure URL fixing patterns in `fix_example_com_urls()`
4. Add UI button in main Streamlit interface

## Data Persistence

Uses Streamlit session state for cross-component data sharing:
- `st.session_state.combined_events` - Stores formatted combined events for essay generation
- Events persist within user session for essay generation workflow

## Testing

- **`test_event_parser.py`** - Standalone test for event parsing functionality
- Contains sample event data for both Lu.ma and Cerebral Valley formats
- Validates chronological sorting and field extraction