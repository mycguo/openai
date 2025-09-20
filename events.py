import os
import re
from openai import OpenAI
from googleapiclient.discovery import build
from google.oauth2 import service_account
import streamlit as st
from openai import AsyncOpenAI
import asyncio
from datetime import datetime, timedelta


# ─── Configuration ────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MODEL = "gpt-3.5-turbo"
SERVICE_ACCOUNT_FILE = "./service-account.json"
DOCUMENT_ID = "1vbvbDxvKj6LTWKiahK79XTHZsrhfeZpLUfqf1Ocl6RE"

SCOPES = [
  "https://www.googleapis.com/auth/documents",
  "https://www.googleapis.com/auth/drive"
]


credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"], scopes=SCOPES
)

docs_service = build("docs", "v1", credentials=credentials)


# ─── Helpers ──────────────────────────────────────────────────────
def get_full_text(document_id: str) -> str:
    """Fetches the entire body text of the document."""
    doc = docs_service.documents().get(documentId=document_id).execute()
    pieces = doc.get("body", {}).get("content", [])
    text = []
    for el in pieces:
        if "paragraph" in el:
            for run in el["paragraph"]["elements"]:
                txt = run.get("textRun", {}).get("content")
                if txt:
                    text.append(txt)
    return "".join(text).strip()


def append_paragraph(document_id: str, text: str):
    """Appends a new paragraph at the end of the document."""
    # Fetch the document to get the current end index
    doc = docs_service.documents().get(documentId=document_id).execute()
    end_index = doc.get("body", {}).get("content", [])[-1].get("endIndex", 1)

    # Create the request to insert text at the end of the document
    requests = [
        {
            "insertText": {
                "location": {"index": end_index - 1},  # Use the valid end index
                "text": text + "\n",
            }
        }
    ]
    docs_service.documents().batchUpdate(
        documentId=document_id, body={"requests": requests}
    ).execute()


def generate_from_openai(prompt: str, temperature: float = 0.0, max_tokens: int = 2060) -> str:
    """Calls ChatCompletions and returns the assistant's reply."""
    client = OpenAI()
    completion = client.chat.completions.create(model=MODEL,
        messages=[{"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens)
    return completion.choices[0].message.content.strip()


# ─── Main features ─────────────────────────────────────────────────
async def scrape_events(url="https://lu.ma/genai-sf?k=c", source_name="Lu.ma GenAI SF"):
    """Use browser-use to scrape events from lu.ma/genai-sf"""
    from browser_use import ChatOpenAI
    from browser_use.agent.service import Agent
    from browser_use.browser import BrowserProfile, BrowserSession
    import os

    # Ensure API key is in environment
    os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

    # Set up browser session
    browser_session = BrowserSession(
        browser_profile=BrowserProfile(
            keep_alive=False,
            headless=False,  # Show browser window during scraping
            record_video_dir=None,
        )
    )

    # Set up the LLM using browser_use's ChatOpenAI
    llm = ChatOpenAI(model='gpt-4o-mini')

    # System prompt for formatting
    system_prompt = """You are extracting event information. Format events as:

Event Name: [Name]
Date and Time: [Date and Time]
Location/Venue: [Venue/Address]
Brief Description: [Brief description including organizer/host]
Event URL: [ACTUAL URL - click on event to get the full URL like https://lu.ma/event-name, NOT just "Link"]

IMPORTANT:
- For Event URL, you MUST click on each event or extract the href attribute to get the ACTUAL URL
- Never use "Link" as the URL - always get the real URL like https://lu.ma/xyz or https://cerebralvalley.ai/events/abc
- Stop scrolling once you have events for 8 days or after 3 scrolls maximum."""

    # Task to scrape events
    task = f"""Go to {url} and extract AI/GenAI event information.

    CRITICAL INSTRUCTIONS:
    1. Load the page
    2. Extract ALL visible events from the current view with their ACTUAL URLs (not "Link")
    3. Scroll down MAXIMUM 2 times to see more events
    4. Extract any additional events
    5. STOP IMMEDIATELY - do not scroll more than 2 times total
    6. Return all collected events

    STOP CONDITIONS:
    - After 2 scrolls maximum
    - When you have at least 5 events
    - When no new events appear after scrolling

    DO NOT CONTINUE SCROLLING BEYOND 2 SCROLLS. STOP AND RETURN RESULTS.

    For each event, extract:
    - Event Name
    - Date and Time
    - Location/Venue
    - Brief Description
    - Event URL (can be relative path)

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
            max_actions=12  # Strict limit to prevent infinite loops
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


def _extract_event_links(events_block: str):
    """Return (event_name, url) tuples from the combined events block."""
    if not events_block:
        return []

    pattern = re.compile(r"\*\*(?P<name>[^*]+)\*\*.*?Sign-up URL:\s*(?P<url>https?://\S+)", re.DOTALL)
    matches = []
    for match in pattern.finditer(events_block):
        name = match.group("name").strip()
        url = match.group("url").strip().rstrip('.,)')
        if name and url:
            matches.append((name, url))

    # Fallback: look for less-structured "Sign-up URL" lines and capture the preceding line as the name
    if not matches:
        lines = events_block.splitlines()
        for idx, line in enumerate(lines):
            if "Sign-up URL:" in line:
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
                "Every time you mention an event by name, immediately include its sign-up URL in parentheses right after the event name, e.g., 'AI Summit (https://example.com)'. "
                "Do not reference an event without its URL, and only use URLs supplied below or in the source content.\n\n"
                f"{links_guidance}"
                f"Event source material:\n{selected}"
            )
        else:
            st.warning("No events data available. Please scrape events first.")
            return False, None

        # Generate essay using OpenAI
        result = generate_from_openai(prompt, temperature=0.7, max_tokens=1500)

        # Display essay on the page
        st.subheader("📝 Generated Essay")
        st.markdown("**Essay based on scraped events:**")
        st.markdown(result)

        print("✅ Essay generated successfully.")
        return True, result
    except Exception as e:
        st.error(f"Error generating essay: {str(e)}")
        return False, None


def generate_events(url="https://lu.ma/genai-sf?k=c", source_name="Lu.ma GenAI SF"):
    """Go to specified URL and get the events for the next 8 days"""
    try:
        # Run the async scraping function
        events_data = asyncio.run(scrape_events(url, source_name))

        if events_data:
            # Format the events for display and Google Doc
            formatted_events = format_events_for_doc(events_data, source_name)

            # Display results on the page
            st.subheader(f"📅 {source_name} Events")
            st.markdown("**Scraped Events:**")

            # Display as rendered markdown instead of plain text
            with st.container():
                st.markdown(formatted_events)

            # No longer writing to Google Doc
            print(f"✅ Successfully scraped {source_name} events.")

            return True, formatted_events
        else:
            st.error(f"Failed to retrieve events from {source_name}")
            return False, None
    except Exception as e:
        st.error(f"Error in generate_events: {str(e)}")
        return False, None


def format_events_for_doc(events_data, source_name="Events"):
    """Format the scraped events into a readable document format"""
    try:
        # Get current date and next 8 days
        today = datetime.now()
        end_date = today + timedelta(days=8)

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
            return '\n\n'.join(events).strip()

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
            return '\n'.join(event_lines)

        return "Unable to parse event information from agent result."

    except Exception as e:
        return f"Error extracting events: {str(e)}"


def clean_event_content(content):
    """Clean up event content for better formatting"""
    # Remove escape characters and clean up formatting
    content = content.replace('\\n', '\n').replace('\\t', '\t')

    # Remove extra whitespace and normalize line breaks
    lines = [line.strip() for line in content.split('\n') if line.strip()]

    # Fix relative URLs to use full lu.ma URLs
    fixed_lines = []
    for line in lines:
        # Check for any URLs that need fixing (example.com or relative paths)
        if 'example.com' in line or '[Event Link](/' in line or '**Link:**' in line or 'Link:' in line:
            line = fix_example_com_urls(line)
        fixed_lines.append(line)

    return '\n'.join(fixed_lines)


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
                full_url = f"https://lu.ma{url_part}"
            elif 'example.com' in url_part:
                # Replace example.com with lu.ma
                event_id = url_part.split('/')[-1]
                full_url = f"https://lu.ma/{event_id}"
            elif not url_part.startswith('http'):
                # Just a slug
                full_url = f"https://lu.ma/{url_part}"
            else:
                # Already a full URL, but check if it needs hostname replacement
                if 'example.com' in url_part:
                    event_id = url_part.split('/')[-1]
                    full_url = f"https://lu.ma/{event_id}"
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

    # Use OpenAI to combine and format the events
    prompt = """Take all the events from both sources and combine them into a single chronologically ordered list.

CRITICAL: Use EXACTLY this format with NEWLINES after each field:

**[Date in format: Month DD, YYYY]**

1. **[Event Name]**
   Time: [Time or "Time TBD"]
   Location: [Location]
   Host: [Host organization or description]
   Sign-up URL: [URL]

2. **[Next Event Name]**
   Time: [Time]
   Location: [Location]
   Host: [Host]
   Sign-up URL: [URL]

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
   Sign-up URL: https://example.com/event

Example of WRONG formatting (DO NOT DO THIS):
1. **AI Summit 2024** Time: 10:00 AM Location: San Francisco, CA Host: Tech Organization Sign-up URL: https://example.com/event

Additional rules:
- Group all events by date, sort dates chronologically
- Combine events from BOTH sources into single date groups
- Extract actual URLs, never show just "Link"

IMPORTANT:
- Combine ALL events from both sources into a single unified list, not two separate sections.
- Show actual URLs directly (e.g., https://lu.ma/event-name), never just show "Link"
- If a URL appears as "[text](url)" markdown format, extract and show just the URL"""

    try:
        result = generate_from_openai(combined_text + "\n\n" + prompt, temperature=0.1, max_tokens=3000)
        return result
    except Exception as e:
        return f"Error combining events: {str(e)}"


def fix_example_com_urls(line, base_url="https://lu.ma"):
    """Replace example.com URLs and relative URLs with proper base URLs"""
    import re

    # Determine base domain from context
    if 'cerebralvalley' in line.lower():
        base_url = "https://cerebralvalley.ai"

    # Pattern 1: Markdown links with example.com
    example_pattern = r'\[([^\]]+)\]\s*\((https://example\.com/[^\)]+)\)'

    def replace_example_url(match):
        link_text = match.group(1)
        url = match.group(2)
        event_id = url.split('/')[-1].strip()
        # For lu.ma, use just the ID; for others, might need /events/ prefix
        if base_url == "https://lu.ma":
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
        if base_url == "https://lu.ma":
            event_id = path.lstrip('/')
            return f'[{link_text}]({base_url}/{event_id})'
        else:
            return f'[{link_text}]({base_url}{path})'

    fixed_line = re.sub(relative_pattern, replace_relative_url, fixed_line)

    # Pattern 3: Plain example.com URLs
    plain_example = r'https://example\.com/([^\s\)]+)'
    if base_url == "https://lu.ma":
        fixed_line = re.sub(plain_example, r'https://lu.ma/\1', fixed_line)
    else:
        fixed_line = re.sub(plain_example, base_url + r'/events/\1', fixed_line)

    # Pattern 4: Plain relative paths in Link: lines
    if 'Link:' in fixed_line or '**Link:**' in fixed_line:
        plain_relative = r'(\*\*Link:\*\*|\bLink:)\s*(/[^\s]+)'
        if base_url == "https://lu.ma":
            fixed_line = re.sub(plain_relative, r'\1 https://lu.ma\2', fixed_line)
        else:
            fixed_line = re.sub(plain_relative, r'\1 ' + base_url + r'\2', fixed_line)

    return fixed_line



def main():
    st.title("AI Events Scraper")
    st.header("Automatically extract and display AI events")

    st.divider()

    st.subheader("🎯 Event Sources")

    col1, col2 = st.columns(2)

    with col1:
        st.write("**Lu.ma GenAI SF Events**")
        button1 = st.button("Scrape Lu.ma GenAI SF", key="luma_button")
        if button1:
            with st.spinner("Scraping Lu.ma events..."):
                success, events = generate_events("https://lu.ma/genai-sf?k=c", "Lu.ma GenAI SF")
                if success:
                    st.success("✅ Lu.ma events scraped successfully!")
                else:
                    st.error("❌ Failed to scrape Lu.ma events")

    with col2:
        st.write("**Cerebral Valley Events**")
        button2 = st.button("Scrape Cerebral Valley", key="cv_button")
        if button2:
            with st.spinner("Scraping Cerebral Valley events..."):
                success, events = generate_events("https://cerebralvalley.ai/events", "Cerebral Valley")
                if success:
                    st.success("✅ Cerebral Valley events scraped successfully!")
                else:
                    st.error("❌ Failed to scrape Cerebral Valley events")

    st.divider()

    # Add a button to scrape both sources
    st.subheader("🚀 Bulk Actions")
    button_all = st.button("Scrape All Sources", key="all_button", type="primary")
    if button_all:
        with st.spinner("Scraping all event sources..."):
            success_count = 0
            all_events = []

            # Scrape Lu.ma
            st.write("1️⃣ Scraping Lu.ma...")
            success, events = generate_events("https://lu.ma/genai-sf?k=c", "Lu.ma GenAI SF")
            if success:
                success_count += 1
                all_events.append(events)
                st.success("✅ Lu.ma done!")

            # Scrape Cerebral Valley
            st.write("2️⃣ Scraping Cerebral Valley...")
            success, events = generate_events("https://cerebralvalley.ai/events", "Cerebral Valley")
            if success:
                success_count += 1
                all_events.append(events)
                st.success("✅ Cerebral Valley done!")

            # Store all events in session state for persistence
            if all_events:
                st.session_state.all_events = all_events

            # Display combined results summary
            if all_events:
                st.divider()
                st.subheader("📊 Results Summary")

                # Create tabs for each source plus combined view
                if len(all_events) == 2:
                    tab1, tab2, tab3 = st.tabs(["Lu.ma Events", "Cerebral Valley Events", "📋 Combined Results"])

                    with tab1:
                        st.markdown("**Lu.ma Events**")
                        st.markdown(all_events[0])

                    with tab2:
                        st.markdown("**Cerebral Valley Events**")
                        st.markdown(all_events[1])

                    with tab3:
                        # Parse and sort all events chronologically
                        formatted_combined = parse_and_format_combined_events(all_events)
                        st.markdown("**All Events Combined (Chronological Order)**")

                        # Debug info
                        if not formatted_combined.strip():
                            st.warning("No combined events found. Debug info:")
                            st.write(f"Number of event sources: {len(all_events)}")
                            for i, events in enumerate(all_events):
                                st.write(f"Source {i+1} length: {len(events) if events else 0}")
                                if events:
                                    st.text_area(f"Source {i+1} raw content (first 500 chars)", events[:500])
                        else:
                            st.markdown(formatted_combined)

                        # Store combined events in session state for essay generation
                        st.session_state.combined_events = formatted_combined

                elif len(all_events) == 1:
                    # If only one source succeeded
                    st.text_area("Events", value=all_events[0], height=300)

            st.balloons()
            st.success(f"🎉 Completed! Successfully scraped {success_count}/2 sources")

    # Display stored events if they exist (persists after page rerun)
    if 'all_events' in st.session_state and st.session_state.all_events:
        st.divider()
        st.subheader("📊 Stored Results")

        # Create tabs for each source plus combined view
        if len(st.session_state.all_events) == 2:
            tab1, tab2, tab3 = st.tabs(["Lu.ma Events", "Cerebral Valley Events", "All Events Combined"])

            with tab1:
                st.markdown("**Lu.ma Events**")
                st.markdown(st.session_state.all_events[0])

            with tab2:
                st.markdown("**Cerebral Valley Events**")
                st.markdown(st.session_state.all_events[1])

            with tab3:
                # Parse and sort all events chronologically
                formatted_combined = parse_and_format_combined_events(st.session_state.all_events)
                st.markdown("**All Events Combined (Chronological Order)**")

                if formatted_combined.strip():
                    st.markdown(formatted_combined)
                    # Store combined events for essay generation
                    st.session_state.combined_events = formatted_combined
                else:
                    st.warning("No combined events found.")
        elif len(st.session_state.all_events) == 1:
            st.markdown("**Scraped Events**")
            st.markdown(st.session_state.all_events[0])

    st.divider()

    st.subheader("💽 Save Current Events")
    combined_events_raw = st.session_state.get('combined_events', '') if 'combined_events' in st.session_state else ''
    if combined_events_raw.strip():
        st.download_button(
            "Download Combined Events",
            data=combined_events_raw,
            file_name="combined_events.txt",
            mime="text/plain",
            help="Save the latest combined events to reuse later without scraping."
        )
    else:
        st.info("No combined events available to save yet. Scrape or load events first.")

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
                st.text_area("Loaded Events", uploaded_content, height=250)
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




if __name__ == "__main__":
    main()
