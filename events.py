import os
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
            headless=True,
            record_video_dir=None,
        )
    )

    # Set up the LLM using browser_use's ChatOpenAI
    llm = ChatOpenAI(model='gpt-4o-mini')

    # System prompt for formatting
    system_prompt = """You are extracting event information. Format events as:

Event Name: [Name]
Date: [Date and Time]
Location: [Venue/Address]
Description: [Brief description]
Link: [Event URL or path]

IMPORTANT: Stop scrolling once you have events for 8 days or after 5 scrolls maximum."""

    # Task to scrape events
    task = f"""Go to {url} and extract AI/GenAI event information.

    INSTRUCTIONS:
    1. Load the page
    2. Extract visible events on the initial view
    3. Scroll down ONCE to load more events if needed
    4. Extract any additional events
    5. STOP after collecting events for the next 8 days OR after 5 scrolls maximum
    6. Return all collected events

    DO NOT scroll indefinitely. Focus on efficiently extracting available events.

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
            max_actions=20  # Limit actions to prevent infinite loops
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


def generate_essay():
    """Generate an essay based on document content and display it"""
    try:
        # Get content from Google Doc
        selected = get_full_text(DOCUMENT_ID)
        prompt = f"Generate an essay on {selected}"

        # Generate essay using OpenAI
        result = generate_from_openai(prompt)

        # Display essay on the page
        st.subheader("📝 Generated Essay")
        st.markdown("**Essay based on Google Doc content:**")
        st.markdown(result)

        # Silently append to Google Doc (hidden from UI)
        try:
            append_paragraph(DOCUMENT_ID, f"\n\n--- Generated Essay ---\n{result}")
            print("✅ Essay generated and saved to Google Doc.")
        except Exception as e:
            print(f"⚠️ Failed to save essay to Google Doc: {str(e)}")

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
            st.text_area(
                "Scraped Events:",
                value=formatted_events,
                height=400,
                help="Events scraped and displayed below"
            )

            # Silently append to Google Doc (hidden from UI)
            try:
                append_paragraph(DOCUMENT_ID, formatted_events)
                print(f"✅ Successfully scraped {source_name} events and saved to Google Doc.")
            except Exception as e:
                print(f"⚠️ Failed to save to Google Doc: {str(e)}")
                # Don't show this error to user, just log it

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

    # Add essay generation section
    st.subheader("📝 Essay Generation")
    st.write("Generate an essay based on the current Google Doc content")

    button_essay = st.button("Generate Essay from Doc Content", key="essay_button")
    if button_essay:
        with st.spinner("Generating essay from Google Doc content..."):
            success, essay = generate_essay()
            if success:
                st.success("✅ Essay generated successfully!")
            else:
                st.error("❌ Failed to generate essay")

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

            # Display combined results summary
            if all_events:
                st.divider()
                st.subheader("📊 Results Summary")

                # Create tabs for each source plus combined view
                if len(all_events) == 2:
                    tab1, tab2, tab3 = st.tabs(["Lu.ma Events", "Cerebral Valley Events", "📋 Combined Results"])

                    with tab1:
                        st.markdown("**Lu.ma Events**")
                        st.markdown(f"```\n{all_events[0]}\n```")

                    with tab2:
                        st.markdown("**Cerebral Valley Events**")
                        st.markdown(f"```\n{all_events[1]}\n```")

                    with tab3:
                        # Combine all events into one view
                        combined_content = "\n\n" + "="*60 + "\n" + "="*60 + "\n\n".join(all_events)
                        st.markdown("**All Events Combined**")
                        st.markdown(f"```\n{combined_content}\n```")

                        # Add download button for combined results
                        st.download_button(
                            label="📥 Download Combined Events",
                            data=combined_content,
                            file_name=f"ai_events_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                            mime="text/plain"
                        )
                elif len(all_events) == 1:
                    # If only one source succeeded
                    st.text_area("Events", value=all_events[0], height=300)

            st.balloons()
            st.success(f"🎉 Completed! Successfully scraped {success_count}/2 sources")




if __name__ == "__main__":
    main()