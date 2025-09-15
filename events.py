import os
from openai import OpenAI
from googleapiclient.discovery import build
from google.oauth2 import service_account
import streamlit as st
from openai import AsyncOpenAI
import asyncio
from datetime import datetime, timedelta


# https://github.com/amrrs/chatgpt-googledocs/blob/main/appscript.js
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
    """Calls ChatCompletions and returns the assistant’s reply."""
    client = OpenAI()
    completion = client.chat.completions.create(model=MODEL, 
        messages=[{"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens)
    return completion.choices[0].message.content.strip()


# ─── Main features ─────────────────────────────────────────────────
async def scrape_events():
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

IMPORTANT: Stop scrolling once you have events for 8 days or after 3 scrolls maximum."""

    # Task to scrape events
    task = """Go to https://lu.ma/genai-sf?k=c and extract AI/GenAI event information.

    INSTRUCTIONS:
    1. Load the page
    2. Extract visible events on the initial view
    3. Scroll down ONCE to load more events if needed
    4. Extract any additional events
    5. STOP after collecting events for the next 8 days OR after 3 scrolls maximum
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


def generate_events():
    """Go to lu.ma page and get the events for the next 8 days"""
    try:
        # Run the async scraping function
        events_data = asyncio.run(scrape_events())

        if events_data:
            # Format the events for the Google Doc
            formatted_events = format_events_for_doc(events_data)

            # Append to Google Doc
            append_paragraph(DOCUMENT_ID, formatted_events)
            print("✅ Appended events to Google Doc.")
            return True
        else:
            st.error("Failed to retrieve events")
            return False
    except Exception as e:
        st.error(f"Error in generate_events: {str(e)}")
        return False


def format_events_for_doc(events_data):
    """Format the scraped events into a readable document format"""
    try:
        # Get current date and next 8 days
        today = datetime.now()
        end_date = today + timedelta(days=8)

        # Create header
        formatted_text = f"GenAI SF Events - {today.strftime('%B %d, %Y')} to {end_date.strftime('%B %d, %Y')}\n\n"
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


def fix_example_com_urls(line):
    """Replace example.com URLs and relative URLs with lu.ma URLs anywhere in the line"""
    import re

    # Pattern 1: Markdown links with example.com
    example_pattern = r'\[([^\]]+)\]\s*\((https://example\.com/[^\)]+)\)'

    def replace_example_url(match):
        link_text = match.group(1)
        url = match.group(2)
        event_id = url.split('/')[-1].strip()
        return f'[{link_text}](https://lu.ma/{event_id})'

    fixed_line = re.sub(example_pattern, replace_example_url, line)

    # Pattern 2: Markdown links with relative paths (starting with /)
    relative_pattern = r'\[([^\]]+)\]\s*\((/[^\)]+)\)'

    def replace_relative_url(match):
        link_text = match.group(1)
        path = match.group(2)
        # Remove leading slash if present
        event_id = path.lstrip('/')
        return f'[{link_text}](https://lu.ma/{event_id})'

    fixed_line = re.sub(relative_pattern, replace_relative_url, fixed_line)

    # Pattern 3: Plain example.com URLs
    plain_example = r'https://example\.com/([^\s\)]+)'
    fixed_line = re.sub(plain_example, r'https://lu.ma/\1', fixed_line)

    # Pattern 4: Plain relative paths in Link: lines
    if 'Link:' in fixed_line or '**Link:**' in fixed_line:
        # Match paths like /cgk29a8q
        plain_relative = r'(\*\*Link:\*\*|\bLink:)\s*(/[^\s]+)'
        fixed_line = re.sub(plain_relative, r'\1 https://lu.ma\2', fixed_line)

    return fixed_line



def main():
    st.title("OpenAI GoogleDocs Integrations")
    st.header("let the chatbot coming to you")

    st.write("The GoogleDoc Link: https://docs.google.com/document/d/1vbvbDxvKj6LTWKiahK79XTHZsrhfeZpLUfqf1Ocl6RE/edit?tab=t.0 ")


    button1 = st.button("go to https://lu.ma/genai-sf?k=c and get the events for the next 8 days")
    if button1:
        generate_events()
        st.success("Events generated and appended to the document.")

    

    
if __name__ == "__main__":
    main()