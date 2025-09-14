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
    system_prompt = """Format the events in the following format:

Event Name: [Name]
Date: [Date and Time]
Location: [Venue/Address]
Description: [Brief description]
Link: [Registration/Info URL]

Focus on events in the next 8 days."""

    # Task to scrape events
    task = "Go to https://lu.ma/genai-sf?k=c and extract all AI/GenAI event information for the next 8 days."

    try:
        # Start browser session
        await browser_session.start()

        # Create the agent
        agent = Agent(
            task=task,
            llm=llm,
            browser_session=browser_session,
            system_message=system_prompt
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

        # If events_data is a string, try to extract event information
        if isinstance(events_data, str):
            formatted_text += "Events Found:\n\n"
            formatted_text += events_data
        else:
            # If it's structured data, format accordingly
            formatted_text += str(events_data)

        formatted_text += "\n\n" + "=" * 50 + "\n"
        formatted_text += f"Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"

        return formatted_text
    except Exception as e:
        return f"Error formatting events: {str(e)}\n\nRaw data:\n{events_data}"



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