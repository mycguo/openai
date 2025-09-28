import os
import re
import asyncio
import streamlit as st
from openai import OpenAI

# Configuration
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MODEL = "gpt-3.5-turbo"

# Page configuration
st.set_page_config(
    page_title="Podcast Transcript Summarizer",
    page_icon="🎙️",
    layout="wide"
)

def generate_summary(transcript: str, temperature: float = 0.0, max_tokens: int = 2000) -> str:
    """Generate a summary from podcast transcript using OpenAI."""
    client = OpenAI()

    prompt = f"""Please provide a comprehensive summary of this podcast transcript.
    Include the following:

    1. **Main Topics**: Key subjects discussed
    2. **Key Insights**: Important points and takeaways
    3. **Notable Quotes**: Memorable statements or insights
    4. **Discussion Highlights**: Major discussion points or debates
    5. **Actionable Items**: Any recommendations or actionable advice mentioned

    Format the summary in clear sections with bullet points where appropriate.

    Transcript:
    {transcript[:10000]}
    """

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )
    return completion.choices[0].message.content.strip()

async def scrape_transcript(url: str) -> str:
    """Scrape podcast transcript using browser-use."""
    try:
        from browser_use import ChatOpenAI
        from browser_use.agent.service import Agent
        from browser_use.browser import BrowserProfile, BrowserSession

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

        # Set up the LLM
        llm = ChatOpenAI(model='gpt-4o-mini')

        # System prompt for transcript extraction
        system_prompt = """You are extracting text content from a webpage. Your task is:
        1. Navigate to the provided URL
        2. Find and click on "Transcript" button/link/tab
        3. Extract all visible text content from the page
        4. Return the complete text

        Simple rules:
        - Look for "Transcript" button, link, or tab (case-insensitive)
        - Click on it to reveal the transcript
        - Extract all text content that appears
        - Return the complete text content as-is"""

        # Task to extract transcript
        task = f"""Go to {url} and get the transcript text.

        Steps:
        1. Load the page
        2. Find "Transcript" button/link and click it
        3. Extract all text content from the page
        4. Return the text content

        Just get the text that appears after clicking transcript - nothing fancy.
        """

        # Start browser session
        await browser_session.start()

        # Create the agent with stricter limits
        agent = Agent(
            task=task,
            llm=llm,
            browser_session=browser_session,
            system_message=system_prompt,
            max_actions=10  # Reduced to prevent infinite loops
        )

        # Run the agent
        result = await agent.run()

        # Clean up
        await browser_session.kill()

        return result

    except Exception as e:
        return f"Error scraping transcript: {str(e)}"

def extract_transcript_text(raw_result: str) -> str:
    """Extract clean transcript text from the raw scraping result."""
    text = raw_result

    # Remove task completion messages
    patterns_to_remove = [
        r"Task completed.*",
        r"Successfully.*",
        r"I have.*",
        r"Here.*transcript.*",
        r"The transcript.*"
    ]

    for pattern in patterns_to_remove:
        text = re.sub(pattern, "", text, flags=re.IGNORECASE)

    # Clean up extra whitespace
    text = re.sub(r'\n\s*\n', '\n\n', text)
    text = text.strip()

    return text

def main():
    st.title("🎙️ Podcast Transcript Summarizer")
    st.markdown("Extract podcast transcripts and generate AI-powered summaries")

    # URL input section
    st.subheader("📝 Enter Podcast URL")

    # Default URL for testing
    default_url = "https://lastweekin.ai/p/lwiai-podcast-221-openai-codex-gemini"

    url = st.text_input(
        "Podcast URL",
        value=default_url,
        placeholder="Enter the URL of the podcast page with transcript",
        help="Enter a URL that contains a podcast with an available transcript"
    )

    # Action buttons
    col1, col2 = st.columns([1, 4])

    with col1:
        extract_button = st.button("🔍 Extract & Summarize", type="primary")

    with col2:
        if st.button("🗑️ Clear Results"):
            for key in ["transcript", "summary"]:
                if key in st.session_state:
                    del st.session_state[key]
            st.rerun()

    # Processing section
    if extract_button and url:
        if not url.startswith(('http://', 'https://')):
            st.error("Please enter a valid URL starting with http:// or https://")
            return

        # Step 1: Extract transcript
        st.subheader("🤖 Extracting Transcript...")

        with st.spinner("Navigating to URL and finding transcript..."):
            try:
                raw_transcript = asyncio.run(scrape_transcript(url))

                if raw_transcript and "Error" not in str(raw_transcript):
                    # Clean the transcript
                    clean_transcript = extract_transcript_text(str(raw_transcript))
                    st.session_state.transcript = clean_transcript
                    st.success("✅ Transcript extracted successfully!")
                else:
                    st.error(f"❌ Failed to extract transcript: {raw_transcript}")
                    return

            except Exception as e:
                st.error(f"❌ Error during transcript extraction: {str(e)}")
                return

        # Step 2: Generate summary
        if "transcript" in st.session_state and st.session_state.transcript:
            st.subheader("📋 Generating Summary...")

            with st.spinner("Analyzing transcript and generating summary..."):
                try:
                    summary = generate_summary(st.session_state.transcript)
                    st.session_state.summary = summary
                    st.success("✅ Summary generated successfully!")

                except Exception as e:
                    st.error(f"❌ Error generating summary: {str(e)}")
                    return

    # Display results
    if "transcript" in st.session_state and st.session_state.transcript:
        st.divider()

        # Summary section
        if "summary" in st.session_state and st.session_state.summary:
            st.subheader("📋 Podcast Summary")
            st.markdown(st.session_state.summary)

            # Copy summary button
            if st.button("📋 Copy Summary to Clipboard"):
                st.code(st.session_state.summary)

        # Transcript section (collapsible)
        with st.expander("📝 View Full Transcript", expanded=False):
            st.text_area(
                "Full Transcript",
                value=st.session_state.transcript,
                height=400,
                disabled=True,
                label_visibility="collapsed"
            )

            # Transcript stats
            word_count = len(st.session_state.transcript.split())
            char_count = len(st.session_state.transcript)
            st.caption(f"📊 Transcript stats: {word_count:,} words, {char_count:,} characters")

if __name__ == "__main__":
    main()