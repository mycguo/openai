import os
import re
import asyncio
from typing import Optional

import streamlit as st
from openai import OpenAI

# Configuration
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MODEL = "gpt-4o-mini"


def _create_client() -> OpenAI:
    return OpenAI()

# Page configuration
st.set_page_config(
    page_title="Podcast Transcript Summarizer",
    page_icon="🎙️",
    layout="wide"
)

def generate_summary(transcript: str, temperature: float = 0.0, max_tokens: int = 2000) -> str:
    """Generate a summary from podcast transcript using OpenAI."""
    client = _create_client()

    prompt = f"""Please provide a comprehensive summary of this podcast transcript.
    Include the following:

    1. **Main Topics**: Key subjects discussed
    2. **Key Insights**: Important points and takeaways

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


def generate_deep_research_article(
    transcript: str,
    summary: Optional[str] = None,
    research: Optional[str] = None,
    temperature: float = 0.2,
    max_tokens: int = 2500
) -> str:
    """Create a LinkedIn-ready article with deeper research context."""
    client = _create_client()

    truncated_transcript = transcript[:12000]
    summary_section = summary or "Summary not available."
    research_section = research or "No external research findings available."

    prompt = f"""You are a senior analyst preparing a professional LinkedIn article based on a podcast.

Podcast Summary:
{summary_section}

Relevant External Research Notes:
{research_section}

Podcast Transcript (excerpt if long):
{truncated_transcript}

Deliverables:
1. Craft a compelling LinkedIn article title.
2. Write an engaging introduction that hooks readers and sets context.
3. Provide 3-5 deeply researched insights, weaving in supporting information, data points, or references a knowledgeable professional might mention.
4. Offer actionable recommendations or next steps for professionals interested in this topic.
5. Close with a forward-looking conclusion and an invitation for discussion.

Formatting rules:
- Use clear headings and bullet points where helpful.
- Keep the tone professional, insightful, and concise.
- Ensure the article stands on its own without referring to the transcript explicitly.
"""

    completion = client.chat.completions.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens
    )

    return completion.choices[0].message.content.strip()


def extract_main_topics(summary: str) -> str:
    """Extract the textual content of the 'Main Topics' section from a summary."""
    if not summary:
        return ""

    normalized = summary.replace("\r\n", "\n")
    parts = re.split(r"\*\*(.+?)\*\*", normalized)
    if len(parts) <= 1:
        return ""

    main_section = ""
    for i in range(1, len(parts), 2):
        heading = parts[i].strip().lower()
        content = parts[i + 1] if i + 1 < len(parts) else ""
        if heading.startswith("main topics"):
            main_section = content
            break

    if not main_section:
        return ""

    main_section = main_section.lstrip(": ")
    # Stop at the next heading if it appears in the same block
    main_section = re.split(r"\*\*[^*]+\*\*", main_section, maxsplit=1)[0]

    lines = []
    for line in main_section.splitlines():
        cleaned = line.strip()
        if not cleaned:
            continue
        cleaned = re.sub(r"^[\-\*\u2022]+", "", cleaned).strip()
        if cleaned:
            lines.append(cleaned)

    if not lines:
        collapsed = re.sub(r"[-•]", " ", main_section)
        collapsed = re.sub(r"\s+", " ", collapsed)
        return collapsed.strip()

    return " ".join(lines)

async def scrape_transcript(url: str) -> str:
    """Scrape podcast transcript using browser-use."""
    browser_session = None

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

        return result

    except Exception as e:
        return f"Error scraping transcript: {str(e)}"

    finally:
        try:
            if browser_session is not None:
                await browser_session.kill()
        except Exception:
            pass

def extract_agent_text(raw_result) -> str:
    """Extract clean text content from a browser-use agent result."""
    def _clean_candidate(candidate):
        if isinstance(candidate, dict):
            # Look for text-like values in dicts
            values = [v for v in candidate.values() if isinstance(v, str)]
            return "\n\n".join(filter(None, (_clean_candidate(v) for v in values)))

        if isinstance(candidate, (list, tuple)):
            return "\n\n".join(filter(None, (_clean_candidate(item) for item in candidate)))

        if not isinstance(candidate, str):
            return ""

        candidate = candidate.strip()
        if not candidate:
            return ""

        # Remove metadata prefixes from extracted content
        if "Extracted content from" in candidate:
            for separator in (":\n", ":\r\n", ": "):
                if separator in candidate:
                    candidate = candidate.split(separator, 1)[1].strip()
                    break

        # Filter out navigation/status messages
        if len(candidate.split()) < 10 and "\n" not in candidate:
            return ""

        return candidate

    def _from_agent_history(history):
        texts = []

        all_results = getattr(history, "all_results", None)
        if isinstance(all_results, (list, tuple)):
            for action in all_results:
                extracted = getattr(action, "extracted_content", None)
                cleaned = _clean_candidate(extracted)
                if cleaned:
                    texts.append(cleaned)

                memory = getattr(action, "long_term_memory", None)
                cleaned_memory = _clean_candidate(memory)
                if cleaned_memory:
                    texts.append(cleaned_memory)

                attachments = getattr(action, "attachments", None)
                if isinstance(attachments, (list, tuple)):
                    for attachment in attachments:
                        attachment_content = getattr(attachment, "content", attachment)
                        cleaned_attachment = _clean_candidate(attachment_content)
                        if cleaned_attachment:
                            texts.append(cleaned_attachment)

        final_result = getattr(history, "final_result", None)
        cleaned_final = _clean_candidate(final_result)
        if cleaned_final:
            texts.append(cleaned_final)

        unique_texts = []
        for item in texts:
            if item and item not in unique_texts:
                unique_texts.append(item)

        return "\n\n".join(unique_texts)

    if hasattr(raw_result, "all_results") or hasattr(raw_result, "final_result"):
        text = _from_agent_history(raw_result)
    else:
        text = _clean_candidate(raw_result)

    if not text:
        text = str(raw_result) if raw_result is not None else ""

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


async def conduct_deep_research(topic: str) -> tuple[str, str]:
    """Use browser-use agent to gather supplementary research on the topic.

    Returns a tuple of (notes, query) for debugging visibility.
    """
    browser_session = None

    try:
        from browser_use import ChatOpenAI
        from browser_use.agent.service import Agent
        from browser_use.browser import BrowserProfile, BrowserSession

        os.environ["OPENAI_API_KEY"] = OPENAI_API_KEY

        browser_session = BrowserSession(
            browser_profile=BrowserProfile(
                keep_alive=False,
                headless=True,
                record_video_dir=None,
            )
        )

        llm = ChatOpenAI(model='gpt-4o-mini')

        system_prompt = """You are a research assistant collecting up-to-date insights from reputable online sources.
        - Prioritize recent articles, reports, or trusted publications.
        - Visit multiple sources (at least three) to cross-verify information.
        - Capture key statistics, expert quotes, and contextual details with source URLs.
        - Summarize findings clearly so they can inform thought leadership content.
        - If a UI action fails twice, choose an alternate method (e.g., try pressing Enter, try another search engine, or open a different result). Avoid getting stuck repeating the same click.
        """

        topic_snippet = " ".join(topic.split())[:400]
        query = f"{topic_snippet} latest insights"
        task = f"""Conduct web research on the following podcast topic and produce structured notes.

Topic:
{topic_snippet}

Steps:
1. Navigate directly to https://duckduckgo.com.
2. Enter the exact query "{query}" into the search box and press Enter to execute the search. If results already appear, skip re-running the query.
   - If DuckDuckGo fails to load, choose another search engine (e.g., Bing) and continue.
3. Review the top results and open at least three credible, recent sources in new tabs.
4. Extract important findings, statistics, expert quotes, and record the source URL with each note.
5. Organize the findings by subtopic and highlight actionable takeaways.
6. Return the consolidated notes, ensuring source URLs are clearly listed.
"""

        await browser_session.start()

        agent = Agent(
            task=task,
            llm=llm,
            browser_session=browser_session,
            system_message=system_prompt,
            max_actions=18
        )

        raw_result = await agent.run()
        notes = extract_agent_text(raw_result)
        return notes, query

    except Exception as e:
        raise RuntimeError(f"Error during deep research: {str(e)}") from e

    finally:
        try:
            if browser_session is not None:
                await browser_session.kill()
        except Exception:
            pass

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
            for key in ["transcript", "summary", "deep_article", "deep_research_notes", "deep_research_query"]:
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
                    clean_transcript = extract_agent_text(raw_transcript)
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

            if st.button("🧠 Deep Research Article"):
                with st.spinner("Researching topic and drafting article..."):
                    try:
                        summary_text = st.session_state.summary or ""
                        transcript_text = st.session_state.get("transcript", "")

                        main_topics_text = extract_main_topics(summary_text)
                        topic_source = main_topics_text or summary_text or transcript_text
                        topic_source = re.sub(r"\s+", " ", topic_source).strip()
                        topic_hint = topic_source[:600]

                        if not topic_hint:
                            fallback = re.sub(r"\s+", " ", transcript_text).strip()
                            topic_hint = fallback[:600]

                        research_notes, research_query = asyncio.run(conduct_deep_research(topic_hint))
                        st.session_state.deep_research_notes = research_notes
                        st.session_state.deep_research_query = research_query

                        article = generate_deep_research_article(
                            transcript=st.session_state.transcript,
                            summary=st.session_state.summary,
                            research=research_notes
                        )
                        st.session_state.deep_article = article
                        st.success("✅ Deep research article ready!")
                    except Exception as e:
                        st.error(f"❌ Error creating article: {str(e)}")

        if "deep_research_notes" in st.session_state and st.session_state.deep_research_notes:
            with st.expander("🔍 Research Notes", expanded=False):
                if st.session_state.get("deep_research_query"):
                    st.caption(f"Research query: {st.session_state.deep_research_query}")
                st.markdown(st.session_state.deep_research_notes)

        if "deep_article" in st.session_state and st.session_state.deep_article:
            st.subheader("🧠 LinkedIn Article Draft")
            st.markdown(st.session_state.deep_article)

            if st.button("📋 Copy Article to Clipboard"):
                st.code(st.session_state.deep_article)

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
