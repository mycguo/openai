"""
Subagents are a way to delegate tasks to specialized agents.

Advantages include:
- Context isolation: Subagents have their own context and do not share it with the main agent.
- Tool isolation: Subagents can have their own set of allowed tools, which can be useful for security and manageability.
- Parallelization: Subagents can run in parallel, which can improve performance.

For more details, see: https://docs.claude.com/en/api/agent-sdk/subagents

This is a Streamlit app version of the subagents example.
"""

import streamlit as st
import asyncio
import logging
from claude_agent_sdk import ClaudeSDKClient, ClaudeAgentOptions, AgentDefinition
from dotenv import load_dotenv

load_dotenv()

# Configure logging to console
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Page configuration
st.set_page_config(
    page_title="Kaya - Your Personal Assistant",
    page_icon="🤖",
    layout="wide"
)

# Initialize session state
if "messages" not in st.session_state:
    st.session_state.messages = []
if "client" not in st.session_state:
    st.session_state.client = None
if "model" not in st.session_state:
    st.session_state.model = "sonnet"
if "current_model" not in st.session_state:
    st.session_state.current_model = None


def get_claude_options(model: str):
    """Configure Claude agent options with subagents."""
    return ClaudeAgentOptions(
        model=model,
        permission_mode="bypassPermissions",  # Bypass all permission prompts for MCP tools
        setting_sources=["project"],
        max_buffer_size=10485760,  # 10MB buffer size to handle large responses
        allowed_tools=[
            'Read',
            'Write',
            'Edit',
            'MultiEdit',
            'Grep',
            'Glob',
            # Task tool is required to use subagents!
            'Task',
            'TodoWrite',
            'WebSearch',
            'WebFetch',
            # Playwright tools
            'mcp__Playwright__browser_close',
            'mcp__Playwright__browser_resize',
            'mcp__Playwright__browser_console_messages',
            'mcp__Playwright__browser_handle_dialog',
            'mcp__Playwright__browser_evaluate',
            'mcp__Playwright__browser_file_upload',
            'mcp__Playwright__browser_fill_form',
            'mcp__Playwright__browser_install',
            'mcp__Playwright__browser_press_key',
            'mcp__Playwright__browser_type',
            'mcp__Playwright__browser_navigate',
            'mcp__Playwright__browser_navigate_back',
            'mcp__Playwright__browser_network_requests',
            'mcp__Playwright__browser_take_screenshot',
            'mcp__Playwright__browser_snapshot',
            'mcp__Playwright__browser_click',
            'mcp__Playwright__browser_drag',
            'mcp__Playwright__browser_hover',
            'mcp__Playwright__browser_select_option',
            'mcp__Playwright__browser_tabs',
            'mcp__Playwright__browser_wait_for',
            # Chrome DevTools tools
            # Input automation
            'mcp__chrome-devtools__click',
            'mcp__chrome-devtools__drag',
            'mcp__chrome-devtools__fill',
            'mcp__chrome-devtools__fill_form',
            'mcp__chrome-devtools__handle_dialog',
            'mcp__chrome-devtools__hover',
            'mcp__chrome-devtools__upload_file',
            # Navigation automation
            'mcp__chrome-devtools__close_page',
            'mcp__chrome-devtools__list_pages',
            'mcp__chrome-devtools__navigate_page',
            'mcp__chrome-devtools__navigate_page_history',
            'mcp__chrome-devtools__new_page',
            'mcp__chrome-devtools__select_page',
            'mcp__chrome-devtools__wait_for',
            # Emulation
            'mcp__chrome-devtools__emulate_cpu',
            'mcp__chrome-devtools__emulate_network',
            'mcp__chrome-devtools__resize_page',
            # Performance
            'mcp__chrome-devtools__performance_analyze_insight',
            'mcp__chrome-devtools__performance_start_trace',
            'mcp__chrome-devtools__performance_stop_trace',
            # Network
            'mcp__chrome-devtools__get_network_request',
            'mcp__chrome-devtools__list_network_requests',
            # Debugging
            'mcp__chrome-devtools__evaluate_script',
            'mcp__chrome-devtools__get_console_message',
            'mcp__chrome-devtools__list_console_messages',
            'mcp__chrome-devtools__take_screenshot',
            'mcp__chrome-devtools__take_snapshot',
        ],
        # We can also specify allowed tools for subagents, by default they inherit all tools including MCP tools.
        agents={
            "youtube-analyst": AgentDefinition(
                description="An expert at analyzing a user's Youtube channel performance. The analyst will produce a markdown report in the /docs directory.",
                prompt="You are an expert at analyzing YouTube data and helping the user understand their performance. You can use the Playwright browser tools to access the user's Youtube Studio. Generate a markdown report in the /docs directory.",
                model="sonnet",
                tools=[
                    'Read',
                    'Write',
                    'Edit',
                    'MultiEdit',
                    'Grep',
                    'Glob',
                    'TodoWrite',
                    'mcp__Playwright__browser_close',
                    'mcp__Playwright__browser_resize',
                    'mcp__Playwright__browser_console_messages',
                    'mcp__Playwright__browser_handle_dialog',
                    'mcp__Playwright__browser_evaluate',
                    'mcp__Playwright__browser_file_upload',
                    'mcp__Playwright__browser_fill_form',
                    'mcp__Playwright__browser_install',
                    'mcp__Playwright__browser_press_key',
                    'mcp__Playwright__browser_type',
                    'mcp__Playwright__browser_navigate',
                    'mcp__Playwright__browser_navigate_back',
                    'mcp__Playwright__browser_network_requests',
                    'mcp__Playwright__browser_take_screenshot',
                    'mcp__Playwright__browser_snapshot',
                    'mcp__Playwright__browser_click',
                    'mcp__Playwright__browser_drag',
                    'mcp__Playwright__browser_hover',
                    'mcp__Playwright__browser_select_option',
                    'mcp__Playwright__browser_tabs',
                    'mcp__Playwright__browser_wait_for',
                ]
            ),
            "researcher": AgentDefinition(
                description="An expert researcher and documentation writer. The agent will perform deep research of a topic and generate a report or documentation in the /docs directory.",
                prompt="You are an expert researcher and report/documentation writer. Use the WebSearch and WebFetch tools to perform research. You can research multiple subtopics/angles to get a holistic understanding of the topic. You can use filesystem tools to track findings and data in the /docs directory. For longer reports, you can break the work into multiple tasks or write sections at a time. But the final output should be a single markdown report. The final report **MUST** include a citations section with links to all sources used. Review the full report, identify any areas for improvement in readability, cohorerence, and relevancy, and make any necessary edits before declaring the task complete. Clean up any extraneous files and only leave the final report in the /docs directory when you are done. You are only permitted to use these specific tools: Read, Write, Edit, MultiEdit, Grep, Glob, TodoWrite, WebSearch, WebFetch. All other tools are prohibited.",
                model="sonnet",
                tools=[
                    'Read',
                    'Write',
                    'Edit',
                    'MultiEdit',
                    'Grep',
                    'Glob',
                    'TodoWrite',
                    'WebSearch',
                    'WebFetch',
                ]
            ),
            "events_agent": AgentDefinition(
                description="You are gathering incoming AI events in bay area. Search for sources from https://luma.com/sf, Meetup, eventbrite, startupgrind, Y combinator, 500 startups, Andreessen Horowitz (a16z), Stanford Events, Berkeley Events, LinkedIn Events, Silicon Valley Forum, Galvanize, StrictlyVC, Bay Area Tech Events, cerebralvalley.ai/events. You must include RSVP URL.",
                prompt="""You are an AI events researcher for the Bay Area. When the user specifies a time period (e.g., "next 9 days"), gather AI/tech events from multiple sources.

**Your Task:**
1. Use browser tools to scrape events from key sources:
   - https://lu.ma/sf (primary source)
   - https://cerebralvalley.ai/events
   - Meetup, Eventbrite, Y Combinator events, a16z events

2. For each event, extract:
   - Event title
   - Date and time
   - Location (physical or virtual)
   - RSVP/Registration URL

3. **CRITICAL**: After gathering events, you MUST provide a text summary directly in your response. Format the results as a clean, readable list.

**Output Format:**
Present events in chronological order like this:

## AI Events - [Date Range]

**[Date] - [Event Title]**
- Time: [Time]
- Location: [Location]
- RSVP: [URL]

**DO NOT:**
- Write to files or create documents
- Just use tools without providing a text summary
- Include event status fields

**REMEMBER:** Always end your work by providing a formatted text response with all the events you found.""",
                model="sonnet",
                tools=[
                    'Read',
                    'TodoWrite',
                    'mcp__Playwright__browser_close',
                    'mcp__Playwright__browser_resize',
                    'mcp__Playwright__browser_console_messages',
                    'mcp__Playwright__browser_handle_dialog',
                    'mcp__Playwright__browser_evaluate',
                    'mcp__Playwright__browser_file_upload',
                    'mcp__Playwright__browser_fill_form',
                    'mcp__Playwright__browser_install',
                    'mcp__Playwright__browser_press_key',
                    'mcp__Playwright__browser_type',
                    'mcp__Playwright__browser_navigate',
                    'mcp__Playwright__browser_navigate_back',
                    'mcp__Playwright__browser_network_requests',
                    'mcp__Playwright__browser_take_screenshot',
                    'mcp__Playwright__browser_snapshot',
                    'mcp__Playwright__browser_click',
                    'mcp__Playwright__browser_drag',
                    'mcp__Playwright__browser_hover',
                    'mcp__Playwright__browser_select_option',
                    'mcp__Playwright__browser_tabs',
                    'mcp__Playwright__browser_wait_for',
                ]
            ),
            "debug_agent": AgentDefinition(
                description="An expert at debugging website issues using Chrome DevTools. The agent will analyze console errors, network requests, performance issues, and accessibility problems.",
                prompt="""You are a website debugging expert using Chrome DevTools.

Your capabilities include:
- Analyzing console errors and warnings
- Inspecting network requests and responses
- Identifying performance bottlenecks
- Checking for accessibility issues
- Examining DOM structure and CSS issues
- Testing JavaScript functionality
- Validating API calls and data flows

When debugging:
1. First ask the user for the website URL or issue description
2. Use browser tools to navigate to the site and take snapshots
3. Check console messages for errors or warnings
4. Inspect network requests for failed calls or slow responses
5. Look for performance issues or resource loading problems
6. Provide clear analysis with specific recommendations
7. Generate a detailed report in the /docs directory if requested

IMPORTANT DATA HANDLING:
- ALWAYS summarize findings rather than dumping raw data
- For console errors: Report count, types, and key messages (not full logs)
- For network requests: Summarize slow requests, failed calls, and statistics
- For performance: Provide metrics and insights, not raw trace data
- Use performance_analyze_insight for performance analysis
- Keep responses concise and actionable
- If generating reports, write them to /docs directory instead of returning inline

Always provide actionable insights and specific steps to fix the issues found.""",
                model="sonnet",
                tools=[
                    'Read',
                    'Write',
                    'Edit',
                    'MultiEdit',
                    'Grep',
                    'Glob',
                    'TodoWrite',
                    # Playwright tools
                    'mcp__Playwright__browser_close',
                    'mcp__Playwright__browser_resize',
                    'mcp__Playwright__browser_console_messages',
                    'mcp__Playwright__browser_handle_dialog',
                    'mcp__Playwright__browser_evaluate',
                    'mcp__Playwright__browser_file_upload',
                    'mcp__Playwright__browser_fill_form',
                    'mcp__Playwright__browser_install',
                    'mcp__Playwright__browser_press_key',
                    'mcp__Playwright__browser_type',
                    'mcp__Playwright__browser_navigate',
                    'mcp__Playwright__browser_navigate_back',
                    'mcp__Playwright__browser_network_requests',
                    'mcp__Playwright__browser_take_screenshot',
                    'mcp__Playwright__browser_snapshot',
                    'mcp__Playwright__browser_click',
                    'mcp__Playwright__browser_drag',
                    'mcp__Playwright__browser_hover',
                    'mcp__Playwright__browser_select_option',
                    'mcp__Playwright__browser_tabs',
                    'mcp__Playwright__browser_wait_for',
                    # Chrome DevTools tools
                    # Input automation
                    'mcp__chrome-devtools__click',
                    'mcp__chrome-devtools__drag',
                    'mcp__chrome-devtools__fill',
                    'mcp__chrome-devtools__fill_form',
                    'mcp__chrome-devtools__handle_dialog',
                    'mcp__chrome-devtools__hover',
                    'mcp__chrome-devtools__upload_file',
                    # Navigation automation
                    'mcp__chrome-devtools__close_page',
                    'mcp__chrome-devtools__list_pages',
                    'mcp__chrome-devtools__navigate_page',
                    'mcp__chrome-devtools__navigate_page_history',
                    'mcp__chrome-devtools__new_page',
                    'mcp__chrome-devtools__select_page',
                    'mcp__chrome-devtools__wait_for',
                    # Emulation
                    'mcp__chrome-devtools__emulate_cpu',
                    'mcp__chrome-devtools__emulate_network',
                    'mcp__chrome-devtools__resize_page',
                    # Performance
                    'mcp__chrome-devtools__performance_analyze_insight',
                    'mcp__chrome-devtools__performance_start_trace',
                    'mcp__chrome-devtools__performance_stop_trace',
                    # Network
                    'mcp__chrome-devtools__get_network_request',
                    'mcp__chrome-devtools__list_network_requests',
                    # Debugging
                    'mcp__chrome-devtools__evaluate_script',
                    'mcp__chrome-devtools__get_console_message',
                    'mcp__chrome-devtools__list_console_messages',
                    'mcp__chrome-devtools__take_screenshot',
                    'mcp__chrome-devtools__take_snapshot',
                ]
            ),
            "ipo_agent": AgentDefinition(
                description="An expert at tracking and analyzing AI companies planning to IPO soon in the United States. The agent will research upcoming IPOs and generate a list.",
                prompt="""You are an expert financial analyst specializing in AI company IPOs in the United States.

Your mission is to identify and analyze AI companies that are planning to go public soon in the US market.

Research sources to check:
- Financial news sites (Bloomberg, Reuters, CNBC, Wall Street Journal)
- IPO tracking platforms (Renaissance Capital, IPO Calendar, Nasdaq IPO Calendar)
- Tech news sources (TechCrunch, The Information, VentureBeat)
- SEC filings (S-1 filings, IPO registrations)


For each AI company planning to IPO, gather:
1. Company name and description
2. Estimated IPO timeline (month/quarter/year)

Research methodology:
1. Start with web searches for "AI companies IPO 2025", "upcoming AI IPOs", "AI startups going public"
2. Check major IPO tracking websites and financial news sources
3. Search for recent S-1 filings and IPO announcements
4. Cross-reference multiple sources to verify information
5. Focus on companies with concrete IPO plans (not just speculation)
""",
                model="sonnet",
                tools=[
                    'Read',
                    'Write',
                    'Edit',
                    'MultiEdit',
                    'Grep',
                    'Glob',
                    'TodoWrite',
                    'WebSearch',
                    'WebFetch',
                    # Playwright tools for accessing IPO tracking websites
                    'mcp__Playwright__browser_close',
                    'mcp__Playwright__browser_resize',
                    'mcp__Playwright__browser_console_messages',
                    'mcp__Playwright__browser_handle_dialog',
                    'mcp__Playwright__browser_evaluate',
                    'mcp__Playwright__browser_fill_form',
                    'mcp__Playwright__browser_install',
                    'mcp__Playwright__browser_press_key',
                    'mcp__Playwright__browser_type',
                    'mcp__Playwright__browser_navigate',
                    'mcp__Playwright__browser_navigate_back',
                    'mcp__Playwright__browser_network_requests',
                    'mcp__Playwright__browser_take_screenshot',
                    'mcp__Playwright__browser_snapshot',
                    'mcp__Playwright__browser_click',
                    'mcp__Playwright__browser_hover',
                    'mcp__Playwright__browser_select_option',
                    'mcp__Playwright__browser_tabs',
                    'mcp__Playwright__browser_wait_for',
                ]
            )
        },
        # Note: Playwright requires Node.js and Chrome to be installed!
        mcp_servers={
            "Playwright": {
                "command": "npx",
                "args": [
                    "-y",
                    "@playwright/mcp@latest"
                ]
            },
            "chrome-devtools": {
                "command": "npx",
                "args": [
                    "chrome-devtools-mcp@latest"
                ]
            }
        }
    )


async def get_or_create_client(model: str):
    """Get existing client or create a new one if needed."""
    # If model changed or no client exists, create a new one
    if st.session_state.client is None or st.session_state.get("current_model") != model:
        # Close existing client if any
        if st.session_state.client is not None:
            try:
                await st.session_state.client.__aexit__(None, None, None)
            except Exception as e:
                logger.warning(f"Error closing previous client: {e}")

        # Create new client
        options = get_claude_options(model)
        client = ClaudeSDKClient(options=options)
        await client.__aenter__()
        st.session_state.client = client
        st.session_state.current_model = model
        logger.info(f"Created new Claude SDK client with model: {model}")

    return st.session_state.client


async def process_message(user_input: str, model: str):
    """Process user message and get response from Claude."""
    logger.info(f"Processing user input: {user_input[:100]}...")

    # Get or create the client (reuses existing if same model)
    client = await get_or_create_client(model)

    await client.query(user_input)
    logger.info("Query sent to Claude SDK")

    response_text = ""
    async for message in client.receive_response():
        # Log the message activity to console
        logger.debug(f"Received message type: {type(message)}")
        if hasattr(message, 'content'):
            logger.debug(f"Message has content, type: {type(message.content)}")
            if isinstance(message.content, list):
                logger.debug(f"Content is list with {len(message.content)} items")
                for i, content_block in enumerate(message.content):
                    block_type = getattr(content_block, 'type', 'unknown')
                    logger.debug(f"Content block {i}: type={block_type}")

                    # Log tool usage to console
                    if hasattr(content_block, 'type') and content_block.type == 'tool_use':
                        tool_name = getattr(content_block, 'name', 'unknown')
                        tool_input = getattr(content_block, 'input', {})
                        logger.info(f"🔧 Using tool: {tool_name}")
                        logger.debug(f"Tool input: {tool_input}")

                    # Extract text content
                    if hasattr(content_block, 'text'):
                        text_content = content_block.text
                        logger.debug(f"Found text content, length: {len(text_content)}")
                        response_text += text_content
            elif hasattr(message.content, 'text'):
                text_content = message.content.text
                logger.debug(f"Found text in content, length: {len(text_content)}")
                response_text += text_content
        else:
            logger.debug(f"Message has no content attribute")

    logger.info(f"Response received, length: {len(response_text)} characters")
    return response_text


# Sidebar
with st.sidebar:
    st.title("⚙️ Settings")

    # Model selection
    model_option = st.selectbox(
        "Select Model",
        ["sonnet", "opus", "haiku"],
        index=0
    )
    st.session_state.model = model_option

    st.divider()

    st.subheader("📋 Available Subagents")
    st.markdown("""
    - **youtube-analyst**: Analyzes YouTube channel performance
    - **researcher**: Performs deep research and generates reports
    - **documentation-writer**: Creates technical documentation
    - **events_agent**: Gathers upcoming AI events in the Bay Area
    - **debug_agent**: Debugs website issues with Chrome DevTools (performance analysis, network inspection, console errors, CPU/network emulation)
    - **ipo_agent**: Tracks AI companies planning to IPO soon in the US market
    """)

    st.divider()

    if st.button("🗑️ Clear Chat History"):
        # Close the client to start a fresh session
        if st.session_state.client is not None:
            try:
                # Schedule client cleanup
                asyncio.create_task(st.session_state.client.__aexit__(None, None, None))
            except Exception as e:
                logger.warning(f"Error closing client: {e}")

        # Reset session state
        st.session_state.messages = []
        st.session_state.client = None
        st.session_state.current_model = None
        st.rerun()

    st.divider()

    st.caption("Powered by Claude Agent SDK")


# Main chat interface
st.title("🤖 Kaya - Your Personal Assistant")
st.caption(f"Using model: {st.session_state.model}")

# Display chat history
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Chat input
if prompt := st.chat_input("What can I help you with today?"):
    # Add user message to chat history
    st.session_state.messages.append({"role": "user", "content": prompt})

    # Display user message
    with st.chat_message("user"):
        st.markdown(prompt)

    # Display assistant response
    with st.chat_message("assistant"):
        with st.spinner("Thinking..."):
            try:
                # Get response from Claude (logging happens in console)
                response = asyncio.run(process_message(prompt, st.session_state.model))

                # Display response
                st.markdown(response)

                # Add assistant message to chat history
                st.session_state.messages.append({"role": "assistant", "content": response})
            except Exception as e:
                error_msg = f"Error: {str(e)}"
                logger.error(f"Error processing message: {e}")
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})
