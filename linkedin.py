import streamlit as st
import asyncio
import logging
import os
from pathlib import Path
import shutil
from browser_use import Agent
from browser_use import BrowserSession, BrowserProfile
from browser_use.llm.openai.chat import ChatOpenAI as BrowserUseChatOpenAI
from langchain_openai import ChatOpenAI as LangChainChatOpenAI
import re
from datetime import datetime
import json
from typing import List, Dict, Tuple, Optional
import time
import PyPDF2
import io

LOG_STATE_KEY = "linkedin_agent_logs"


class StreamlitLogHandler(logging.Handler):
    """Logging handler that streams log records into a Streamlit container."""

    def __init__(self, state_key: str, container: "st.delta_generator.DeltaGenerator"):
        super().__init__()
        self.state_key = state_key
        self.container = container
        if self.state_key not in st.session_state:
            st.session_state[self.state_key] = []

    def emit(self, record: logging.LogRecord) -> None:
        try:
            msg = self.format(record)
        except Exception:
            msg = record.getMessage()

        log_buffer = st.session_state.setdefault(self.state_key, [])
        log_buffer.append(msg)
        # Keep the last 500 log lines to avoid unbounded growth
        if len(log_buffer) > 500:
            del log_buffer[:-500]

        # Render the newest buffer contents in the Streamlit container
        if self.container is not None:
            self.container.text("\n".join(log_buffer))


def setup_streamlit_logging(container: "st.delta_generator.DeltaGenerator") -> logging.Logger:
    """Configure or refresh the Streamlit log handler and return the logger."""

    logger = logging.getLogger("linkedin_agent")
    logger.setLevel(logging.INFO)
    logger.propagate = False

    handler = None
    for existing in logger.handlers:
        if getattr(existing, "name", "") == "streamlit_log_handler":
            handler = existing
            break

    if handler is None:
        handler = StreamlitLogHandler(LOG_STATE_KEY, container)
        handler.setFormatter(logging.Formatter("%(asctime)s — %(levelname)s — %(message)s"))
        handler.name = "streamlit_log_handler"
        logger.addHandler(handler)
    else:
        handler.container = container

    return logger


st.set_page_config(
    page_title="LinkedIn AI Jobs Matcher",
    page_icon="💼",
    layout="wide"
)

st.title("💼 LinkedIn AI Jobs Matcher")
st.markdown("Upload your resume and find the best matching AI/GenAI jobs from LinkedIn")

async def scrape_linkedin_jobs(resume_text: str, num_jobs: int = 100, job_search_url: str = "https://www.linkedin.com/jobs/collections/gen-ai/") -> str:
    """Scrape LinkedIn AI jobs using browser-use"""

    logger = logging.getLogger("linkedin_agent")

    async def wait_for_linkedin_login(browser_session: BrowserSession, timeout: int = 300, poll_interval: float = 3.0) -> str:
        """Poll the active page until LinkedIn login appears complete."""

        start = time.monotonic()
        last_url = ""

        while time.monotonic() - start < timeout:
            try:
                current_url = await browser_session.get_current_page_url()
            except Exception as url_error:  # pragma: no cover - defensive logging
                logger.warning("Waiting for LinkedIn login: unable to read current URL (%s)", url_error)
                current_url = ""

            normalized_url = current_url.lower() if current_url else ""

            if normalized_url and "linkedin.com" in normalized_url and not any(
                blocked in normalized_url for blocked in ["/login", "checkpoint", "authwall", "uas/login"]
            ):
                logger.info("Detected LinkedIn login completion at %s", current_url)
                return current_url

            if current_url and current_url != last_url:
                logger.info("Still waiting for LinkedIn login... current page: %s", current_url)
                last_url = current_url

            await asyncio.sleep(poll_interval)

        raise TimeoutError("LinkedIn login was not detected within the allotted time (5 minutes).")

    # Configure browser path/headless mode for local vs. Streamlit deployments
    browser_executable_env = os.environ.get("LINKEDIN_BROWSER_EXECUTABLE")

    candidate_paths: List[str] = []
    if browser_executable_env:
        candidate_paths.append(browser_executable_env)

    # Common Chromium/Chrome names that might be installed via packages.txt or Playwright cache
    for name in [
        "chromium-browser",
        "chromium",
        "google-chrome-stable",
        "google-chrome",
        "chrome"
    ]:
        detected = shutil.which(name)
        if detected:
            candidate_paths.append(detected)

    # Playwright cache location (default install path) if it exists
    default_playwright_cache = Path.home() / ".cache/ms-playwright"
    if default_playwright_cache.exists():
        candidate_paths.extend(
            str(path)
            for path in sorted(default_playwright_cache.glob("chromium-*/chrome-linux/chrome"))
        )

    executable_path_str = next((path for path in candidate_paths if path and Path(path).exists()), None)

    if executable_path_str:
        logger.info("Using Chromium executable at %s", executable_path_str)
    else:
        logger.info("No explicit Chromium executable found; relying on browser_use defaults")

    default_headless = "true" if os.environ.get("STREAMLIT_RUNTIME") else "false"
    headless_flag = os.environ.get("LINKEDIN_BROWSER_HEADLESS", default_headless).lower()
    headless = headless_flag in {"1", "true", "yes", "on"}
    logger.info("Launching browser with headless=%s", headless)

    # Try with explicit browser profile to avoid CDP issues
    browser_profile = BrowserProfile(
        headless=headless,
        keep_alive=True,
        executable_path=executable_path_str
    )
    browser_session = BrowserSession(browser_profile=browser_profile)

    task = f"""
    Look at this LinkedIn jobs page and list {num_jobs} job postings you can see.

    RULES:
    - Maximum 20 actions total (scroll, click, open share modal, close share modal, go back)
    - You may use extract_structured_data AT MOST ONCE per job detail pane (set extract_links=True) to retrieve the job posting URL and summary. Do NOT call it on the listings page.
    - Do NOT create files
    - After 20 actions, STOP and provide your answer
    - Remain on the job listings page. If you open a job detail or company profile, immediately use the browser BACK action to return before continuing.
    - Avoid navigating to company profile pages unless required to capture a job URL.
    - Each job entry MUST include the unique job posting URL (e.g., https://www.linkedin.com/jobs/view/123). After opening a job card, either use the job detail pane's "Share" / "Copy link" option (three-dot menu or share icon) OR a single extract_structured_data call to retrieve the canonical URL. Close any share modal before moving on.
    - Keep track of the jobs you've already collected (by title/company). Do not click the same job card more than once.
    - If you ever land on any URL other than {job_search_url}, immediately navigate back to {job_search_url} before continuing.

    STEPS:
    1. Review the visible job cards and note unique title/company pairs.
    2. Click a job card once to load its details in the right-side pane.
    3. Within the job detail pane, either open the Share/Copy link option OR run one extract_structured_data call with extract_links=True to capture the canonical job URL (contains /jobs/view/...). Read the URL that appears and include it in your notes. Close any modal afterwards.
    4. Record the job title, company, location, posted time, short description, and URL.
    5. Move to the next unseen job card. Scroll the listings pane as needed to reveal more jobs.
    6. After collecting the requested number of jobs, provide your final answer immediately using the required format.

    FORMAT YOUR FINAL RESPONSE EXACTLY LIKE THIS:

    JOB_START
    Title: Senior AI Engineer
    Company: OpenAI
    Location: San Francisco, CA
    Posted: 3 days ago
    Description: Build AI systems...
    URL: https://linkedin.com/jobs/view/123
    JOB_END

    JOB_START
    Title: Machine Learning Engineer
    Company: Google
    Location: Mountain View, CA
    Posted: 1 week ago
    Description: Develop ML models...
    URL: https://linkedin.com/jobs/view/456
    JOB_END

    List {num_jobs} jobs total. Start collecting job information NOW.
    """

    agent_llm = BrowserUseChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.1,
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    # Agent with simplified action settings
    agent = Agent(
        task=task,
        llm=agent_llm,
        browser_session=browser_session,
        max_actions=40,  # Provide extra room for navigation and backtracking
        max_failures=6   # Allow more retries when recovering from navigation issues
    )

    try:
        logger.info("Starting browser session...")
        await browser_session.start()

        logger.info("Navigating to LinkedIn login page...")
        await browser_session.navigate_to("https://www.linkedin.com/login")

        logger.info("Waiting for manual LinkedIn login...")
        try:
            await wait_for_linkedin_login(browser_session)
        except TimeoutError as timeout_error:
            error_msg = str(timeout_error)
            logger.error(error_msg)
            st.error(error_msg)
            return None

        logger.info("Login detected. Navigating to job search page...")
        await browser_session.navigate_to(job_search_url)
        await asyncio.sleep(3)

        # Run the agent after login is confirmed
        logger.info("Running agent...")
        result = await agent.run()

        logger.info("Agent completed. Result type: %s", type(result))

        # Extract content from browser_use result
        content = ""
        logger.debug("Result has all_results: %s", hasattr(result, 'all_results'))

        # Multiple extraction strategies
        action_results = []

        if hasattr(result, 'all_results') and getattr(result, 'all_results'):
            action_results = list(result.all_results)
            logger.debug("Found %s action results via all_results attribute", len(action_results))
        else:
            maybe_action_results = getattr(result, 'action_results', None)
            try:
                if callable(maybe_action_results):
                    action_results = maybe_action_results() or []
                    logger.debug(
                        "Found %s action results via action_results() method",
                        len(action_results)
                    )
            except Exception as action_error:  # pragma: no cover - guard against SDK changes
                logger.warning("Error retrieving action results: %s", action_error)

        if action_results:
            for i, action_result in enumerate(action_results):
                logger.debug("Action result %s type: %s", i, type(action_result))
                logger.debug(
                    "Action result %s attributes: %s",
                    i,
                    [attr for attr in dir(action_result) if not attr.startswith('_')]
                )

                # Check for attachments
                if hasattr(action_result, 'attachments') and action_result.attachments:
                    logger.debug("Found %s attachments in action %s", len(action_result.attachments), i)
                    for j, attachment in enumerate(action_result.attachments):
                        logger.debug("Attachment %s type: %s", j, type(attachment))
                        logger.debug(
                            "Attachment %s attributes: %s",
                            j,
                            [attr for attr in dir(attachment) if not attr.startswith('_')]
                        )

                        # Try different path attributes
                        path = None
                        if hasattr(attachment, 'path') and attachment.path:
                            path = attachment.path
                        elif hasattr(attachment, 'file_path') and attachment.file_path:
                            path = attachment.file_path
                        elif hasattr(attachment, 'filepath') and attachment.filepath:
                            path = attachment.filepath

                        if path:
                            try:
                                logger.info("Reading attachment: %s", path)
                                with open(path, 'r', encoding='utf-8') as f:
                                    file_content = f.read()
                                    content += file_content + "\n"
                                    logger.debug("Read %s characters from attachment", len(file_content))
                            except Exception as file_error:
                                logger.warning("Error reading attachment %s: %s", path, file_error)

                # Check for extracted_content
                if hasattr(action_result, 'extracted_content') and action_result.extracted_content:
                    extracted = str(action_result.extracted_content)
                    content += extracted + "\n"
                    logger.debug("Added %s characters from extracted_content", len(extracted))

                # Check for other content fields
                for attr in ['content', 'result', 'output', 'text']:
                    if hasattr(action_result, attr):
                        attr_value = getattr(action_result, attr)
                        if attr_value and isinstance(attr_value, str):
                            content += attr_value + "\n"
                            logger.debug("Added %s characters from %s", len(attr_value), attr)

        # Check result itself for content
        attributes_extracted_content = getattr(result, 'extracted_content', None)
        if attributes_extracted_content and not callable(attributes_extracted_content):
            if isinstance(attributes_extracted_content, (list, tuple, set)):
                joined_attr_content = "\n".join(str(chunk) for chunk in attributes_extracted_content if chunk)
                content += joined_attr_content + "\n"
                logger.debug(
                    "Added %s chunks from result.extracted_content attribute",
                    len(attributes_extracted_content)
                )
            else:
                content += str(attributes_extracted_content) + "\n"
                logger.debug("Added 1 chunk from result.extracted_content attribute")

        maybe_extracted_content = getattr(result, 'extracted_content', None)
        if callable(maybe_extracted_content):
            try:
                extracted_chunks = maybe_extracted_content() or []
                if extracted_chunks:
                    content += "\n".join(str(chunk) for chunk in extracted_chunks)
                    logger.debug(
                        "Added %s extracted content chunks via method", len(extracted_chunks)
                    )
            except Exception as extracted_error:  # pragma: no cover - defensive
                logger.warning("Error retrieving extracted content: %s", extracted_error)

        # Check if result has a final response or message
        if hasattr(result, 'final_response') and result.final_response:
            content += str(result.final_response) + "\n"
            logger.debug("Added content from result.final_response")

        maybe_final_result = getattr(result, 'final_result', None)
        if callable(maybe_final_result):
            try:
                final_result = maybe_final_result()
                if final_result and final_result not in content:
                    content += str(final_result) + "\n"
                    logger.debug("Added content from result.final_result()")
            except Exception as final_error:  # pragma: no cover - defensive
                logger.warning("Error retrieving final result: %s", final_error)

        # Also check the last action result for the agent's final response
        if action_results:
            last_result = action_results[-1]
            if hasattr(last_result, 'extracted_content') and last_result.extracted_content:
                last_content = str(last_result.extracted_content)
                if 'JOB_START' in last_content and last_content not in content:
                    content += last_content + "\n"
                    logger.debug("Added job data from last action result: %s characters", len(last_content))

        # Fallback to string conversion if no content found
        if not content:
            content = str(result)
            logger.debug("Using fallback string conversion: %s characters", len(content))

        # Extract job data from the full result string if needed
        if not content or 'JOB_START' not in content:
            result_str = str(result)
            if 'JOB_START' in result_str:
                content = result_str
                logger.debug("Found job data in full result string: %s characters", len(content))

        logger.info("Final extracted content length: %s", len(content) if content else 0)
        logger.debug("Content preview: %s", content[:500] if content else 'None')

        return content

    except Exception as e:
        error_msg = f"Error during scraping: {str(e)}"
        logger.exception("Error during scraping: %s", e)
        st.error(error_msg)
        return None
    finally:
        try:
            await browser_session.kill()
            logger.info("Browser session closed")
        except Exception as cleanup_error:
            logger.warning("Cleanup error: %s", cleanup_error)

PLACEHOLDER_TOKENS = {
    '[exact job title]',
    '[company name]',
    '[location]',
    '[posted time]',
    '[full description text]',
    '[full linkedin job url]'
}

MAX_DESCRIPTION_CHARS = 4000

JOB_URL_REGEX = re.compile(
    r"https://www\.linkedin\.com/(?:jobs|jobs-guest)/view/[^\s\"'>]+"
)


def _value_is_placeholder(value: Optional[str]) -> bool:
    if not value:
        return True

    lowered = value.lower()
    if '[' in lowered and any(token in lowered for token in PLACEHOLDER_TOKENS):
        return True

    return False


def _parse_job_block(block: str) -> Optional[Dict]:
    lines = [line.strip() for line in block.splitlines() if line.strip()]
    if not lines:
        return None

    job: Dict[str, str] = {}
    description_lines: List[str] = []
    capturing_description = False

    for line in lines:
        lowered = line.lower()

        if lowered.startswith('job_start') or lowered.startswith('job_end'):
            continue

        if lowered.startswith('title') and ':' in line:
            job['title'] = line.split(':', 1)[1].strip()
            capturing_description = False
            continue

        if lowered.startswith('company') and ':' in line:
            job['company'] = line.split(':', 1)[1].strip()
            capturing_description = False
            continue

        if lowered.startswith('location') and ':' in line:
            job['location'] = line.split(':', 1)[1].strip()
            capturing_description = False
            continue

        if lowered.startswith('posted') and ':' in line:
            job['posted'] = line.split(':', 1)[1].strip()
            capturing_description = False
            continue

        if lowered.startswith('description') and ':' in line:
            description_lines = [line.split(':', 1)[1].strip()]
            capturing_description = True
            continue

        if lowered.startswith('url') and ':' in line:
            job['link'] = line.split(':', 1)[1].strip()
            capturing_description = False
            continue

        if capturing_description:
            description_lines.append(line)

    if description_lines:
        description_text = ' '.join(description_lines).strip()
        job['description'] = description_text[:MAX_DESCRIPTION_CHARS]
    else:
        job.setdefault('description', '')

    job.setdefault('posted', 'Recently posted')

    essential_fields = ['title', 'company', 'location']
    for field in essential_fields:
        if _value_is_placeholder(job.get(field)):
            return None

    if not job.get('title') or not job.get('company'):
        return None

    if not job.get('link') and job.get('title'):
        url_match = JOB_URL_REGEX.search(block)

        if url_match:
            job['link'] = url_match.group(0)
        else:
            job['link'] = f"https://www.linkedin.com/jobs/search/?keywords={job['title'].replace(' ', '%20')}"

    if job.get('link') and 'jobs/search' in job['link']:
        fallback_match = JOB_URL_REGEX.search(block)
        if fallback_match:
            job['link'] = fallback_match.group(0)

    job['raw_content'] = block[:1000]
    return job


def parse_linkedin_jobs(raw_content, max_jobs: int = None) -> List[Dict]:
    """Parse the raw scraped content into structured job data and drop placeholders."""

    logger = logging.getLogger("linkedin_agent")

    logger.debug(
        "parse_linkedin_jobs called with content length: %s",
        len(raw_content) if raw_content else 0
    )
    logger.debug("max_jobs: %s", max_jobs)

    if not raw_content:
        logger.debug("No raw content provided")
        return []

    if not isinstance(raw_content, str):
        raw_content = str(raw_content)

    logger.debug("Raw content preview: %s...", raw_content[:300])

    # Look for JOB_START...JOB_END blocks
    blocks = re.findall(r'JOB_START(.*?)JOB_END', raw_content, flags=re.DOTALL)
    logger.debug("Found %s JOB_START...JOB_END blocks", len(blocks))

    if not blocks:
        logger.debug("No JOB_START blocks found, trying fallback parsing")
        rough_blocks = re.split(r'\n\s*\n', raw_content)
        blocks = [block for block in rough_blocks if 'Title:' in block and 'Company:' in block]
        logger.debug("Fallback found %s blocks with Title: and Company:", len(blocks))

    parsed: List[Dict] = []
    seen_links: set[str] = set()

    for i, block in enumerate(blocks):
        logger.debug("Processing block %s/%s", i + 1, len(blocks))
        logger.debug("Block preview: %s...", block[:100])

        # Stop if we've reached the maximum number of jobs
        if max_jobs and len(parsed) >= max_jobs:
            logger.debug("Reached max_jobs limit (%s), stopping", max_jobs)
            break

        job = _parse_job_block(block)
        if not job:
            logger.debug("Block %s failed to parse", i + 1)
            continue

        logger.debug(
            "Parsed job: %s at %s",
            job.get('title', 'No title'),
            job.get('company', 'No company')
        )

        link = job.get('link')
        if link and link in seen_links:
            logger.debug("Duplicate link found, skipping: %s", link)
            continue

        seen_links.add(link)
        parsed.append(job)

        # Stop if we've reached the maximum after adding this job
        if max_jobs and len(parsed) >= max_jobs:
            logger.debug(
                "Reached max_jobs limit (%s) after adding job, stopping",
                max_jobs
            )
            break

    if parsed:
        logger.debug("Attempting to backfill canonical job URLs if missing")
        all_links_in_content = [link for link in JOB_URL_REGEX.findall(raw_content or "")]
        logger.debug("Found %s candidate job URLs in raw content", len(all_links_in_content))

        allocated_links: set[str] = {
            job['link']
            for job in parsed
            if isinstance(job.get('link'), str) and 'jobs/view' in job['link']
        }

        for job in parsed:
            needs_link = not job.get('link') or 'jobs/search' in job['link'] or _value_is_placeholder(job.get('link'))
            if not needs_link:
                continue

            replacement = None
            for candidate in all_links_in_content:
                if candidate not in allocated_links:
                    replacement = candidate
                    break

            if replacement:
                logger.debug(
                    "Updated job '%s' with canonical URL %s",
                    job.get('title', 'Unknown title'),
                    replacement
                )
                job['link'] = replacement
                allocated_links.add(replacement)
            else:
                logger.debug(
                    "No canonical URL available for job '%s'; keeping fallback %s",
                    job.get('title', 'Unknown title'),
                    job.get('link')
                )

    logger.debug("Final parsed jobs count: %s", len(parsed))
    return parsed

def rank_jobs_by_resume(jobs: List[Dict], resume_text: str) -> List[Tuple[Dict, float, str]]:
    """Rank jobs based on resume match using OpenAI one posting at a time."""

    if not jobs:
        return []

    llm = LangChainChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0.1,
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    jobs_to_rank = jobs[:min(50, len(jobs))]

    progress_bar = st.progress(0.0)
    status_placeholder = st.empty()

    ranked_jobs: List[Tuple[Dict, float, str]] = []

    truncated_resume = resume_text[:3000]

    for idx, job in enumerate(jobs_to_rank, start=1):
        status_placeholder.info(
            f"Scoring job {idx}/{len(jobs_to_rank)}: {job.get('title', 'Unknown Title')}"
        )

        job_summary = (
            f"Title: {job.get('title', 'N/A')}\n"
            f"Company: {job.get('company', 'N/A')}\n"
            f"Location: {job.get('location', 'N/A')}\n"
            f"Posted: {job.get('posted', 'Recently')}\n"
            f"Description: {job.get('description', 'No description available')[:600]}"
        )

        prompt = f"""
        You are an AI assistant expert in candidate-job matching.

        Candidate resume:
        {truncated_resume}

        Job details:
        {job_summary}

        Evaluate how well this candidate matches the job. Respond ONLY with JSON using this schema:
        {{"score": <integer 0-100>, "explanation": "short reasoning (2 sentences max)"}}

        Focus on alignment between key skills, experience seniority, and industry/domain. Penalize large gaps.
        """

        try:
            response = llm.invoke(prompt)
            response_text = getattr(response, 'content', str(response)).strip()
            response_text = re.sub(r'^```json\s*', '', response_text)
            response_text = re.sub(r'```$', '', response_text)

            data = json.loads(response_text)
            score = float(data.get('score', 0))
            explanation = str(data.get('explanation', '')).strip()

        except Exception as error:
            st.warning(
                f"Ranking error for job '{job.get('title', 'Unknown Title')}': {error}"
            )
            score = 50.0
            explanation = "Could not evaluate precisely; default score applied."

        ranked_jobs.append((job, max(0.0, min(score, 100.0)), explanation))

        progress_bar.progress(idx / len(jobs_to_rank))

    status_placeholder.empty()

    ranked_jobs.sort(key=lambda item: item[1], reverse=True)

    remaining_jobs = [job for job in jobs if job not in [rj[0] for rj in ranked_jobs]]
    for job in remaining_jobs:
        ranked_jobs.append((job, 50.0, "Not analyzed in detail"))

    return ranked_jobs

def main():
    if 'scraped_jobs' not in st.session_state:
        st.session_state.scraped_jobs = None
    if 'ranked_jobs' not in st.session_state:
        st.session_state.ranked_jobs = None
    if 'resume_text' not in st.session_state:
        st.session_state.resume_text = None

    log_expander = st.expander("🤖 Agent Logs", expanded=False)
    with log_expander:
        col_log_left, col_log_right = st.columns([3, 1])
        with col_log_right:
            if st.button("Clear logs", key="clear_agent_logs", use_container_width=True):
                st.session_state[LOG_STATE_KEY] = []
        log_placeholder = col_log_left.empty()

    setup_streamlit_logging(log_placeholder)

    # Ensure the latest logs render on the current run
    existing_log_lines = st.session_state.get(LOG_STATE_KEY, [])
    if existing_log_lines:
        log_placeholder.text("\n".join(existing_log_lines))
    else:
        log_placeholder.text("No agent logs yet. Run a scrape to populate this panel.")

    col1, col2 = st.columns([1, 2])

    with col1:
        st.header("📄 Upload Resume")

        resume_file = st.file_uploader(
            "Choose your resume file",
            type=['txt', 'pdf'],
            help="Upload your resume in TXT or PDF format"
        )

        resume_text = st.text_area(
            "Or paste your resume text here:",
            height=300,
            placeholder="Paste your resume content here..."
        )

        if resume_file:
            if resume_file.type == "text/plain":
                resume_text = str(resume_file.read(), "utf-8")
                st.success("✅ Resume uploaded successfully!")
            elif resume_file.type == "application/pdf":
                try:
                    # Read PDF file
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(resume_file.read()))
                    resume_text = ""

                    # Extract text from all pages
                    for page_num in range(len(pdf_reader.pages)):
                        page = pdf_reader.pages[page_num]
                        resume_text += page.extract_text() + "\n"

                    st.success(f"✅ PDF resume uploaded successfully! Extracted {len(pdf_reader.pages)} pages.")
                except Exception as e:
                    st.error(f"❌ Error reading PDF: {str(e)}")
                    st.info("Please try uploading a TXT file or paste your resume text below.")

        if resume_text:
            st.session_state.resume_text = resume_text
            st.success(f"Resume loaded: {len(resume_text)} characters")

    with col2:
        st.header("🔍 LinkedIn AI Jobs")

        with st.expander("Reuse previously scraped jobs", expanded=False):
            uploaded_jobs_file = st.file_uploader(
                "Upload saved LinkedIn jobs JSON",
                type=["json"],
                accept_multiple_files=False,
                key="uploaded_jobs_file",
                help="Load jobs from a previous scraping session (downloaded from this app)."
            )

            if uploaded_jobs_file is not None:
                try:
                    loaded_jobs = json.load(uploaded_jobs_file)
                    if isinstance(loaded_jobs, list) and all(isinstance(item, dict) for item in loaded_jobs):
                        st.session_state.scraped_jobs = loaded_jobs
                        st.success(f"Loaded {len(loaded_jobs)} jobs from uploaded file.")
                    else:
                        st.error("Uploaded file must be a JSON array of job objects.")
                except Exception as load_error:
                    st.error(f"Could not load jobs file: {load_error}")

        # Configuration options
        st.subheader("⚙️ Scraping Configuration")

        col2a, col2b = st.columns([1, 1])
        with col2a:
            num_jobs = st.number_input(
                "Number of jobs to scrape",
                min_value=10,
                max_value=20,
                value=10,
                step=2,
                help="Choose how many jobs to scrape (10-20)"
            )
        with col2b:
            st.caption(f"📊 Will scrape up to {num_jobs} jobs")

        # URL configuration
        st.markdown("**LinkedIn Search URL:**")
        url_option = st.selectbox(
            "Choose job search type:",
            options=[
                "AI/GenAI Jobs",
                "Software Engineering Jobs",
                "Data Science Jobs",
                "Product Management Jobs",
                "Custom URL"
            ],
            index=0,
            help="Select the type of jobs to search for"
        )

        # Map selections to URLs
        url_mapping = {
            "AI/GenAI Jobs": "https://www.linkedin.com/jobs/collections/gen-ai/",
            "Software Engineering Jobs": "https://www.linkedin.com/jobs/search/?keywords=software%20engineer",
            "Data Science Jobs": "https://www.linkedin.com/jobs/search/?keywords=data%20scientist",
            "Product Management Jobs": "https://www.linkedin.com/jobs/search/?keywords=product%20manager",
            "Custom URL": ""
        }

        if url_option == "Custom URL":
            job_search_url = st.text_input(
                "Enter custom LinkedIn jobs URL:",
                value="https://www.linkedin.com/jobs/search/?keywords=",
                help="Enter any LinkedIn jobs search URL"
            )
        else:
            job_search_url = url_mapping[url_option]
            st.code(job_search_url, language="text")

        st.markdown("---")

        if st.button("🔍 Scrape LinkedIn AI Jobs", type="primary", disabled=not resume_text):
            st.info("📝 **Instructions for LinkedIn Login:**")
            st.markdown("""
            1. A browser window will open shortly
            2. **Log in to LinkedIn manually** in the browser window
            3. Once logged in, the scraper will automatically continue
            4. The process may take a few minutes to complete
            """)
            with st.spinner("Opening browser for LinkedIn login... Please log in manually when the browser opens."):
                start_time = time.time()

                raw_content = asyncio.run(scrape_linkedin_jobs(resume_text, num_jobs, job_search_url))

                if raw_content:
                    st.session_state.scraped_jobs = parse_linkedin_jobs(raw_content, num_jobs)

                    elapsed = time.time() - start_time
                    st.success(f" Found {len(st.session_state.scraped_jobs)} jobs in {elapsed:.1f} seconds")

                    with st.expander("View Raw Scraped Content"):
                        # Safely display content
                        if isinstance(raw_content, str):
                            st.text(raw_content[:5000])
                        else:
                            st.text(f"Content type: {type(raw_content)}\nContent: {str(raw_content)[:5000]}")
                else:
                    st.error("Failed to scrape jobs. Please try again.")

        if st.session_state.scraped_jobs:
            st.info(f"===== {len(st.session_state.scraped_jobs)} jobs scraped from LinkedIn")

            st.download_button(
                "💾 Download scraped jobs as JSON",
                data=json.dumps(st.session_state.scraped_jobs, indent=2),
                file_name="linkedin_jobs.json",
                mime="application/json",
                use_container_width=True
            )

            if st.button("  Rank Jobs by Resume Match"):
                with st.spinner("Analyzing job matches with your resume..."):
                    st.session_state.ranked_jobs = rank_jobs_by_resume(
                        st.session_state.scraped_jobs,
                        st.session_state.resume_text
                    )
                    st.success("Jobs ranked successfully!")

    if st.session_state.ranked_jobs:
        st.header("  Top Job Matches")
        st.markdown("Jobs are ranked by compatibility with your resume (highest match first)")

        tabs = st.tabs(["  Top 10 Matches", "===== All Jobs", "===== Match Distribution"])

        with tabs[0]:
            for idx, (job, score, explanation) in enumerate(st.session_state.ranked_jobs[:10], 1):
                with st.expander(f"#{idx} - {job.get('title', 'Unknown Title')} @ {job.get('company', 'Unknown Company')} (Score: {score}/100)"):
                    col1, col2 = st.columns([3, 1])

                    with col1:
                        st.markdown(f"**Company:** {job.get('company', 'Not specified')}")
                        st.markdown(f"**Location:** {job.get('location', 'Not specified')}")
                        st.markdown(f"**Posted:** {job.get('posted', 'Recently')}")

                        if job.get('description'):
                            st.markdown("**Description Preview:**")
                            st.text(job.get('description')[:500])

                        st.markdown(f"**Match Analysis:**")
                        st.info(explanation)

                    with col2:
                        st.metric("Match Score", f"{score}%")
                        if job.get('link'):
                            st.link_button("View Job", job.get('link'), use_container_width=True)

        with tabs[1]:
            jobs_data = []
            for job, score, _ in st.session_state.ranked_jobs:
                jobs_data.append({
                    'Score': f"{score}%",
                    'Title': job.get('title', 'N/A'),
                    'Company': job.get('company', 'N/A'),
                    'Location': job.get('location', 'N/A'),
                    'Posted': job.get('posted', 'N/A')
                })

            st.dataframe(jobs_data, use_container_width=True, height=600)

        with tabs[2]:
            import pandas as pd

            scores = [score for _, score, _ in st.session_state.ranked_jobs]

            score_ranges = {
                '90-100': len([s for s in scores if s >= 90]),
                '80-89': len([s for s in scores if 80 <= s < 90]),
                '70-79': len([s for s in scores if 70 <= s < 80]),
                '60-69': len([s for s in scores if 60 <= s < 70]),
                '50-59': len([s for s in scores if 50 <= s < 60]),
                'Below 50': len([s for s in scores if s < 50])
            }

            chart_data = pd.DataFrame.from_dict(score_ranges, orient='index', columns=['Count'])
            st.bar_chart(chart_data)

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Average Score", f"{sum(scores)/len(scores):.1f}%")
            with col2:
                st.metric("Top Score", f"{max(scores)}%")
            with col3:
                st.metric("Jobs > 70% Match", len([s for s in scores if s >= 70]))

if __name__ == "__main__":
    main()
