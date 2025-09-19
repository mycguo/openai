import streamlit as st
import asyncio
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

st.set_page_config(
    page_title="LinkedIn AI Jobs Matcher",
    page_icon="💼",
    layout="wide"
)

st.title("💼 LinkedIn AI Jobs Matcher")
st.markdown("Upload your resume and find the best matching AI/GenAI jobs from LinkedIn")

async def scrape_linkedin_jobs(resume_text: str, num_jobs: int = 100, job_search_url: str = "https://www.linkedin.com/jobs/collections/gen-ai/") -> str:
    """Scrape LinkedIn AI jobs using browser-use"""

    async def wait_for_linkedin_login(browser_session: BrowserSession, timeout: int = 300, poll_interval: float = 3.0) -> str:
        """Poll the active page until LinkedIn login appears complete."""

        start = time.monotonic()
        last_url = ""

        while time.monotonic() - start < timeout:
            try:
                current_url = await browser_session.get_current_page_url()
            except Exception as url_error:  # pragma: no cover - defensive logging
                print(f"Waiting for LinkedIn login: unable to read current URL ({url_error})")
                current_url = ""

            normalized_url = current_url.lower() if current_url else ""

            if normalized_url and "linkedin.com" in normalized_url and not any(
                blocked in normalized_url for blocked in ["/login", "checkpoint", "authwall", "uas/login"]
            ):
                print(f"Detected LinkedIn login completion at {current_url}")
                return current_url

            if current_url and current_url != last_url:
                print(f"Still waiting for LinkedIn login... current page: {current_url}")
                last_url = current_url

            await asyncio.sleep(poll_interval)

        raise TimeoutError("LinkedIn login was not detected within the allotted time (5 minutes).")

    # Try with explicit browser profile to avoid CDP issues
    browser_profile = BrowserProfile(
        headless=False,
        driver_type="chromium",
        keep_alive=True
    )
    browser_session = BrowserSession(browser_profile=browser_profile)

    task = f"""
    Look at this LinkedIn jobs page and list {num_jobs} job postings you can see.

    RULES:
    - Maximum 5 actions total (scroll, click)
    - Do NOT use extract_structured_data
    - Do NOT create files
    - After 5 actions, STOP and provide your answer

    STEPS:
    1. Look at job titles and companies visible on the page
    2. Scroll ONCE to see more jobs if needed
    3. Click on 1-2 jobs to see details if needed
    4. IMMEDIATELY provide your final answer with job information

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
        max_actions=25,  # Enough for navigation + scrolling + reading
        max_failures=2   # Fewer retries to prevent extract_structured_data loops
    )

    try:
        print("Starting browser session...")
        await browser_session.start()

        print("Navigating to LinkedIn login page...")
        await browser_session.navigate_to("https://www.linkedin.com/login")

        print("Waiting for manual LinkedIn login...")
        try:
            await wait_for_linkedin_login(browser_session)
        except TimeoutError as timeout_error:
            error_msg = str(timeout_error)
            print(error_msg)
            st.error(error_msg)
            return None

        print("Login detected. Navigating to job search page...")
        await browser_session.navigate_to(job_search_url)
        await asyncio.sleep(3)

        # Run the agent after login is confirmed
        print("Running agent...")
        result = await agent.run()

        print(f"Agent completed. Result type: {type(result)}")

        # Extract content from browser_use result
        content = ""
        print(f"DEBUG: Result has all_results: {hasattr(result, 'all_results')}")

        # Multiple extraction strategies
        if hasattr(result, 'all_results') and result.all_results:
            print(f"DEBUG: Found {len(result.all_results)} action results")

            for i, action_result in enumerate(result.all_results):
                print(f"DEBUG: Action result {i} type: {type(action_result)}")
                print(f"DEBUG: Action result {i} attributes: {[attr for attr in dir(action_result) if not attr.startswith('_')]}")

                # Check for attachments
                if hasattr(action_result, 'attachments') and action_result.attachments:
                    print(f"Found {len(action_result.attachments)} attachments in action {i}")
                    for j, attachment in enumerate(action_result.attachments):
                        print(f"DEBUG: Attachment {j} type: {type(attachment)}")
                        print(f"DEBUG: Attachment {j} attributes: {[attr for attr in dir(attachment) if not attr.startswith('_')]}")

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
                                print(f"Reading attachment: {path}")
                                with open(path, 'r', encoding='utf-8') as f:
                                    file_content = f.read()
                                    content += file_content + "\n"
                                    print(f"Read {len(file_content)} characters from attachment")
                            except Exception as file_error:
                                print(f"Error reading attachment {path}: {file_error}")

                # Check for extracted_content
                if hasattr(action_result, 'extracted_content') and action_result.extracted_content:
                    extracted = str(action_result.extracted_content)
                    content += extracted + "\n"
                    print(f"Added {len(extracted)} characters from extracted_content")

                # Check for other content fields
                for attr in ['content', 'result', 'output', 'text']:
                    if hasattr(action_result, attr):
                        attr_value = getattr(action_result, attr)
                        if attr_value and isinstance(attr_value, str):
                            content += attr_value + "\n"
                            print(f"Added {len(attr_value)} characters from {attr}")

        # Check result itself for content
        if hasattr(result, 'extracted_content') and result.extracted_content:
            content += str(result.extracted_content) + "\n"
            print(f"Added content from result.extracted_content")

        # Check if result has a final response or message
        if hasattr(result, 'final_response') and result.final_response:
            content += str(result.final_response) + "\n"
            print(f"Added content from result.final_response")

        # Also check the last action result for the agent's final response
        if hasattr(result, 'all_results') and result.all_results:
            last_result = result.all_results[-1]
            if hasattr(last_result, 'extracted_content') and last_result.extracted_content:
                last_content = str(last_result.extracted_content)
                if 'JOB_START' in last_content and last_content not in content:
                    content += last_content + "\n"
                    print(f"Added job data from last action result: {len(last_content)} characters")

        # Fallback to string conversion if no content found
        if not content:
            content = str(result)
            print(f"Using fallback string conversion: {len(content)} characters")

        # Extract job data from the full result string if needed
        if not content or 'JOB_START' not in content:
            result_str = str(result)
            if 'JOB_START' in result_str:
                content = result_str
                print(f"Found job data in full result string: {len(content)} characters")

        print(f"Final extracted content length: {len(content) if content else 0}")
        print(f"Content preview: {content[:500] if content else 'None'}")

        return content

    except Exception as e:
        error_msg = f"Error during scraping: {str(e)}"
        print(error_msg)
        st.error(error_msg)
        return None
    finally:
        try:
            await browser_session.kill()
            print("Browser session closed")
        except Exception as cleanup_error:
            print(f"Cleanup error: {cleanup_error}")

PLACEHOLDER_TOKENS = {
    '[exact job title]',
    '[company name]',
    '[location]',
    '[posted time]',
    '[full description text]',
    '[full linkedin job url]'
}

MAX_DESCRIPTION_CHARS = 4000


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
        job['link'] = f"https://www.linkedin.com/jobs/search/?keywords={job['title'].replace(' ', '%20')}"

    job['raw_content'] = block[:1000]
    return job


def parse_linkedin_jobs(raw_content, max_jobs: int = None) -> List[Dict]:
    """Parse the raw scraped content into structured job data and drop placeholders."""

    print(f"DEBUG: parse_linkedin_jobs called with content length: {len(raw_content) if raw_content else 0}")
    print(f"DEBUG: max_jobs: {max_jobs}")

    if not raw_content:
        print("DEBUG: No raw content provided")
        return []

    if not isinstance(raw_content, str):
        raw_content = str(raw_content)

    print(f"DEBUG: Raw content preview: {raw_content[:300]}...")

    # Look for JOB_START...JOB_END blocks
    blocks = re.findall(r'JOB_START(.*?)JOB_END', raw_content, flags=re.DOTALL)
    print(f"DEBUG: Found {len(blocks)} JOB_START...JOB_END blocks")

    if not blocks:
        print("DEBUG: No JOB_START blocks found, trying fallback parsing")
        rough_blocks = re.split(r'\n\s*\n', raw_content)
        blocks = [block for block in rough_blocks if 'Title:' in block and 'Company:' in block]
        print(f"DEBUG: Fallback found {len(blocks)} blocks with Title: and Company:")

    parsed: List[Dict] = []
    seen_links: set[str] = set()

    for i, block in enumerate(blocks):
        print(f"DEBUG: Processing block {i+1}/{len(blocks)}")
        print(f"DEBUG: Block preview: {block[:100]}...")

        # Stop if we've reached the maximum number of jobs
        if max_jobs and len(parsed) >= max_jobs:
            print(f"DEBUG: Reached max_jobs limit ({max_jobs}), stopping")
            break

        job = _parse_job_block(block)
        if not job:
            print(f"DEBUG: Block {i+1} failed to parse")
            continue

        print(f"DEBUG: Parsed job: {job.get('title', 'No title')} at {job.get('company', 'No company')}")

        link = job.get('link')
        if link and link in seen_links:
            print(f"DEBUG: Duplicate link found, skipping: {link}")
            continue

        seen_links.add(link)
        parsed.append(job)

        # Stop if we've reached the maximum after adding this job
        if max_jobs and len(parsed) >= max_jobs:
            print(f"DEBUG: Reached max_jobs limit ({max_jobs}) after adding job, stopping")
            break

    print(f"DEBUG: Final parsed jobs count: {len(parsed)}")
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
                max_value=500,
                value=10,
                step=10,
                help="Choose how many jobs to scrape (10-500)"
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
