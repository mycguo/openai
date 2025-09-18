import streamlit as st
import asyncio
from browser_use import Agent
from browser_use import BrowserSession, BrowserProfile
from browser_use.llm.openai.chat import ChatOpenAI as BrowserUseChatOpenAI
from langchain_openai import ChatOpenAI as LangChainChatOpenAI
import re
from datetime import datetime
import json
from typing import List, Dict, Tuple
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

async def scrape_linkedin_jobs(resume_text: str, num_jobs: int = 100) -> str:
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
    job_search_url = "https://www.linkedin.com/jobs/collections/gen-ai/"

    task = f"""
    Extract LinkedIn job postings in a specific format. You are already logged in.

    1. Navigate to {job_search_url} if not already there
    2. Scroll down several times to load {num_jobs} job listings
    3. For each job listing visible on the page, extract the following information and format it EXACTLY like this:

    JOB_START
    Title: [Exact job title from the heading]
    Company: [Company name]
    Location: [Location text like "San Francisco, CA" or "Remote"]
    Posted: [Time posted like "2 weeks ago" or "3 days ago"]
    Description: [The job description]
    URL: [Full LinkedIn job URL - look for href links to /jobs/view/]
    JOB_END

    Example format:
    JOB_START
    Title: Tech Lead Manager, Safeguards ML Infrastructure
    Company: Anthropic
    Location: San Francisco, CA
    Posted: 2 weeks ago
    Description: Anthropic's mission is to create reliable, interpretable, and steerable AI systems. We want AI to be safe and beneficial for our users and for society as a whole. Our team is a quickly growing group of committed researchers, engineers, policy experts...
    URL: https://www.linkedin.com/jobs/view/4255922706
    JOB_END

    Extract exactly {num_jobs} jobs in this format. Be precise with the formatting.
    """

    agent_llm = BrowserUseChatOpenAI(
        model="gpt-4.1-mini",
        temperature=0.1,
        api_key=st.secrets["OPENAI_API_KEY"]
    )

    # Agent with basic stability settings
    agent = Agent(
        task=task,
        llm=agent_llm,
        browser_session=browser_session,
        max_actions=20,
        max_failures=5
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

        # Simple string extraction - try the most direct approach first
        content = str(result)

        print(f"Extracted content length: {len(content) if content else 0}")
        print(f"Content preview: {content[:200] if content else 'None'}")

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

def parse_linkedin_jobs(raw_content) -> List[Dict]:
    """Parse the raw scraped content into structured job data"""
    jobs = []

    if not raw_content:
        return jobs

    # Ensure raw_content is a string
    if not isinstance(raw_content, str):
        raw_content = str(raw_content)

    # Look for the new structured format first (JOB_START...JOB_END)
    job_blocks = re.findall(r'JOB_START(.*?)JOB_END', raw_content, re.DOTALL)

    if job_blocks:
        # Parse the new structured format
        for block in job_blocks:
            job = {}

            # Extract each field using the structured format
            title_match = re.search(r'Title:\s*(.+?)(?:\n|$)', block)
            company_match = re.search(r'Company:\s*(.+?)(?:\n|$)', block)
            location_match = re.search(r'Location:\s*(.+?)(?:\n|$)', block)
            posted_match = re.search(r'Posted:\s*(.+?)(?:\n|$)', block)
            description_match = re.search(r'Description:\s*(.+?)(?:\n(?:URL|$))', block, re.DOTALL)
            url_match = re.search(r'URL:\s*(https?://[^\s\n]+)', block)

            if title_match:
                job['title'] = title_match.group(1).strip()
                job['company'] = company_match.group(1).strip() if company_match else 'Company not specified'
                job['location'] = location_match.group(1).strip() if location_match else 'Location not specified'
                job['posted'] = posted_match.group(1).strip() if posted_match else 'Recently posted'
                job['description'] = description_match.group(1).strip()[:500] if description_match else ''
                job['link'] = url_match.group(1).strip() if url_match else f"https://www.linkedin.com/jobs/search/?keywords={job.get('title', '').replace(' ', '%20')}"
                job['raw_content'] = block[:1000]

                jobs.append(job)

    # Fallback to old parsing logic if new format not found
    else:
        job_blocks = re.split(r'(?=(?:^|\n)(?:\d+\.|Job \d+|"))', raw_content)

        for block in job_blocks:
            if len(block.strip()) < 50:
                continue

            job = {}

            title_match = re.search(r'(?:Title|Position|Role):\s*(.+?)(?:\n|$)', block, re.IGNORECASE)
            if not title_match:
                title_match = re.search(r'^([^:\n]+(?:Engineer|Developer|Scientist|Manager|Analyst|Designer|Lead|Architect|Specialist|Consultant)[^:\n]*)', block, re.IGNORECASE | re.MULTILINE)

            company_match = re.search(r'(?:Company|Organization|Employer):\s*(.+?)(?:\n|$)', block, re.IGNORECASE)
            if not company_match:
                company_match = re.search(r'(?:at|@)\s+([A-Z][A-Za-z0-9\s&,.\-]+?)(?:\n|$|"|\||Location)', block)

            location_match = re.search(r'(?:Location|Where):\s*(.+?)(?:\n|$)', block, re.IGNORECASE)
            if not location_match:
                location_match = re.search(r'(?:in|📍)\s+([A-Za-z\s,]+(?:Remote|Hybrid|On-site)?)', block, re.IGNORECASE)

            posted_match = re.search(r'(?:Posted|Published|Added):\s*(.+?)(?:\n|$)', block, re.IGNORECASE)
            if not posted_match:
                posted_match = re.search(r'(\d+\s*(?:hour|day|week|month)s?\s*ago)', block, re.IGNORECASE)

            description_match = re.search(r'(?:Description|About|Overview|Summary):\s*(.+?)(?:\n\n|$)', block, re.IGNORECASE | re.DOTALL)
            if not description_match:
                lines = block.split('\n')
                desc_lines = [l for l in lines if len(l) > 50 and not re.match(r'^(Title|Company|Location|Posted|Link)', l, re.IGNORECASE)]
                if desc_lines:
                    description_match = type('obj', (), {'group': lambda x: ' '.join(desc_lines[:3])})()

            link_match = re.search(r'(?:Link|URL|href):\s*(https?://[^\s\n]+)', block, re.IGNORECASE)
            if not link_match:
                link_match = re.search(r'(https?://(?:www\.)?linkedin\.com/jobs/[^\s\n]+)', block)

            if title_match:
                job['title'] = title_match.group(1).strip()
                job['company'] = company_match.group(1).strip() if company_match else 'Company not specified'
                job['location'] = location_match.group(1).strip() if location_match else 'Location not specified'
                job['posted'] = posted_match.group(1).strip() if posted_match else 'Recently posted'
                job['description'] = description_match.group(1).strip()[:500] if description_match else ''
                job['link'] = link_match.group(1).strip() if link_match else f"https://www.linkedin.com/jobs/search/?keywords={job.get('title', '').replace(' ', '%20')}"
                job['raw_content'] = block[:1000]

                jobs.append(job)

    return jobs

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
        col2a, col2b = st.columns([1, 1])
        with col2a:
            num_jobs = st.number_input(
                "Number of jobs to scrape",
                min_value=10,
                max_value=500,
                value=100,
                step=10,
                help="Choose how many jobs to scrape (10-500)"
            )
        with col2b:
            st.write("")  # Empty space for alignment
            st.caption(f"📊 Will scrape up to {num_jobs} jobs")

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

                raw_content = asyncio.run(scrape_linkedin_jobs(resume_text, num_jobs))

                if raw_content:
                    st.session_state.scraped_jobs = parse_linkedin_jobs(raw_content)

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
