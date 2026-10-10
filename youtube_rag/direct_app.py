"""Streamlit Q&A with a public YouTube URL passed directly to Gemini."""

import streamlit as st

from media_rag.config import RagError
from media_rag.models import timestamp
from media_rag.ui import plain_markdown
from media_rag.youtube_urls import parse_youtube_url, timestamp_url
from .direct_gemini import ask_youtube, load_direct_settings, validate_question


def generate(settings, request):
    st.session_state.pop("youtube_direct_answer", None)
    st.session_state.pop("youtube_direct_error", None)
    try:
        with st.spinner("Gemini is analyzing the YouTube video…"):
            answer = ask_youtube(settings, request["url"], request["question"])
        st.session_state.youtube_direct_answer = answer
    except Exception as exc:
        st.session_state.youtube_direct_error = str(exc) if isinstance(exc, RagError) else \
            "Gemini could not complete this request. Check the app logs and retry."


def render_direct(overrides):
    st.caption("Ask about one public YouTube video. Its URL and your question are sent to Gemini; "
               "this mode does not save the video or add it to your library.")
    try:
        settings = load_direct_settings(overrides)
    except RagError as exc:
        st.error(str(exc))
        return
    if not settings.gemini_api_key:
        st.info("Add GEMINI_API_KEY to Streamlit secrets to enable direct YouTube questions.")
        st.code('[youtube_rag]\nGEMINI_API_KEY = "<your Gemini API key>"', language="toml")
    with st.form("youtube_direct_question"):
        url = st.text_input("YouTube URL", max_chars=2048, placeholder="https://www.youtube.com/watch?v=…",
                            key="youtube_direct_url")
        question = st.text_area("What would you like to know?", max_chars=2000,
                                placeholder="What are the main takeaways, and where are they explained?",
                                key="youtube_direct_question_text")
        submit = st.form_submit_button("Ask Gemini", type="primary", use_container_width=True,
                                       disabled=not settings.gemini_api_key)
    st.caption("Gemini's YouTube URL support is a preview for public videos. Private and unlisted videos "
               "are unsupported; availability and your Gemini quota can limit requests.")
    if submit:
        for key in ("youtube_direct_request", "youtube_direct_answer", "youtube_direct_error"):
            st.session_state.pop(key, None)
        try:
            request = {"url": parse_youtube_url(url).url, "question": validate_question(question)}
            st.session_state.youtube_direct_request = request
            generate(settings, request)
        except RagError as exc:
            st.session_state.youtube_direct_error = str(exc)
    request = st.session_state.get("youtube_direct_request")
    if error := st.session_state.get("youtube_direct_error"):
        st.error(error)
        if request and st.button("Retry Gemini answer", disabled=not settings.gemini_api_key):
            generate(settings, request)
            st.rerun()
    answer = st.session_state.get("youtube_direct_answer")
    if not answer or not request:
        return
    st.caption("Results for: " + plain_markdown(request["question"]))
    st.link_button("Open original video", request["url"])
    if answer["status"] == "insufficient_evidence":
        st.info(plain_markdown(answer["message"]))
        return
    st.subheader("Answer")
    for index, claim in enumerate(answer["claims"], 1):
        st.markdown(f"{plain_markdown(claim['text'])} **[{index}]**")
    st.subheader("Check the suggested moments")
    st.caption("These timestamps are suggested by Gemini and have not been independently verified. "
               "Open the original video to check each claim.")
    for index, claim in enumerate(answer["claims"], 1):
        label = f"[{index}] {timestamp(claim['start_sec'])}–{timestamp(claim['end_sec'])}"
        with st.container(border=True):
            st.write(label)
            st.link_button("Open on YouTube", timestamp_url(request["url"], claim["start_sec"]))
            with st.expander("Play this moment"):
                st.video(request["url"], start_time=claim["start_sec"], end_time=claim["end_sec"])
