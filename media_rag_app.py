"""Streamlit entry point: streamlit run media_rag_app.py --server.port 8503."""

from pathlib import Path

import streamlit as st

from media_rag.config import RagError, Settings
from media_rag.media import check_tools
from media_rag.models import timestamp
from media_rag.service import MediaLibrary
from media_rag.ui import plain_markdown


@st.cache_resource(show_spinner=False)
def get_library(settings):
    library = MediaLibrary(settings)
    library.initialize()
    return library


def show_error(exc):
    st.error(str(exc) if isinstance(exc, RagError) else
             "The media library could not complete this request. Check the local services and try again.")


def render_answer(answer, evidence, library):
    if answer["status"] == "insufficient_evidence":
        st.info(plain_markdown(answer["message"]))
        return
    st.subheader("Answer")
    for claim in answer["claims"]:
        references = " ".join(f"[{source_id}]" for source_id in claim["source_ids"])
        st.markdown(f"{plain_markdown(claim['text'])} **{references}**")
    cited = {source_id for claim in answer["claims"] for source_id in claim["source_ids"]}
    st.subheader("Listen and verify")
    for item in evidence:
        if item.source_id not in cited:
            continue
        with st.container(border=True):
            st.markdown(f"**[{item.source_id}] {plain_markdown(item.title)}**")
            st.caption(f"{timestamp(item.start)}–{timestamp(item.end)} in the original recording")
            try:
                url = library.storage.playback_url(item.object_key)
                if item.kind == "video":
                    st.video(url, format="video/mp4")
                else:
                    st.audio(url, format="audio/mpeg")
            except Exception as exc:
                show_error(exc)


def generate_answer(library, search):
    with st.spinner("Examining the retrieved clips with Gemini…"):
        answer, evidence = library.answer(search["question"], search["hits"])
    st.session_state.media_rag_answer = {"answer": answer, "evidence": evidence}


def main():
    st.set_page_config(page_title="Media Library · Native RAG", page_icon="🎞️", layout="wide")
    st.title("Ask your audio and video")
    st.caption("Find the right moment. Get an answer grounded in the original recording.")
    try:
        overrides = dict(st.secrets.get("media_rag", {}))
    except st.errors.StreamlitSecretNotFoundError:
        overrides = {}
    try:
        settings = Settings.load(overrides)
    except RagError as exc:
        show_error(exc)
        st.stop()

    with st.sidebar:
        st.header("Media Library")
        st.caption("MinIO · PostgreSQL / pgvector · Marengo 3.5 · Gemini")
        st.button("Refresh library", use_container_width=True)
        with st.expander("Connections and setup"):
            st.write("TwelveLabs key: " + ("Configured" if settings.twelvelabs_api_key else "Missing"))
            st.write("Gemini key: " + ("Configured" if settings.gemini_api_key else "Missing"))
            st.caption("Use .env.media-rag or the [media_rag] section of Streamlit secrets.")
            st.code("python scripts/init_media_rag.py\n"
                    "docker compose --env-file .env.media-rag -f compose.media-rag.yaml up -d --build", language="bash")

    if settings.missing_infrastructure():
        st.info("Set up your local media library to get started.")
        st.markdown("Run `python scripts/init_media_rag.py`, add your API keys to `.env.media-rag`, "
                    "then follow the startup commands in `docs/media-rag.md`.")
        st.stop()
    try:
        check_tools()
        library = get_library(settings)
        assets = library.database.list_assets()
    except Exception as exc:
        show_error(exc)
        st.info("Start the MinIO and PostgreSQL services, then refresh this page. See docs/media-rag.md.")
        st.stop()

    ready = [asset for asset in assets if asset["status"] == "ready"]
    pending = [asset for asset in assets if asset["status"] in {"queued", "indexing"}]
    with st.sidebar:
        st.metric("Ready to search", len(ready))
        st.metric("Waiting or indexing", len(pending))
        st.caption("Local library for one trusted user. Media sent for indexing goes to TwelveLabs; "
                   "selected clips sent for answering go to Google Gemini.")

    ask_tab, upload_tab, library_tab = st.tabs(["Ask library", "Add media", "Library"])
    with ask_tab:
        if not ready:
            st.info("Add an audio or video file, then let the indexing worker finish. "
                    "Refresh the library when it is ready.")
        if not settings.twelvelabs_api_key:
            st.info("Add TWELVELABS_API_KEY to enable native media search.")
        if not settings.gemini_api_key:
            st.caption("You can search without Gemini. Add GEMINI_API_KEY to generate cited answers.")
        ready_by_id = {str(asset["id"]): asset for asset in ready}
        with st.form("media_rag_question"):
            question = st.text_area("What would you like to know?", max_chars=2000,
                                    placeholder="Where does the speaker explain how to evaluate a RAG system?")
            selected = st.multiselect("Search specific recordings", list(ready_by_id),
                                      format_func=lambda value: ready_by_id[value]["title"],
                                      placeholder="All ready recordings")
            columns = st.columns([2, 1, 1])
            mode = columns[0].selectbox("Look in", ["Audio and video", "Speech and sounds", "Visuals"])
            top_k = columns[1].selectbox("Evidence clips", [3, 5, 8], index=1)
            min_score = columns[2].number_input("Minimum similarity", min_value=0.0, max_value=1.0,
                                              value=0.15, step=0.05,
                                              help="A retrieval cutoff, not a probability of correctness. Tune on your own examples.")
            disabled = not ready or not settings.twelvelabs_api_key
            actions = st.columns(2)
            ask = actions[0].form_submit_button("Ask & cite", type="primary", use_container_width=True,
                                               disabled=disabled or not settings.gemini_api_key)
            search_only = actions[1].form_submit_button("Search only", use_container_width=True, disabled=disabled)
        if ask or search_only:
            st.session_state.pop("media_rag_search", None)
            st.session_state.pop("media_rag_answer", None)
            try:
                modality = {"Audio and video": None, "Speech and sounds": "audio", "Visuals": "visual"}[mode]
                with st.spinner("Finding relevant moments…"):
                    hits = library.retrieve(question, selected or None, modality, top_k, min_score)
                search = {"question": question.strip(), "hits": hits}
                st.session_state.media_rag_search = search
                if ask and hits:
                    generate_answer(library, search)
            except Exception as exc:
                show_error(exc)

        search = st.session_state.get("media_rag_search")
        if search:
            st.caption("Results for: " + plain_markdown(search["question"]))
            if not search["hits"]:
                st.info("No relevant moments were found. Try a different question, a broader recording selection, "
                        "or a lower similarity cutoff.")
            else:
                st.subheader("Retrieved moments")
                for hit in search["hits"]:
                    with st.expander(f"{plain_markdown(hit.title)} · {timestamp(hit.start)}–{timestamp(hit.end)}"):
                        st.caption(f"Matched {', '.join(hit.modalities)} · similarity {hit.score:.3f}")
                        try:
                            url = library.storage.playback_url(hit.object_key)
                            if hit.kind == "video":
                                st.video(url, start_time=hit.start, end_time=hit.end)
                            else:
                                st.audio(url, start_time=hit.start, end_time=hit.end)
                        except Exception as exc:
                            show_error(exc)
                if "media_rag_answer" not in st.session_state:
                    if st.button("Answer this search with Gemini", disabled=not settings.gemini_api_key):
                        try:
                            generate_answer(library, search)
                        except Exception as exc:
                            show_error(exc)
        if answer := st.session_state.get("media_rag_answer"):
            render_answer(answer["answer"], answer["evidence"], library)

    with upload_tab:
        st.subheader("Add a recording")
        st.write("Upload audio or video. The worker indexes its sound and visual content directly.")
        st.caption("MP3, WAV, MP4, MOV, or WebM · up to 200 MB per file. "
                   "Indexing sends the recording to TwelveLabs.")
        with st.form("media_rag_upload", clear_on_submit=True):
            uploaded = st.file_uploader("Recording", type=["mp3", "wav", "mp4", "mov", "webm"])
            title = st.text_input("Recording title", max_chars=200, placeholder="Episode or presentation title")
            submitted = st.form_submit_button("Add to library", type="primary")
        if submitted:
            try:
                if uploaded is None:
                    raise RagError("Choose an audio or video file first.")
                with st.spinner("Saving your recording…"):
                    asset, created = library.add_upload(uploaded, title or Path(uploaded.name).stem)
                if created:
                    st.success(f"Added {plain_markdown(asset['title'])}. The indexing worker will pick it up shortly.")
                else:
                    st.info(f"This file is already in the library as {plain_markdown(asset['title'])} ({asset['status']}).")
            except Exception as exc:
                show_error(exc)

    with library_tab:
        st.subheader("Your recordings")
        if not assets:
            st.info("Your library is empty. Add your first recording in the Add media tab.")
        for asset in assets:
            with st.container(border=True):
                columns = st.columns([4, 1, 1])
                columns[0].markdown(f"**{plain_markdown(asset['title'])}**")
                columns[0].caption(asset["stage"])
                columns[1].write(asset["status"].capitalize())
                columns[2].write(timestamp(asset["duration"]))
                if asset["status"] == "ready":
                    st.caption(f"{asset['kind'].capitalize()} · {asset['embedding_count']} indexed embeddings")
                if asset["status"] == "failed":
                    st.error(asset["error"] or "Indexing failed.")
                    if st.button("Retry indexing", key=f"retry_{asset['id']}"):
                        try:
                            library.database.retry(asset["id"])
                            st.rerun()
                        except Exception as exc:
                            show_error(exc)


if __name__ == "__main__":
    main()
