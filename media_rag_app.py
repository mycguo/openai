"""Streamlit entry point: streamlit run media_rag_app.py --server.port 8503."""

from pathlib import Path

import streamlit as st

from media_rag.config import RagError, Settings
from media_rag.media import check_tools
from media_rag.models import timestamp
from media_rag.service import MediaLibrary
from media_rag.ui import plain_markdown


@st.cache_resource(show_spinner=False)
def get_library(settings, youtube=False):
    if youtube:
        from youtube_rag.service import YouTubeLibrary
        library = YouTubeLibrary(settings)
    else:
        library = MediaLibrary(settings)
    library.initialize()
    return library


def error_message(exc):
    return str(exc) if isinstance(exc, RagError) else \
           "The media library could not complete this request. Check the connections and try again."


def show_error(exc):
    message = error_message(exc)
    st.error(message)
    return message


def render_source_link(source_url, start=0):
    if source_url:
        from media_rag.youtube_urls import timestamp_url
        try:
            url = timestamp_url(source_url, start)
        except (RagError, ValueError, OverflowError):
            return  # Invalid stored metadata must not become an external link.
        st.link_button("Open on YouTube", url)


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
            render_source_link(item.source_url, item.start)
            try:
                url = library.storage.playback_url(item.object_key)
                if item.kind == "video":
                    st.video(url, format="video/mp4")
                else:
                    st.audio(url, format="audio/mpeg")
            except Exception as exc:
                show_error(exc)


def generate_answer(library, search):
    st.session_state.pop("media_rag_answer_error", None)
    try:
        with st.spinner("Examining the retrieved clips with Gemini…"):
            answer, evidence = library.answer(search["question"], search["hits"])
    except Exception as exc:
        st.session_state.media_rag_answer_error = error_message(exc)
        return
    st.session_state.media_rag_answer = {"answer": answer, "evidence": evidence}


def index_in_streamlit(library, asset_id, retry=False):
    with st.status("Indexing recording…", expanded=True) as status:
        st.write("Keep this page open while indexing. You can resume interrupted processing from the Library tab.")
        try:
            library.index_recording(asset_id, on_progress=lambda stage: status.update(label=stage), retry=retry)
        except Exception as exc:
            status.update(label="Indexing did not finish", state="error")
            st.session_state.media_rag_index_error = show_error(exc)
            return False
        status.update(label="Ready to search", state="complete", expanded=False)
    st.session_state.media_rag_notice = "Your recording is ready to search."
    return True


def main(youtube=False):
    st.set_page_config(page_title="youtube-rag" if youtube else "Media Library · Native RAG", page_icon="🎞️", layout="wide",
                       initial_sidebar_state="collapsed")
    st.html(Path(__file__).with_name("media_rag") / "styles.css")
    st.title("youtube-rag" if youtube else "Ask your audio and video")
    st.caption("Find the right moment. Get an answer grounded in the original recording.")
    if notice := st.session_state.pop("media_rag_notice", None):
        st.success(notice)
    if error := st.session_state.pop("media_rag_index_error", None):
        with st.status("Indexing did not finish", state="error", expanded=True):
            st.error(error)
    try:
        table = "youtube_rag" if youtube and "youtube_rag" in st.secrets else "media_rag"
        overrides = dict(st.secrets.get(table, {}))
    except st.errors.StreamlitSecretNotFoundError:
        overrides = {}
    try:
        settings = Settings.load(overrides)
    except RagError as exc:
        show_error(exc)
        st.stop()

    if settings.missing_infrastructure():
        if settings.storage_provider == "neon":
            st.info("Configure the Neon database and object storage credentials to get started.")
            st.markdown("Follow the setup commands in `docs/media-rag-neon.md`.")
        else:
            st.info("Set up your local media library to get started.")
            st.markdown("Run `python scripts/init_media_rag.py`, add your API keys to `.env.media-rag`, "
                        "then follow the startup commands in `docs/media-rag.md`.")
        st.stop()
    try:
        check_tools()
        library = get_library(settings, youtube=youtube)
        assets = library.database.list_youtube_assets() if youtube else library.database.list_assets()
    except Exception as exc:
        show_error(exc)
        if settings.storage_provider == "neon":
            st.info("Check the Neon database and private bucket configuration. See docs/media-rag-neon.md.")
        else:
            st.info("Start the MinIO and PostgreSQL services, then refresh this page. See docs/media-rag.md.")
        st.stop()

    ready = [asset for asset in assets if asset["status"] == "ready"]
    pending = [asset for asset in assets if asset["status"] in {"queued", "indexing"}]
    status, refresh = st.columns([5, 1], vertical_alignment="center")
    status.caption(f"{len(ready)} ready to search · {len(pending)} waiting or indexing")
    refresh.button("Refresh library", use_container_width=True)

    ask_tab, upload_tab, library_tab = st.tabs(["Ask library", "Add YouTube video" if youtube else "Add media", "Library"])
    with ask_tab:
        if not ready:
            st.info("Add a YouTube URL to index its audio and video, or resume a saved video from the Library tab."
                    if youtube else "Upload an audio or video file to index it here, or index a saved recording from the Library tab.")
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
            st.session_state.pop("media_rag_answer_error", None)
            try:
                modality = {"Audio and video": None, "Speech and sounds": "audio", "Visuals": "visual"}[mode]
                with st.spinner("Finding relevant moments…"):
                    scope = selected or (list(ready_by_id) if youtube else None)
                    hits = library.retrieve(question, scope, modality, top_k, min_score)
                search = {"question": question.strip(), "hits": hits}
                st.session_state.media_rag_search = search
                if ask and hits:
                    generate_answer(library, search)
            except Exception as exc:
                show_error(exc)

        search = st.session_state.get("media_rag_search")
        if search:
            if error := st.session_state.get("media_rag_answer_error"):
                st.error(error)
                st.info("Search found relevant clips, but Gemini has not completed a cited answer. "
                        "You can review the clips below or retry the cited answer.")
            st.caption("Results for: " + plain_markdown(search["question"]))
            if not search["hits"]:
                st.info("No relevant moments were found. Try a different question, a broader recording selection, "
                        "or a lower similarity cutoff.")
            else:
                st.subheader("Retrieved moments")
                for hit in search["hits"]:
                    with st.expander(f"{plain_markdown(hit.title)} · {timestamp(hit.start)}–{timestamp(hit.end)}"):
                        st.caption(f"Matched {', '.join(hit.modalities)} · similarity {hit.score:.3f}")
                        render_source_link(hit.source_url, hit.start)
                        try:
                            url = library.storage.playback_url(hit.object_key)
                            if hit.kind == "video":
                                st.video(url, start_time=hit.start, end_time=hit.end)
                            else:
                                st.audio(url, start_time=hit.start, end_time=hit.end)
                        except Exception as exc:
                            show_error(exc)
                if "media_rag_answer" not in st.session_state:
                    label = "Retry cited answer" if st.session_state.get("media_rag_answer_error") else \
                            "Answer this search with Gemini"
                    if st.button(label, key="media_rag_generate_answer", disabled=not settings.gemini_api_key):
                        generate_answer(library, search)
                        st.rerun()
        if answer := st.session_state.get("media_rag_answer"):
            render_answer(answer["answer"], answer["evidence"], library)

    with upload_tab:
        st.subheader("Add a YouTube video" if youtube else "Add a recording")
        if youtube:
            st.write("Paste a YouTube video URL to index its sound and visual content.")
            st.caption("Publicly accessible videos · up to 60 minutes and 200 MB · downloaded at up to 720p. "
                       "Keep this page open during import and indexing.")
        else:
            st.write("Upload audio or video and index its sound and visual content directly in this app.")
            st.caption("MP3, WAV, MP4, MOV, or WebM · up to 200 MB per file. "
                       "Indexing sends the recording to TwelveLabs. Keep this page open until it finishes.")
        if not settings.twelvelabs_api_key:
            st.info("Add TWELVELABS_API_KEY to import and index videos." if youtube else
                    "Add TWELVELABS_API_KEY to upload and index recordings.")
        with st.form("media_rag_upload", clear_on_submit=True):
            if youtube:
                youtube_url = st.text_input("YouTube URL", max_chars=2048,
                                            placeholder="https://www.youtube.com/watch?v=…")
            else:
                uploaded = st.file_uploader("Recording", type=["mp3", "wav", "mp4", "mov", "webm"])
            title = st.text_input("Video title (optional)" if youtube else "Recording title", max_chars=200,
                                  placeholder="Use the YouTube title" if youtube else "Episode or presentation title")
            submitted = st.form_submit_button("Import & index" if youtube else "Upload & index", type="primary",
                                              disabled=not settings.twelvelabs_api_key)
        if submitted:
            try:
                if youtube:
                    with st.status("Importing YouTube video…", expanded=True) as status:
                        try:
                            asset, created = library.add_youtube(
                                youtube_url, title, on_progress=lambda stage: status.update(label=stage))
                        except Exception:
                            status.update(label="Import did not finish", state="error")
                            raise
                        status.update(label="Video saved", state="complete", expanded=False)
                else:
                    if uploaded is None:
                        raise RagError("Choose an audio or video file first.")
                    with st.spinner("Saving your recording…"):
                        asset, created = library.add_upload(uploaded, title or Path(uploaded.name).stem)
                if created:
                    st.success(f"Saved {plain_markdown(asset['title'])}.")
                else:
                    st.info(f"This file is already in the library as {plain_markdown(asset['title'])} ({asset['status']}).")
                if asset["status"] != "ready":
                    index_in_streamlit(library, asset["id"], retry=asset["status"] == "failed")
                    st.rerun()
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
                columns[0].caption("Ready to index" if asset["status"] == "queued" else asset["stage"])
                columns[1].write(asset["status"].capitalize())
                columns[2].write(timestamp(asset["duration"]))
                render_source_link(asset.get("source_url", ""))
                if asset["status"] == "ready":
                    st.caption(f"{asset['kind'].capitalize()} · {asset['embedding_count']} indexed embeddings")
                if asset["status"] in {"queued", "indexing"}:
                    label = "Index recording" if asset["status"] == "queued" else "Resume indexing"
                    if asset["status"] == "indexing":
                        st.caption("If processing was interrupted, wait up to five minutes, then resume.")
                    if st.button(label, key=f"index_{asset['id']}", disabled=not settings.twelvelabs_api_key):
                        index_in_streamlit(library, asset["id"])
                        st.rerun()
                if asset["status"] == "failed":
                    st.error(asset["error"] or "Indexing failed.")
                    if st.button("Retry indexing", key=f"retry_{asset['id']}",
                                 disabled=not settings.twelvelabs_api_key):
                        index_in_streamlit(library, asset["id"], retry=True)
                        st.rerun()

        with st.expander("Connections and setup"):
            storage_name = "Neon Object Storage" if settings.storage_provider == "neon" else "MinIO"
            st.caption(f"{storage_name} · PostgreSQL / pgvector · Marengo 3.5 · Gemini")
            st.write("TwelveLabs key: " + ("Configured" if settings.twelvelabs_api_key else "Missing"))
            st.write("Gemini key: " + ("Configured" if settings.gemini_api_key else "Missing"))
            st.caption("Gemini model: " + plain_markdown(settings.gemini_model))
            st.caption("Use MEDIA_RAG_ENV_FILE or the [youtube_rag] section of Streamlit secrets. "
                       "The [media_rag] section is also accepted." if youtube else
                       "Use MEDIA_RAG_ENV_FILE or the [media_rag] section of Streamlit secrets.")
            if youtube:
                st.caption("YouTube setup and deployment: docs/youtube-rag.md")
            if settings.storage_provider == "neon":
                st.caption("Neon setup and indexing: docs/media-rag-neon.md")
            else:
                st.code("python scripts/init_media_rag.py\n"
                        "docker compose --env-file .env.media-rag -f compose.media-rag.yaml up -d --build", language="bash")
            st.caption("Library for one trusted user. Media sent for indexing goes to TwelveLabs; "
                       "selected clips sent for answering go to Google Gemini.")


if __name__ == "__main__":
    main()
