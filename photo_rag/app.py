"""Ask My Photo frontend; all inference runs only on explicit user actions."""

from pathlib import Path

import streamlit as st

from media_rag.config import ROOT, RagError
from media_rag.ui import plain_markdown
from .config import MAX_BATCH, load_settings
from .service import PhotoLibrary


@st.cache_resource(show_spinner=False)
def get_library(settings):
    library = PhotoLibrary(settings)
    library.initialize()
    return library


def error_message(exc):
    return str(exc) if isinstance(exc, RagError) else \
        "The photo library could not complete this request. Check the connections and retry."


def secret_overrides():
    try:
        values = {key: value for key, value in st.secrets.items() if isinstance(value, (str, int, float, bool))}
        # Merge partial app tables so a Gemini-only YouTube table does not hide storage settings.
        for name in ("media_rag", "youtube_rag", "ask_my_photo"):
            values.update(dict(st.secrets.get(name, {})))
        return values
    except st.errors.StreamlitSecretNotFoundError:
        return {}


def show_photo(library, photo, reference=None):
    try:
        st.image(library.storage.playback_url(photo["thumbnail_key"]), width="stretch")
        title = plain_markdown(photo["title"])
        st.markdown(f"**{('[' + str(reference) + '] ') if reference else ''}{title}**")
        st.caption(f"{photo['width']} × {photo['height']}")
        st.link_button("Open original photo", library.storage.playback_url(photo["object_key"]))
    except Exception as exc:
        st.error(error_message(exc))


def generate_answer(library, search):
    st.session_state.pop("photo_rag_answer", None)
    st.session_state.pop("photo_rag_answer_error", None)
    try:
        with st.spinner("Examining the matching photos with Gemini…"):
            answer, _ = library.answer(search["question"], search["hits"])
        st.session_state.photo_rag_answer = answer
    except Exception as exc:
        st.session_state.photo_rag_answer_error = error_message(exc)


def main():
    st.set_page_config(page_title="Ask My Photo", page_icon="📷", layout="wide", initial_sidebar_state="collapsed")
    st.html(ROOT / "media_rag" / "styles.css")
    st.title("Ask My Photo")
    st.caption("Find a memory. Ask a question. See the photos behind the answer.")
    if notice := st.session_state.pop("photo_rag_notice", None):
        st.success(notice)
    for error in st.session_state.pop("photo_rag_notices_error", []):
        st.error(error)
    try:
        settings = load_settings(secret_overrides())
        if settings.missing_infrastructure():
            st.info("Reuse your media app's database and storage settings in .env.media-rag or Streamlit secrets. "
                    "See docs/ask-my-photo.md for setup.")
            return
        library = get_library(settings)
        photos, counts = library.database.list_photos(), library.database.counts()
    except Exception as exc:
        st.error(error_message(exc))
        st.info("Check the existing PostgreSQL and private storage services. See docs/ask-my-photo.md.")
        return
    status, refresh = st.columns([5, 1], vertical_alignment="center")
    status.caption(f"{counts.get('ready', 0)} photos ready · "
                   f"{sum(counts.get(state, 0) for state in ('queued', 'indexing', 'failed'))} waiting or needing attention")
    refresh.button("Refresh library", use_container_width=True)
    if not settings.gemini_api_key:
        st.info("Add GEMINI_API_KEY to index photos and answer questions. You can save photos now and index them later.")

    ask_tab, upload_tab, library_tab = st.tabs(["Ask your photos", "Add photos", "Photo library"])
    with ask_tab:
        ready = {str(photo["id"]): photo for photo in photos if photo["status"] == "ready"}
        with st.form("photo_rag_question"):
            question = st.text_area("What would you like to know?", max_chars=2000,
                placeholder="Which photos show a bicycle? What does the sign in those photos say?")
            selected = st.multiselect("Search specific photos", list(ready),
                format_func=lambda value: ready[value]["title"], placeholder="All ready photos")
            options = st.columns(2)
            top_k = options[0].selectbox("Evidence photos", [3, 5, 8], index=1)
            min_score = options[1].number_input("Minimum similarity", min_value=0.0, max_value=1.0,
                value=0.15, step=0.05, help="A search cutoff, not a probability that the answer is correct.")
            disabled = not counts.get("ready") or not settings.gemini_api_key
            actions = st.columns(2)
            ask = actions[0].form_submit_button("Ask & cite", type="primary", use_container_width=True, disabled=disabled)
            search_only = actions[1].form_submit_button("Find photos", use_container_width=True, disabled=disabled)
        if not counts.get("ready"):
            st.info("Add photos to your library to start asking questions.")
        if ask or search_only:
            for key in ("photo_rag_search", "photo_rag_answer", "photo_rag_answer_error"):
                st.session_state.pop(key, None)
            try:
                with st.spinner("Finding matching photos…"):
                    hits = library.retrieve(question, selected or None, top_k, min_score)
                search = {"question": question.strip(), "hits": hits}
                st.session_state.photo_rag_search = search
                if ask and hits:
                    generate_answer(library, search)
            except Exception as exc:
                st.error(error_message(exc))
        search = st.session_state.get("photo_rag_search")
        if search:
            if error := st.session_state.get("photo_rag_answer_error"):
                st.error(error)
                st.info("The matching photos are available below. You can retry the answer without searching again.")
            if search["hits"] and settings.gemini_api_key and st.button(
                    "Retry answer" if st.session_state.get("photo_rag_answer_error") else "Answer from these photos"):
                generate_answer(library, search)
                st.rerun()
            st.caption("Results for: " + plain_markdown(search["question"]))
            if answer := st.session_state.get("photo_rag_answer"):
                if answer["status"] == "insufficient_evidence":
                    st.info(plain_markdown(answer["message"]))
                else:
                    st.subheader("Answer")
                    for claim in answer["claims"]:
                        refs = " ".join(f"[{value}]" for value in claim["source_ids"])
                        st.markdown(f"{plain_markdown(claim['text'])} **{refs}**")
            if not search["hits"]:
                st.info("No matching photos were found. Try a broader question or a lower similarity cutoff.")
            else:
                st.subheader("Matching photos")
                st.caption("References point to these photos. These are search results, rather than an exhaustive library count.")
                columns = st.columns(3)
                for index, photo in enumerate(search["hits"], 1):
                    with columns[(index - 1) % 3].container(border=True):
                        show_photo(library, photo, index)
                        st.caption(f"Similarity {photo['score']:.3f}")

    with upload_tab:
        st.subheader("Add to your photo library")
        st.write("Upload your photos once, then ask questions about the library.")
        st.caption("JPEG, PNG, WebP, HEIC and HEIF · up to 15 MB and 50 megapixels per photo · 10 photos per batch. "
                   "Gemini receives resized copies for indexing. Keep this page open while processing.")
        with st.form("photo_rag_upload", clear_on_submit=True):
            uploads = st.file_uploader("Photos", type=["jpg", "jpeg", "png", "webp", "heic", "heif"],
                                       accept_multiple_files=True, max_upload_size=15)
            title = st.text_input("Title (optional, single photo)", max_chars=200)
            submitted = st.form_submit_button("Add photos", type="primary")
        if submitted:
            st.session_state.photo_rag_notices_error = []
            if not uploads or len(uploads) > MAX_BATCH or (title.strip() and len(uploads) != 1):
                st.error("Choose 1–10 photos. A custom title can be used when uploading one photo.")
            else:
                saved = 0
                with st.status("Adding photos…", expanded=True) as progress:
                    for index, uploaded in enumerate(uploads, 1):
                        progress.update(label=f"Processing photo {index} of {len(uploads)}")
                        try:
                            photo, _ = library.add_upload(uploaded, title)
                            saved += 1
                            if photo["status"] != "ready" and settings.gemini_api_key:
                                library.index_photo(photo["id"], retry=photo["status"] == "failed")
                        except Exception as exc:
                            st.session_state.photo_rag_notices_error.append(error_message(exc))
                    progress.update(label="Photos processed", state="complete", expanded=False)
                st.session_state.photo_rag_notice = f"{saved} photo(s) saved or already in your library."
                st.rerun()

    with library_tab:
        st.subheader("Your photos")
        if not photos:
            st.info("Your library is empty. Add your first photos in the Add photos tab.")
        else:
            if len(photos) == 500:
                st.caption("Browsing the latest 500 photos. Questions search all ready photos in the library.")
            pages = (len(photos) + 11) // 12
            page = st.selectbox("Library page", range(pages), format_func=lambda value: f"Page {value + 1} of {pages}")
            columns = st.columns(3)
            for index, photo in enumerate(photos[page * 12:(page + 1) * 12]):
                with columns[index % 3].container(border=True):
                    show_photo(library, photo)
                    st.caption(photo["status"].capitalize())
                    if photo["status"] == "failed":
                        st.error(plain_markdown(photo["error"] or "Indexing failed."))
                    if photo["status"] != "ready":
                        label = {"queued": "Index photo", "failed": "Retry indexing", "indexing": "Resume indexing"}[photo["status"]]
                        if st.button(label, key=f"photo-index-{photo['id']}", disabled=not settings.gemini_api_key):
                            try:
                                with st.spinner("Indexing photo…"):
                                    library.index_photo(photo["id"], retry=photo["status"] == "failed")
                                st.session_state.photo_rag_notice = "Your photo is ready to search."
                                st.rerun()
                            except Exception as exc:
                                st.error(error_message(exc))
    with st.expander("Connections and setup"):
        st.caption(f"{'Neon Object Storage' if settings.storage_provider == 'neon' else 'MinIO'} · PostgreSQL / pgvector · Gemini")
        st.write("Gemini key: " + ("Configured" if settings.gemini_api_key else "Missing"))
        st.caption("Reuses .env.media-rag and the media_rag / youtube_rag secret tables. "
                   "Optional ask_my_photo overrides take precedence. See docs/ask-my-photo.md.")
        st.caption("One shared library for trusted users. Restrict access to this app. "
                   "Photos go to Google for indexing and answers. Model previews omit EXIF/GPS; stored originals retain it.")
