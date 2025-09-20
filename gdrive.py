# https://github.com/isaaccasm/rag-google-doc
# pip install llama-index google-api-python-client openai
import hashlib
import io
import json
import logging
import os
from typing import Optional, List

import streamlit as st

from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
from google.oauth2.credentials import Credentials
from google.oauth2 import service_account
from llama_index.core import VectorStoreIndex
try:
    from llama_index.core import Node
except ImportError:  # pragma: no cover - newer versions expose nodes under data_structs
    try:
        from llama_index.core.data_structs import Node
    except ImportError:  # pragma: no cover - fallback for engine indices
        from llama_index.core.schema import Node
from llama_index.core.storage.storage_context import StorageContext
from googleapiclient.http import MediaIoBaseDownload

try:
    from docx import Document as DocxDocument
except ImportError:  # pragma: no cover - optional dependency
    DocxDocument = None
try:
    from pypdf import PdfReader
except ImportError:  # pragma: no cover - optional dependency
    PdfReader = None


logger = logging.getLogger(__name__)


DEFAULT_FOLDER_IDS: List[str] = [
    '1aVf1BtyWByR9zeEC-EiK9k4wQfmZzbZW',
]
DEFAULT_QUESTION = "What are the advancements in reinforcement learning?"


OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# Authenticate and Access Google Drive API
SCOPES = ['https://www.googleapis.com/auth/drive.readonly', 'https://www.googleapis.com/auth/documents.readonly']
CREDS_FILE = 'credentials-gdrive.json'


def custom_chunk_splitter(documents):
    """
    Split documents into chunks based on '###' delimiters.
    """
    chunks = []
    logger.info("Splitting %d documents into chunks", len(documents))
    for doc in documents:
        if hasattr(doc, "text"):
            raw_text = doc.text
            metadata = getattr(doc, "metadata", {})
        elif isinstance(doc, dict):
            raw_text = doc.get("content", "")
            metadata = doc.get("metadata", {"source": doc.get("name", "doc")})
        else:
            raw_text = ""
            metadata = {}
        doc_chunks = raw_text.split('###')  # Split text into chunks by delimiter
        logger.debug("Document generated %d raw chunks", len(doc_chunks))
        for i, chunk in enumerate(doc_chunks):
            if chunk.strip():  # Ignore empty chunks
                chunks.append(
                    Node(
                        doc_id=f"{metadata.get('source', 'doc')}_chunk_{i}",
                        text=chunk.strip(),
                        extra_info=metadata
                    )
                )
    logger.info("Generated %d cleaned chunks", len(chunks))
    return chunks


class RagGoogleDoc:
    def __init__(self, folder_ids, local_dir_docs='Data/Docs', save_index_address=None, progress_callback=None):
        creds = service_account.Credentials.from_service_account_info(
            st.secrets["gcp_service_account"], scopes=SCOPES
        )


        self.drive_service = build('drive', 'v3', credentials=creds)
        self.docs_service = build('docs', 'v1', credentials=creds)

        self.folder_ids = [folder_ids] if isinstance(folder_ids, str) else folder_ids
        self.local_dir_docs = local_dir_docs
        self.save_index_address = save_index_address
        self.progress_callback = progress_callback
        self.summary_data = {
            'folders': {}
        }

        self.drive_service = build('drive', 'v3', credentials=creds)
        self.docs_service = build('docs', 'v1', credentials=creds)

    def _report(self, message: str):
        logger.info(message)
        if self.progress_callback:
            try:
                self.progress_callback(message)
            except Exception:
                logger.exception("Progress callback failed for message: %s", message)

    def _register_folder(self, folder_id: str, folder_name: str, parent: Optional[str] = None):
        entry = self.summary_data['folders'].setdefault(
            folder_id,
            {'name': folder_name, 'parent': parent, 'files': []}
        )
        entry['name'] = folder_name
        entry['parent'] = parent
        return entry

    def _record_file(
        self,
        folder_id: str,
        folder_name: str,
        file_name: str,
        mime_type: str,
        char_count: int,
        parent: Optional[str] = None,
    ):
        folder_entry = self._register_folder(folder_id, folder_name, parent)
        folder_entry['files'].append({
            'name': file_name,
            'mime': mime_type,
            'chars': char_count,
        })
        self._report(
            "Fetched doc '%s' (%d chars) from folder '%s'" % (file_name, char_count, folder_name)
        )

    def _resolve_drive_context(self, folder_id: str):
        """Fetch metadata for a folder to determine drive context."""
        try:
            meta = self.drive_service.files().get(
                fileId=folder_id,
                fields='id, name, driveId',
                supportsAllDrives=True,
            ).execute()
            self._report(
                "Resolved folder '%s' (driveId=%s)" % (
                    meta.get('name', folder_id),
                    meta.get('driveId') or 'user-drive',
                )
            )
            return meta
        except Exception as exc:
            logger.exception("Unable to resolve drive context for %s: %s", folder_id, exc)
            return {}

    def _list_drive_items(self, query: str, item_desc: str, drive_id: Optional[str] = None):
        """List Google Drive items with pagination support."""
        logger.debug("Drive query for %s: %s", item_desc, query)
        items = []
        page_token = None

        while True:
            list_kwargs = {
                'q': query,
                'fields': 'nextPageToken, files(id, name, mimeType, shortcutDetails)',
                'pageToken': page_token,
                'pageSize': 100,
                'includeItemsFromAllDrives': True,
                'supportsAllDrives': True,
            }
            if drive_id:
                list_kwargs['corpora'] = 'drive'
                list_kwargs['driveId'] = drive_id
            else:
                list_kwargs['corpora'] = 'allDrives'

            response = self.drive_service.files().list(**list_kwargs).execute()

            batch = response.get('files', [])
            items.extend(batch)
            self._report("Retrieved %d %s (running total=%d)" % (len(batch), item_desc, len(items)))

            page_token = response.get('nextPageToken')
            if not page_token:
                break

        return items

    def _fetch_docs_for_folder(self, folder_id: str, folder_name: str, drive_id: Optional[str] = None):
        """Return list of Google Docs within a specific folder."""
        files_query = (
            f"'{folder_id}' in parents "
            "and trashed = false "
            "and (mimeType='application/vnd.google-apps.document' "
            "or mimeType='application/vnd.openxmlformats-officedocument.wordprocessingml.document' "
            "or mimeType='application/pdf')"
        )
        if drive_id is None:
            folder_meta = self._resolve_drive_context(folder_id)
            drive_id = folder_meta.get('driveId')
            folder_name = folder_meta.get('name', folder_name)
        files = self._list_drive_items(files_query, f"docs in {folder_name}", drive_id=drive_id)
        if not files:
            self._report("No docs found in folder '%s'" % folder_name)
        return files

    def get_document_and_texts(self):
        """Fetch Google Docs, DOCX, and PDF files under the configured folders."""
        documents = []

        for folder_id in self.folder_ids:
            folder_meta = self._resolve_drive_context(folder_id)
            drive_id = folder_meta.get('driveId')
            folder_name = folder_meta.get('name', folder_id)
            self._register_folder(folder_id, folder_name, parent=None)
            self._report("Fetching subfolders for folder '%s'" % folder_name)

            # Fetch first-level subdirectories
            subfolders_query = (
                f"'{folder_id}' in parents "
                "and trashed = false "
                "and (mimeType='application/vnd.google-apps.folder' "
                "or mimeType='application/vnd.google-apps.shortcut')"
            )
            subfolders = self._list_drive_items(subfolders_query, "subfolders", drive_id=drive_id)
            self._report("Found %d subfolders" % len(subfolders))

            # Fetch direct files within the folder
            direct_files = self._fetch_docs_for_folder(folder_id, folder_name, drive_id=drive_id)
            for file in direct_files:
                doc_id = file['id']
                doc_name = file['name']
                self._report("Fetching top-level document: %s" % doc_name)
                mime_type = file.get('mimeType', '')
                if mime_type == 'application/vnd.google-apps.document':
                    text = self.get_google_docs_text(doc_id)
                elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                    text = self.get_docx_text(doc_id, doc_name)
                elif mime_type == 'application/pdf':
                    text = self.get_pdf_text(doc_id, doc_name)
                else:
                    logger.warning("Skipping unsupported mime type %s for %s", mime_type, doc_name)
                    continue
                documents.append({"name": doc_name, "content": text})
                self._record_file(
                    folder_id,
                    folder_name,
                    doc_name,
                    mime_type,
                    len(text),
                    parent=None,
                )

            if not subfolders:
                self._report("No subfolders to process under '%s'" % folder_name)
                continue

            # For each subdirectory, fetch its Google Docs files
            for subfolder in subfolders:
                subfolder_id = subfolder['id']
                subfolder_name = subfolder['name']
                mime_type = subfolder.get('mimeType')
                parent_name = folder_name
                if mime_type == 'application/vnd.google-apps.shortcut':
                    shortcut_details = subfolder.get('shortcutDetails', {})
                    target_id = shortcut_details.get('targetId')
                    target_mime = shortcut_details.get('targetMimeType')
                    if target_mime == 'application/vnd.google-apps.folder' and target_id:
                        self._report(
                            "Resolving folder shortcut '%s' -> %s" % (subfolder_name, target_id)
                        )
                        subfolder_id = target_id
                    else:
                        logger.warning(
                            "Skipping shortcut '%s' (mimeType=%s, targetMime=%s)",
                            subfolder_name,
                            mime_type,
                            target_mime,
                        )
                        continue

                self._register_folder(subfolder_id, subfolder_name, parent=parent_name)
                self._report("Processing subfolder: %s" % subfolder_name)
                files = self._fetch_docs_for_folder(subfolder_id, subfolder_name, drive_id=drive_id)

                for file in files:
                    doc_id = file['id']
                    doc_name = file['name']
                    mime_type = file.get('mimeType', '')
                    self._report("Fetching document: %s" % doc_name)
                    if mime_type == 'application/vnd.google-apps.document':
                        text = self.get_google_docs_text(doc_id)
                    elif mime_type == 'application/vnd.openxmlformats-officedocument.wordprocessingml.document':
                        text = self.get_docx_text(doc_id, doc_name)
                    elif mime_type == 'application/pdf':
                        text = self.get_pdf_text(doc_id, doc_name)
                    else:
                        logger.warning("Skipping unsupported mime type %s for %s", mime_type, doc_name)
                        continue
                    documents.append({"name": doc_name, "content": text})
                    self._record_file(
                        subfolder_id,
                        subfolder_name,
                        doc_name,
                        mime_type,
                        len(text),
                        parent=parent_name,
                    )

        return documents

    def get_google_docs_text(self, doc_id):
        document = self.docs_service.documents().get(documentId=doc_id).execute()
        content = document.get('body', {}).get('content', [])
        output = ''

        for element in content:
            if 'paragraph' in element:
                paragraph = element['paragraph']
                style = paragraph.get('paragraphStyle', {})
                named_style = style.get('namedStyleType', 'NORMAL_TEXT')

                text_content = ""
                if named_style == 'HEADING_2':
                    text_content += '### '

                for text_run in paragraph.get('elements', []):
                    if 'textRun' in text_run:
                        text_content += text_run['textRun']['content']

                output += text_content

        return output

    def get_docx_text(self, file_id: str, file_name: str) -> str:
        if DocxDocument is None:
            raise RuntimeError(
                "python-docx is required to process .docx files. Please install python-docx."
            )

        logger.info("Downloading DOCX file %s", file_name)
        request = self.drive_service.files().get_media(fileId=file_id, supportsAllDrives=True)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                logger.debug("Download progress %s%% for %s", int(status.progress() * 100), file_name)

        fh.seek(0)
        doc = DocxDocument(fh)
        text = "\n".join(paragraph.text for paragraph in doc.paragraphs if paragraph.text)
        logger.info("Extracted %d chars from DOCX %s", len(text), file_name)
        return text

    def get_pdf_text(self, file_id: str, file_name: str) -> str:
        if PdfReader is None:
            raise RuntimeError(
                "pypdf is required to process PDF files. Please install pypdf."
            )

        logger.info("Downloading PDF file %s", file_name)
        request = self.drive_service.files().get_media(fileId=file_id, supportsAllDrives=True)
        fh = io.BytesIO()
        downloader = MediaIoBaseDownload(fh, request)

        done = False
        while not done:
            status, done = downloader.next_chunk()
            if status:
                logger.debug("Download progress %s%% for %s", int(status.progress() * 100), file_name)

        fh.seek(0)
        reader = PdfReader(fh)
        text_parts = []
        for page_num, page in enumerate(reader.pages):
            try:
                page_text = page.extract_text() or ""
            except Exception as exc:  # pragma: no cover - fallback for malformed PDFs
                logger.warning("Failed to extract text from page %d of %s: %s", page_num, file_name, exc)
                page_text = ""
            if page_text:
                text_parts.append(page_text)

        combined_text = "\n".join(text_parts)
        logger.info("Extracted %d chars from PDF %s", len(combined_text), file_name)
        return combined_text

    def save_documents(self, documents):
        if not os.path.exists(self.local_dir_docs):
            os.makedirs(self.local_dir_docs)
            logger.info("Created local doc directory at %s", self.local_dir_docs)
        for i, doc in enumerate(documents):
            with open(f'{self.local_dir_docs}/doc_{i}.txt', "w") as f:
                f.write(doc["content"])
            logger.debug("Saved document %s/doc_%d.txt", self.local_dir_docs, i)

    def create_chunks_with_ids(self, doc_name, chunks):
        for i, chunk in enumerate(chunks):
            chunk.doc_id = f"{doc_name}_chunk_{i}"
        return chunks

    def run(self):
        documents = self.get_document_and_texts()
        logger.info("Total documents fetched: %d", len(documents))
        if not documents:
            logger.warning("No documents fetched; index will be empty")
        self.save_documents(documents)

        # Apply custom chunk splitting
        custom_chunks = custom_chunk_splitter(documents)
        logger.info("Total chunks after cleaning: %d", len(custom_chunks))
        if not custom_chunks:
            logger.warning("No chunks generated; downstream queries will be empty")

        # Persist index with the chunks
        if self.save_index_address and not os.path.exists(self.save_index_address):
            os.makedirs(self.save_index_address)
            logger.info("Created index directory %s", self.save_index_address)

        storage_context = StorageContext.from_defaults()

        index = VectorStoreIndex(
            nodes=custom_chunks,
            storage_context=storage_context,
        )
        logger.info("VectorStoreIndex built with %d nodes", len(custom_chunks))

        if self.save_index_address:
            storage_context.persist(persist_dir=self.save_index_address)
            logger.info("Persisted index to %s", self.save_index_address)

        return index

    def get_summary(self):
        summary_list = []
        for folder_id, data in self.summary_data['folders'].items():
            summary_list.append({
                'folder_id': folder_id,
                'folder_name': data['name'],
                'parent': data['parent'],
                'files': data['files'],
                'file_count': len(data['files']),
            })

        summary_list.sort(key=lambda item: ((item['parent'] or ''), item['folder_name']))
        return summary_list


# Step 5: Query the RAG system
def query_index(query, index):
    logger.info("Running query: %s", query)
    query_engine = index.as_query_engine()
    response = query_engine.query(query)
    response_text = getattr(response, "response", None)
    if response_text is None:
        response_text = str(response)
    logger.info("Query response length: %d", len(response_text.strip()))
    if not response_text.strip():
        logger.warning("Query returned an empty response")
    return response

def configure_logging():
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")


def build_index(
    folder_ids: List[str],
    save_index_path: str = 'Data/google_doc_index',
    progress_callback=None,
) -> tuple[VectorStoreIndex, list[dict]]:
    if not folder_ids:
        raise ValueError("At least one folder ID is required.")

    logger.info("Building index with %d folder(s)", len(folder_ids))
    rag = RagGoogleDoc(
        folder_ids,
        save_index_address=save_index_path,
        progress_callback=progress_callback,
    )
    index = rag.run()
    summary = rag.get_summary()
    return index, summary


def render_ingest_summary(summary):
    if not summary:
        st.info("No documents were ingested.")
        return

    st.markdown("### 📁 Ingested Content Summary")
    lines = []
    for entry in summary:
        parent_note = f" (parent: {entry['parent']})" if entry['parent'] else ""
        lines.append(
            f"- **{entry['folder_name']}** (`{entry['folder_id']}`){parent_note} — {entry['file_count']} file(s)"
        )
        for file_info in entry['files']:
            lines.append(
                f"    - {file_info['name']} — {file_info['mime']} ({file_info['chars']} chars)"
            )

    st.markdown("\n".join(lines))


def run_rag_pipeline(
    folder_ids: List[str],
    question: str,
    save_index_path: str = 'Data/google_doc_index',
    index: Optional[VectorStoreIndex] = None,
) -> str:
    if not folder_ids:
        raise ValueError("At least one folder ID is required.")
    if not question.strip():
        raise ValueError("Question cannot be empty.")

    if index is None:
        index, _ = build_index(folder_ids, save_index_path)

    response = query_index(question, index)
    answer_text = getattr(response, "response", str(response)).strip()
    if not answer_text:
        logger.warning("Final answer was empty")
        return "[empty]"
    return answer_text


def is_running_with_streamlit() -> bool:
    if os.environ.get("STREAMLIT_SERVER_PORT") or os.environ.get("STREAMLIT_RUNTIME"):
        return True
    try:
        from streamlit.runtime.scriptrunner import get_script_run_ctx  # type: ignore
        return get_script_run_ctx() is not None
    except Exception:
        return False


def render_app():
    configure_logging()
    st.set_page_config(page_title="Google Drive RAG", page_icon="📂", layout="wide")

    st.title("📂 Google Drive RAG Explorer")
    st.markdown(
        "Provide Google Drive folder IDs (one per line or comma-separated). The pipeline will recurse the first level, "
        "download Google Docs, DOCX, and PDF files, build a vector index, and answer your question using RAG."
    )

    default_folder_input = "\n".join(DEFAULT_FOLDER_IDS)
    folder_input = st.text_area(
        "Google Drive Folder IDs",
        value=default_folder_input,
        help="Enter one folder ID per line.",
        height=120,
    )

    parsed_folder_ids = [fid.strip() for fid in folder_input.replace(',', '\n').splitlines() if fid.strip()]

    ingest_button = st.button("Ingest Data", type="primary")

    if ingest_button:
        if not parsed_folder_ids:
            st.error("Please provide at least one folder ID before ingesting.")
        else:
            save_index_path = 'Data/google_doc_index'
            with st.status("Ingesting documents and building index...", expanded=True) as status:
                try:
                    def progress_update(message: str):
                        status.write(message)

                    index, summary = build_index(
                        parsed_folder_ids,
                        save_index_path,
                        progress_callback=progress_update,
                    )
                    st.session_state['rag_index'] = index
                    st.session_state['ingested_folders'] = parsed_folder_ids
                    st.session_state['index_path'] = save_index_path
                    st.session_state['ingest_summary'] = summary
                    status.update(label="Ingestion complete.", state="complete")
                    st.success("Ingestion complete. You can now run retrieval.")
                    st.caption(f"Index persisted to `{save_index_path}`.")
                    if summary:
                        render_ingest_summary(summary)
                except Exception as exc:
                    logger.exception("Ingestion failed: %s", exc)
                    status.write(f"Error: {exc}")
                    status.update(label="Ingestion failed.", state="error")
                    st.error(f"Ingestion failed: {exc}")

    st.divider()

    question = st.text_area(
        "Question",
        value=DEFAULT_QUESTION,
        height=140,
    )

    run_button = st.button("Submit", type="primary")

    if run_button:
        if not parsed_folder_ids:
            st.error("Please provide at least one folder ID before running retrieval.")
            return
        if not question.strip():
            st.error("Please enter a question.")
            return

        save_index_path = 'Data/google_doc_index'

        existing_index = st.session_state.get('rag_index')
        ingested_folders = st.session_state.get('ingested_folders') or []
        ingested_path = st.session_state.get('index_path')

        if existing_index is None:
            st.error("Please ingest data first.")
            return

        if ingested_folders != parsed_folder_ids:
            st.error("Folder IDs changed. Please click 'Ingest Data' again before retrieval.")
            return

        if ingested_path != save_index_path:
            st.error("Index configuration changed. Please re-ingest before retrieval.")
            return

        with st.spinner("Running retrieval..."):
            try:
                answer_text = run_rag_pipeline(
                    parsed_folder_ids,
                    question,
                    save_index_path,
                    index=existing_index,
                )

                st.success("Retrieval complete")
                st.markdown("### ✅ Answer")
                st.markdown(answer_text)
                st.caption(f"Index persisted to `{save_index_path}`.")
                summary = st.session_state.get('ingest_summary')
                if summary:
                    render_ingest_summary(summary)
            except Exception as exc:
                logger.exception("Retrieval failed: %s", exc)
                st.error(f"Retrieval failed: {exc}")



if __name__ == "__main__":
    if is_running_with_streamlit():
        render_app()
    else:
        configure_logging()
        logger.info("Detected CLI execution; please use `streamlit run gdrive.py` to launch the app.")
        print("This script is now a Streamlit app. Run it with `streamlit run gdrive.py`.")
