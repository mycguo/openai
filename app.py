import os
import openai
from googleapiclient.discovery import build
from google.oauth2 import service_account
import streamlit as st


# https://github.com/amrrs/chatgpt-googledocs/blob/main/appscript.js
# ─── Configuration ────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets("OPENAI_API_KEY")
MODEL = "gpt-3.5-turbo"
SERVICE_ACCOUNT_FILE = "path/to/your-service-account.json"
DOCUMENT_ID = "YOUR_DOC_ID_HERE"

SCOPES = ["https://www.googleapis.com/auth/documents"]
credentials = service_account.Credentials.from_service_account_file(
    SERVICE_ACCOUNT_FILE, scopes=SCOPES
)
docs_service = build("docs", "v1", credentials=credentials)

openai.api_key = OPENAI_API_KEY


# ─── Helpers ──────────────────────────────────────────────────────
def get_full_text(document_id: str) -> str:
    """Fetches the entire body text of the document."""
    doc = docs_service.documents().get(documentId=document_id).execute()
    pieces = doc.get("body", {}).get("content", [])
    text = []
    for el in pieces:
        if "paragraph" in el:
            for run in el["paragraph"]["elements"]:
                txt = run.get("textRun", {}).get("content")
                if txt:
                    text.append(txt)
    return "".join(text).strip()


def append_paragraph(document_id: str, text: str):
    """Appends a new paragraph at the end of the document."""
    requests = [
        {
            "insertText": {
                "location": {"index": 1_000_000},  # large index → end of doc
                "text": text + "\n",
            }
        }
    ]
    docs_service.documents().batchUpdate(
        documentId=document_id, body={"requests": requests}
    ).execute()


def generate_from_openai(prompt: str, temperature: float = 0.0, max_tokens: int = 2060) -> str:
    """Calls ChatCompletions and returns the assistant’s reply."""
    resp = openai.ChatCompletion.create(
        model=MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
    )
    return resp.choices[0].message.content.strip()


# ─── Main features ─────────────────────────────────────────────────
def generate_linkedin_posts():
    """Mimics generateIdeas(): writes 5 LinkedIn posts based on doc text."""
    selected = get_full_text(DOCUMENT_ID)
    prompt = f"Help me write 5 LinkedIn post on {selected}"
    result = generate_from_openai(prompt)
    append_paragraph(DOCUMENT_ID, result)
    print("✅ Appended LinkedIn ideas.")


def generate_essay():
    """Mimics generatePrompt(): writes an essay based on doc text."""
    selected = get_full_text(DOCUMENT_ID)
    prompt = f"Generate an essay on {selected}"
    result = generate_from_openai(prompt)
    append_paragraph(DOCUMENT_ID, result)
    print("✅ Appended essay.")


# ─── CLI Entrypoint ────────────────────────────────────────────────
if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Generate content in a Google Doc via OpenAI."
    )
    parser.add_argument(
        "action", choices=["ideas", "essay"], help="ideas ⇒ LinkedIn posts; essay ⇒ an essay"
    )
    args = parser.parse_args()

    if args.action == "ideas":
        generate_linkedin_posts()
    else:
        generate_essay()


def main():
    st.title("OpenAI GoogleDocs Integrations")
    st.header("let the chatbot coming to you")


    
    
    st.markdown("<div style='height:300px;'></div>", unsafe_allow_html=True)
    st.markdown(""" \n \n \n \n \n \n \n\n\n\n\n\n
        # Footnote on tech stack
        web framework: https://streamlit.io/ \n
        LLM model: "deepseek-ai/deepseek-r1" \n
        vector store: FAISS (Facebook AI Similarity Search) \n
        Embeddings model: GoogleGenerativeAIEmbeddings(model="models/embedding-001") \n
        LangChain: Connect LLMs for Retrieval-Augmented Generation (RAG), memory, chaining and reasoning. \n
        PyPDF2 and docx: for importing PDF and Word \n
        audio: assemblyai https://www.assemblyai.com/ \n
        Video: moviepy https://zulko.github.io/moviepy/ \n
    """)    

if __name__ == "__main__":
    main()