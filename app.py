import os
from openai import OpenAI
from googleapiclient.discovery import build
from google.oauth2 import service_account
import streamlit as st
from openai import AsyncOpenAI


# https://github.com/amrrs/chatgpt-googledocs/blob/main/appscript.js
# ─── Configuration ────────────────────────────────────────────────
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
MODEL = "gpt-3.5-turbo"
SERVICE_ACCOUNT_FILE = "./service-account.json"
DOCUMENT_ID = "1vbvbDxvKj6LTWKiahK79XTHZsrhfeZpLUfqf1Ocl6RE"

SCOPES = [
  "https://www.googleapis.com/auth/documents",
  "https://www.googleapis.com/auth/drive"
]


credentials = service_account.Credentials.from_service_account_info(
    st.secrets["gcp_service_account"], scopes=SCOPES
)

docs_service = build("docs", "v1", credentials=credentials)


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
    # Fetch the document to get the current end index
    doc = docs_service.documents().get(documentId=document_id).execute()
    end_index = doc.get("body", {}).get("content", [])[-1].get("endIndex", 1)

    # Create the request to insert text at the end of the document
    requests = [
        {
            "insertText": {
                "location": {"index": end_index - 1},  # Use the valid end index
                "text": text + "\n",
            }
        }
    ]
    docs_service.documents().batchUpdate(
        documentId=document_id, body={"requests": requests}
    ).execute()


def generate_from_openai(prompt: str, temperature: float = 0.0, max_tokens: int = 2060) -> str:
    """Calls ChatCompletions and returns the assistant’s reply."""
    client = OpenAI()
    completion = client.chat.completions.create(model=MODEL, 
        messages=[{"role": "user", "content": prompt}], temperature=temperature, max_tokens=max_tokens)
    return completion.choices[0].message.content.strip()


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



def main():
    st.title("OpenAI GoogleDocs Integrations")
    st.header("let the chatbot coming to you")

    st.write("The GoogleDoc Link: https://docs.google.com/document/d/1vbvbDxvKj6LTWKiahK79XTHZsrhfeZpLUfqf1Ocl6RE/edit?tab=t.0 ")


    button1 = st.button("generate linkedin posts")
    if button1:
        generate_linkedin_posts()
        st.success("LinkedIn posts generated and appended to the document.")

    
    button2 = st.button("generate essay")
    if button2:
        generate_essay()
        st.success('Essay generated successfully!')

    
if __name__ == "__main__":
    main()