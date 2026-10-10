"""YouTube-native RAG: streamlit run youtube_rag_app.py --server.port 8504."""

from media_rag_app import main


if __name__ == "__main__":
    main(youtube=True)
