"""Community Cloud entrypoint with dependencies isolated from the other apps."""

from pathlib import Path
import sys


# Streamlit adds the entrypoint directory to sys.path; the shared app lives at root.
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from media_rag_app import main


if __name__ == "__main__":
    main()
