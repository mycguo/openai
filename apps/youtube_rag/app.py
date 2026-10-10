"""Community Cloud entrypoint for the separate youtube-rag application."""

from pathlib import Path
import sys


sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from media_rag_app import main


if __name__ == "__main__":
    main(youtube=True)
