"""Run with python -m media_rag.worker; --once processes at most one queued asset."""

import argparse
import time

from .config import RagError, Settings
from .marengo import Marengo, MarengoError
from .media import check_tools
from .service import MediaLibrary, index_asset


def run_once(library, marengo):
    asset = library.database.claim()
    if asset is None:
        return False
    try:
        index_asset(library, asset, marengo)
        print(f"Indexed asset {asset['id']}", flush=True)
    except Exception as exc:
        message = str(exc) if isinstance(exc, RagError) else "Indexing failed. Check storage and worker connectivity, then retry."
        library.database.fail(asset, message,
                              reset_asset=isinstance(exc, MarengoError) and exc.reset_asset,
                              reset_task=isinstance(exc, MarengoError) and exc.reset_task)
        print(f"Asset {asset['id']} failed ({type(exc).__name__}); retry from the library.", flush=True)
    return True


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--once", action="store_true")
    args = parser.parse_args()
    try:
        settings = Settings.load()
        if settings.missing_infrastructure():
            raise RagError("Configure .env.media-rag before starting the worker.")
        check_tools()
        marengo = Marengo(settings)
        library = MediaLibrary(settings)
        library.initialize()
        print("Native media indexing worker ready.", flush=True)
        while True:
            processed = run_once(library, marengo)
            if args.once:
                return 0
            if not processed:
                time.sleep(settings.poll_interval)
    except KeyboardInterrupt:
        return 0
    except Exception as exc:
        print(str(exc) if isinstance(exc, RagError) else
              "Worker could not connect to the media library. Check the local services.", flush=True)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
