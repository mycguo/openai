"""TwelveLabs Embed API v2; explicit Marengo 3.5 and 512-dimensional vectors."""

import math
from pathlib import Path
import time

import requests

from .config import DIMENSIONS, MODEL, RagError
from .models import Segment, validate_vector


class MarengoError(RagError):
    def __init__(self, message, reset_asset=False, reset_task=False):
        super().__init__(message)
        self.reset_asset = reset_asset
        self.reset_task = reset_task


class Marengo:
    BASE_URL = "https://api.twelvelabs.io/v1.3"

    def __init__(self, settings, session=None):
        if not settings.twelvelabs_api_key:
            raise RagError("Set TWELVELABS_API_KEY before indexing recordings or searching.")
        self.settings = settings
        self.session = session or requests.Session()
        self.session.headers.update({"x-api-key": settings.twelvelabs_api_key})

    def request(self, method, path, **kwargs):
        try:
            response = self.session.request(method, self.BASE_URL + path, timeout=(15, 180), **kwargs)
        except requests.RequestException as exc:
            raise MarengoError("TwelveLabs could not be reached. Retry after checking connectivity.") from exc
        if response.status_code >= 400:
            suffix = {401: "Check TWELVELABS_API_KEY.", 403: "Check your account's access to Marengo 3.5.",
                      429: "Wait for the rate limit to reset, then retry."}.get(response.status_code,
                              "Check your account and media format, then retry.")
            # Provider error bodies are intentionally excluded from logs and persistent state.
            raise MarengoError(f"TwelveLabs returned HTTP {response.status_code}. {suffix}",
                               reset_asset=response.status_code == 404 and path.startswith("/assets/"),
                               reset_task=response.status_code == 404 and path.startswith("/embed-v2/tasks/"))
        try:
            payload = response.json()
        except ValueError as exc:
            raise MarengoError("TwelveLabs returned an unreadable response.") from exc
        if not isinstance(payload, dict):
            raise MarengoError("TwelveLabs returned an unexpected response.")
        return payload

    def upload(self, path, mime_type):
        # Upload directly; localhost MinIO URLs cannot be fetched by TwelveLabs.
        with Path(path).open("rb") as handle:
            payload = self.request("POST", "/assets", data={"method": "direct", "enable_hls": "false"},
                                   files={"file": (Path(path).name, handle, mime_type)})
        if not isinstance(payload.get("_id"), str) or not payload["_id"]:
            raise MarengoError("TwelveLabs did not return an uploaded asset ID.")
        return payload["_id"]

    def wait(self, path, heartbeat, is_asset=False):
        deadline = time.monotonic() + self.settings.processing_timeout
        while time.monotonic() < deadline:
            heartbeat()
            payload = self.request("GET", path)
            if payload.get("status") == "ready":
                return payload
            if payload.get("status") == "failed":
                raise MarengoError("TwelveLabs could not process the media. Check the file and retry.",
                                   reset_asset=is_asset, reset_task=not is_asset)
            if payload.get("status") not in {"processing", "pending", "queued"}:
                raise MarengoError("TwelveLabs returned an unknown processing status.")
            time.sleep(self.settings.poll_interval)
        raise MarengoError("Indexing exceeded its time limit. Retry to resume the saved remote task.")

    def wait_asset(self, asset_id, heartbeat):
        return self.wait(f"/assets/{asset_id}", heartbeat, is_asset=True)

    def create_task(self, asset_id, kind, has_audio):
        options = ["audio"] if kind == "audio" else ["visual"] + (["audio"] if has_audio else [])
        media = {
            "media_source": {"asset_id": asset_id},
            "embedding_option": options,
            "embedding_scope": ["clip"],
            # Dynamic segmentation retains the last short segment of the source.
            "segmentation": {"temporal": {"strategy": "dynamic", "dynamic": {"min_duration_sec": 2}}},
        }
        if len(options) > 1:
            media["embedding_type"] = ["separate_embedding"]
        payload = self.request("POST", "/embed-v2/tasks", json={
            "input_type": kind, "model_name": MODEL, "embedding_dimension": DIMENSIONS, kind: media,
        })
        if not isinstance(payload.get("_id"), str) or not payload["_id"]:
            raise MarengoError("TwelveLabs did not return an embedding task ID.")
        return payload["_id"]

    def wait_task(self, task_id, heartbeat):
        return self.wait(f"/embed-v2/tasks/{task_id}", heartbeat)

    def embed_query(self, question):
        payload = self.request("POST", "/embed-v2", json={
            "input_type": "multi_input", "model_name": MODEL,
            "embedding_dimension": DIMENSIONS, "multi_input": {"input_text": question},
        })
        data = payload.get("data")
        if not isinstance(data, list) or len(data) != 1 or not isinstance(data[0], dict):
            raise MarengoError("TwelveLabs did not return one query embedding.")
        return validate_vector(data[0].get("embedding"))


def parse_segments(payload, duration, kind, has_audio):
    if payload.get("metadata", {}).get("embedding_dimension", DIMENSIONS) != DIMENSIONS:
        raise RagError("The returned embeddings use a different vector dimension.")
    data = payload.get("data")
    if not isinstance(data, list) or not data:
        raise RagError("TwelveLabs returned no searchable media segments.")
    allowed = {"audio"} if kind == "audio" else {"visual"} | ({"audio"} if has_audio else set())
    segments = {}
    for item in data:
        if not isinstance(item, dict):
            raise RagError("TwelveLabs returned an invalid media segment.")
        if item.get("embedding_scope") not in {"clip", "local"}:
            continue
        modality = item.get("embedding_option")
        if modality not in allowed:
            raise RagError("TwelveLabs returned an unexpected embedding modality.")
        try:
            start, end = float(item["start_sec"]), float(item["end_sec"])
        except (KeyError, TypeError, ValueError) as exc:
            raise RagError("TwelveLabs returned invalid segment timestamps.") from exc
        if (not all(math.isfinite(value) for value in (start, end))
                or not 0 <= start < min(end, duration) or end > duration + 0.5):
            raise RagError("A returned segment falls outside the source timeline.")
        end = min(duration, end)
        segments[(start, end, modality)] = Segment(start, end, modality, validate_vector(item.get("embedding")))
    if not segments or {item.modality for item in segments.values()} != allowed:
        raise RagError("TwelveLabs did not return all requested media modalities.")
    return list(segments.values())
