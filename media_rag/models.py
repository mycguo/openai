"""Validated records shared by ingestion, retrieval, and answer generation."""

from dataclasses import dataclass, field
import math

from .config import DIMENSIONS, RagError


def validate_vector(values):
    try:
        vector = [float(value) for value in values]
    except (TypeError, ValueError, OverflowError) as exc:
        raise RagError("The embedding provider returned an invalid vector.") from exc
    if len(vector) != DIMENSIONS or not all(math.isfinite(value) for value in vector):
        raise RagError(f"Expected a finite {DIMENSIONS}-dimension Marengo embedding.")
    if not any(value != 0 for value in vector):
        raise RagError("The embedding provider returned a zero vector.")
    return vector


@dataclass(frozen=True)
class Segment:
    start: float
    end: float
    modality: str
    vector: list[float] = field(repr=False)


@dataclass(frozen=True)
class Hit:
    asset_id: str
    title: str
    object_key: str
    kind: str
    duration: float
    start: float
    end: float
    score: float
    modalities: tuple[str, ...]
    has_audio: bool = True
    source_url: str = ""


@dataclass(frozen=True)
class Evidence:
    source_id: int
    title: str
    kind: str
    start: float
    end: float
    object_key: str
    source_url: str = ""


def timestamp(seconds):
    seconds = max(0, int(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{hours:02}:{minutes:02}:{seconds:02}" if hours else f"{minutes:02}:{seconds:02}"


def merge_hits(hits, limit=5, context_seconds=8.0):
    """Collapse modality duplicates and adjacent moments within a bounded clip."""
    merged = []
    for hit in sorted(hits, key=lambda item: item.score, reverse=True):
        start = max(0.0, hit.start - context_seconds)
        end = min(hit.duration, hit.end + context_seconds)
        if end - start > 90:
            # A coarse provider segment can exceed the Gemini evidence budget.
            # Keep a bounded window centered inside the matching segment.
            center = (hit.start + hit.end) / 2
            start = max(0.0, min(center - 45, hit.duration - 90))
            end = min(hit.duration, start + 90)
        index = next((i for i, item in enumerate(merged)
                      if item.asset_id == hit.asset_id
                      and start <= item.end and end >= item.start
                      and max(end, item.end) - min(start, item.start) <= 90), None)
        if index is not None:
            old = merged[index]
            merged[index] = Hit(
                old.asset_id, old.title, old.object_key, old.kind, old.duration,
                min(old.start, start), max(old.end, end), max(old.score, hit.score),
                tuple(sorted(set(old.modalities + hit.modalities))), old.has_audio, old.source_url,
            )
        elif len(merged) < limit:
            merged.append(Hit(
                hit.asset_id, hit.title, hit.object_key, hit.kind, hit.duration,
                start, end, hit.score, hit.modalities, hit.has_audio, hit.source_url,
            ))
    return merged
