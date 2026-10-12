"""Reuse media configuration; keep the photo embedding space independent."""

import math
from numbers import Real

from media_rag.config import RagError, Settings


MODEL = "gemini-embedding-2"
DIMENSIONS = 768
MAX_PHOTO_BYTES = 15_000_000
MAX_PIXELS = 50_000_000
MAX_IMAGE_BYTES = 1_500_000
MAX_BATCH = 10


def load_settings(overrides=None):
    return Settings.load({**(overrides or {}), "TWELVELABS_API_KEY": "", "TWELVELABS_API_KEY_FILE": ""})


def validate_vector(values):
    if (not isinstance(values, (list, tuple)) or len(values) != DIMENSIONS
            or any(isinstance(value, bool) or not isinstance(value, Real) or not math.isfinite(value)
                   for value in values) or not any(values)):
        raise RagError("Gemini returned an invalid photo embedding. Retry indexing this photo.")
    return [float(value) for value in values]


def validate_question(value):
    if not isinstance(value, str) or not value.strip() or len(value) > 2000:
        raise RagError("Ask a question of 1–2,000 characters.")
    return value.strip()
