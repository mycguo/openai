"""Treat provider output and recording metadata as text in Markdown widgets."""

import re


def plain_markdown(value):
    # HTML escaping alone does not prevent Markdown images from making requests.
    return re.sub(r"([\\`*_{}\[\]()<>#+.!|~\-])", r"\\\1", str(value))
