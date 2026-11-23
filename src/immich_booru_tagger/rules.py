"""Pure deterministic rules for Booru tag normalization and metadata tagging."""

from __future__ import annotations

import re

MANAGED_CATEGORIES: tuple[str, ...] = (
    "rating",
    "copyright",
    "character",
    "general",
    "meta",
    "artist",
)
PROCESSED_MARKER: str = "meta/booru_tagged"

_MANAGED_PREFIXES: tuple[str, ...] = tuple(f"{category}/" for category in MANAGED_CATEGORIES)
_WHITESPACE_OR_SLASH_RE = re.compile(r"[\s/]+")


def normalize_booru_tag(category: str, tag_name: str) -> str:
    """Normalize a category and raw tag name into a two-level Booru Hierarchical Tag."""
    if category not in MANAGED_CATEGORIES:
        raise ValueError(f"Unsupported managed tag category: {category!r}")

    stripped = tag_name.strip().lower()
    normalized_name = _WHITESPACE_OR_SLASH_RE.sub("_", stripped)
    if not normalized_name or not normalized_name.strip("_"):
        raise ValueError(f"Tag name cannot be empty after normalization: {tag_name!r}")

    return f"{category}/{normalized_name}"


def is_managed_tag(tag_value: str) -> bool:
    """Return True if tag_value belongs to the managed Booru namespace."""
    return tag_value.startswith(_MANAGED_PREFIXES)


def compute_dimension_meta_tags(width: int | None, height: int | None) -> list[str]:
    """Compute deterministic Dimension Meta Tags from original EXIF dimensions."""
    if width is None or height is None or width <= 0 or height <= 0:
        return []

    tags: list[str] = []

    if width >= 3200 or height >= 2400:
        tags.append("meta/absurdres")
    elif width >= 1600 or height >= 1200:
        tags.append("meta/highres")
    elif width <= 500 and height <= 500:
        tags.append("meta/lowres")

    if height * 2 >= width * 5:
        tags.append("meta/tall_image")
    elif width * 2 >= height * 5:
        tags.append("meta/wide_image")

    return tags
