"""Pure deterministic rules for Booru tag normalization and metadata tagging."""

from __future__ import annotations

import os
import re
from collections.abc import Sequence

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


def normalize_extra_tag(raw_tag: str) -> str:
    """Validate and normalize an Album Extra Tag, enforcing Booru format for managed prefixes."""
    stripped = raw_tag.strip()
    if not stripped:
        raise ValueError("extra_tags entries must not be empty")

    category, sep, tag_name = stripped.partition("/")
    if sep and category.lower() in MANAGED_CATEGORIES:
        return normalize_booru_tag(category.lower(), tag_name)
    return stripped


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


def extract_artist_tag(
    file_name: str,
    patterns: Sequence[str | re.Pattern[str]],
) -> str | None:
    """Extract and normalize the first matching artist tag from an asset filename stem."""
    if not file_name or not patterns:
        return None

    stem = os.path.splitext(file_name)[0]
    if not stem:
        return None

    for pattern in patterns:
        compiled = re.compile(pattern) if isinstance(pattern, str) else pattern
        match = compiled.search(stem)
        if match is None:
            continue

        if "artist" in compiled.groupindex:
            raw_artist = match.group("artist")
        elif compiled.groups >= 1:
            raw_artist = match.group(1)
        else:
            continue

        if raw_artist is None:
            continue

        try:
            return normalize_booru_tag("artist", raw_artist)
        except ValueError:
            continue

    return None
