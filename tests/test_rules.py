"""Tests for deterministic Booru tag normalization and dimension meta tag rules."""

import pytest

from immich_booru_tagger.rules import (
    MANAGED_CATEGORIES,
    PROCESSED_MARKER,
    compute_dimension_meta_tags,
    is_managed_tag,
    normalize_booru_tag,
)


def test_constants_match_domain_glossary() -> None:
    assert PROCESSED_MARKER == "meta/booru_tagged"
    assert MANAGED_CATEGORIES == (
        "rating",
        "copyright",
        "character",
        "general",
        "meta",
        "artist",
    )


@pytest.mark.parametrize(
    ("category", "raw_name", "expected"),
    [
        ("copyright", "fate/grand_order", "copyright/fate_grand_order"),
        ("copyright", "Fate / Grand  Order", "copyright/fate_grand_order"),
        ("character", "Rem (Re:Zero)", "character/rem_(re:zero)"),
        ("general", "  1girl   solo  ", "general/1girl_solo"),
        ("rating", "Sensitive", "rating/sensitive"),
        ("meta", "booru_tagged", "meta/booru_tagged"),
        ("artist", "Example Artist (Pixiv)", "artist/example_artist_(pixiv)"),
    ],
)
def test_normalize_booru_tag_formats_valid_inputs(
    category: str, raw_name: str, expected: str
) -> None:
    assert normalize_booru_tag(category, raw_name) == expected


@pytest.mark.parametrize(
    ("category", "raw_name"),
    [
        ("unknown", "tag"),
        ("general", ""),
        ("general", "   "),
        ("general", "/"),
        ("general", " / "),
    ],
)
def test_normalize_booru_tag_rejects_invalid_category_or_empty_name(
    category: str, raw_name: str
) -> None:
    with pytest.raises(ValueError):
        normalize_booru_tag(category, raw_name)


@pytest.mark.parametrize(
    ("tag_value", "expected"),
    [
        ("rating/general", True),
        ("copyright/touhou", True),
        ("character/hakurei_reimu", True),
        ("general/1girl", True),
        ("meta/booru_tagged", True),
        ("artist/alice", True),
        ("favorites/wallpapers", False),
        ("vacation", False),
        ("rating_custom/safe", False),
        ("general", False),
    ],
)
def test_is_managed_tag_checks_namespace_prefix(
    tag_value: str, expected: bool
) -> None:
    assert is_managed_tag(tag_value) is expected


@pytest.mark.parametrize(
    ("width", "height", "expected"),
    [
        (None, None, []),
        (None, 2400, []),
        (3200, None, []),
        (0, 1000, []),
        (1000, 0, []),
        (-500, 1000, []),
        # Exclusive resolution tiers
        (3200, 2000, ["meta/absurdres"]),
        (2000, 2400, ["meta/absurdres"]),
        (3199, 2399, ["meta/highres"]),
        (1600, 1200, ["meta/highres"]),
        (1600, 1000, ["meta/highres"]),
        (1000, 1200, ["meta/highres"]),
        (1599, 1199, []),
        (501, 500, []),
        (500, 501, []),
        (500, 500, ["meta/lowres"]),
        (400, 300, ["meta/lowres"]),
        # Extreme aspect ratios (stacked independently)
        (1000, 2500, ["meta/absurdres", "meta/tall_image"]),
        (2500, 1000, ["meta/highres", "meta/wide_image"]),
        (400, 1000, ["meta/tall_image"]),
        (1000, 400, ["meta/wide_image"]),
        (1000, 401, []),
        (401, 1000, []),
        (200, 500, ["meta/lowres", "meta/tall_image"]),
        (500, 200, ["meta/lowres", "meta/wide_image"]),
    ],
)
def test_compute_dimension_meta_tags_boundary_cases(
    width: int | None, height: int | None, expected: list[str]
) -> None:
    assert compute_dimension_meta_tags(width, height) == expected
