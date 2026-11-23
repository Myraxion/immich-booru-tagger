"""Tests for relative and absolute time bound parsing."""

from datetime import datetime, timedelta, timezone

import pytest

from immich_booru_tagger.timefilter import parse_time_bound


TZ_PLUS_8 = timezone(timedelta(hours=8))
FIXED_NOW = datetime(2026, 3, 31, 15, 30, 45, tzinfo=TZ_PLUS_8)


@pytest.mark.parametrize(
    ("expr", "expected"),
    [
        ("30m", datetime(2026, 3, 31, 15, 0, 45, tzinfo=TZ_PLUS_8)),
        ("2h", datetime(2026, 3, 31, 13, 30, 45, tzinfo=TZ_PLUS_8)),
        ("7d", datetime(2026, 3, 24, 15, 30, 45, tzinfo=TZ_PLUS_8)),
        ("2w", datetime(2026, 3, 17, 15, 30, 45, tzinfo=TZ_PLUS_8)),
        ("1mo", datetime(2026, 2, 28, 15, 30, 45, tzinfo=TZ_PLUS_8)),
        ("14mo", datetime(2025, 1, 31, 15, 30, 45, tzinfo=TZ_PLUS_8)),
        ("1y", datetime(2025, 3, 31, 15, 30, 45, tzinfo=TZ_PLUS_8)),
        (" 1MO ", datetime(2026, 2, 28, 15, 30, 45, tzinfo=TZ_PLUS_8)),
    ],
)
def test_parse_relative_time_bounds(expr: str, expected: datetime) -> None:
    assert parse_time_bound(expr, now=FIXED_NOW) == expected


def test_parse_absolute_date_and_iso_timestamps() -> None:
    assert parse_time_bound("2026-01-15", now=FIXED_NOW, end_of_day=False) == datetime(
        2026, 1, 15, 0, 0, 0, tzinfo=TZ_PLUS_8
    )
    assert parse_time_bound("2026-01-15", now=FIXED_NOW, end_of_day=True) == datetime(
        2026, 1, 15, 23, 59, 59, 999999, tzinfo=TZ_PLUS_8
    )
    assert parse_time_bound("2026-01-15T12:00:00", now=FIXED_NOW) == datetime(
        2026, 1, 15, 12, 0, 0, tzinfo=TZ_PLUS_8
    )
    assert parse_time_bound("2026-01-15T12:00:00Z", now=FIXED_NOW) == datetime(
        2026, 1, 15, 12, 0, 0, tzinfo=timezone.utc
    )


@pytest.mark.parametrize("bad_expr", ["", "   ", "0d", "-1d", "7x", "2026-13-01", "yesterday"])
def test_parse_time_bound_rejects_invalid_expressions(bad_expr: str) -> None:
    with pytest.raises(ValueError):
        parse_time_bound(bad_expr, now=FIXED_NOW)
