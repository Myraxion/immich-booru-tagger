"""Relative and absolute time bound parsing for asset filtering."""

from __future__ import annotations

import calendar
import re
from datetime import date, datetime, time, timedelta, timezone

_RELATIVE_PATTERN = re.compile(r"^([1-9]\d*)\s*(m|h|d|w|mo|y)$", re.IGNORECASE)
_DATE_ONLY_PATTERN = re.compile(r"^\d{4}-\d{2}-\d{2}$")

_FIXED_UNITS = {
    "m": lambda n: timedelta(minutes=n),
    "h": lambda n: timedelta(hours=n),
    "d": lambda n: timedelta(days=n),
    "w": lambda n: timedelta(weeks=n),
}


def _subtract_months(dt: datetime, months: int) -> datetime:
    total_months = dt.year * 12 + (dt.month - 1) - months
    year, month_zero = divmod(total_months, 12)
    month = month_zero + 1
    if year < 1:
        raise ValueError(f"Resulting year out of range when subtracting {months} months")
    last_day = calendar.monthrange(year, month)[1]
    return dt.replace(year=year, month=month, day=min(dt.day, last_day))


def ensure_aware(dt: datetime, *, default_tz: timezone | None = None) -> datetime:
    """Return a timezone-aware datetime, defaulting naive values to local or given tz."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=default_tz) if default_tz is not None else dt.astimezone()
    return dt


def validate_time_range(
    created_after: datetime | None,
    created_before: datetime | None,
) -> None:
    """Raise ValueError when both bounds are set and created_after > created_before."""
    if created_after is None or created_before is None:
        return
    if ensure_aware(created_after) > ensure_aware(created_before):
        raise ValueError("Time range lower bound (--since) must not be later than upper bound (--until)")


def parse_time_bound(
    expr: str,
    *,
    now: datetime | None = None,
    end_of_day: bool = False,
) -> datetime:
    """Parse a relative duration (e.g. 7d, 2mo, 1y) or ISO date/datetime string."""
    cleaned = expr.strip()
    if not cleaned:
        raise ValueError("Time expression must not be empty")

    reference = ensure_aware(now) if now is not None else datetime.now().astimezone()

    if match := _RELATIVE_PATTERN.match(cleaned):
        amount = int(match.group(1))
        unit = match.group(2).lower()
        if unit in _FIXED_UNITS:
            return reference - _FIXED_UNITS[unit](amount)
        months = amount * 12 if unit == "y" else amount
        return _subtract_months(reference, months)

    if _DATE_ONLY_PATTERN.match(cleaned):
        try:
            parsed_date = date.fromisoformat(cleaned)
        except ValueError as exc:
            raise ValueError(f"Invalid date {expr!r}: {exc}") from exc
        boundary_time = time(23, 59, 59, 999999) if end_of_day else time(0, 0, 0)
        return datetime.combine(parsed_date, boundary_time, tzinfo=reference.tzinfo)

    try:
        parsed_dt = datetime.fromisoformat(cleaned)
    except ValueError as exc:
        raise ValueError(
            f"Invalid time expression {expr!r}. Expected relative duration "
            "(e.g. 30m, 2h, 7d, 2w, 3mo, 1y) or ISO date/datetime (e.g. 2026-01-15)."
        ) from exc

    return parsed_dt.replace(tzinfo=reference.tzinfo) if parsed_dt.tzinfo is None else parsed_dt
