"""Extract semantic text and structured metadata filters from natural-language queries."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Any, Dict, List, Optional, Tuple


@dataclass
class ParsedQuery:
    """Structured output of query understanding."""

    original_query: str
    semantic_text: str
    metadata_filters: Dict[str, Any] = field(default_factory=dict)


_TIME_OF_DAY_RANGES: Dict[str, Tuple[int, int]] = {
    "morning": (6, 12),
    "afternoon": (12, 17),
    "evening": (17, 21),
    "night": (21, 6),
}

_RELATIVE_DATE_PATTERNS = (
    (re.compile(r"\byesterday\b", re.IGNORECASE), -1),
    (re.compile(r"\btoday\b", re.IGNORECASE), 0),
)

_CAMERA_PATTERNS = (
    re.compile(r"\b(?:camera|cam)\s*[_-]?\s*(\w+)\b", re.IGNORECASE),
    re.compile(r"\bgate\s*(\w+)\b", re.IGNORECASE),
)

_TIME_PATTERNS: List[Tuple[re.Pattern[str], str]] = [
    (re.compile(r"\blast\s+(\d+)\s+minutes?\b", re.IGNORECASE), "last_minutes"),
    (re.compile(r"\blast\s+(\d+)\s+hours?\b", re.IGNORECASE), "last_hours"),
    (
        re.compile(
            r"\bbetween\s+(\d{1,2})(?::(\d{2}))?\s*(?:am|pm)?\s+and\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b",
            re.IGNORECASE,
        ),
        "between",
    ),
    (re.compile(r"\bafter\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE), "after"),
    (re.compile(r"\bbefore\s+(\d{1,2})(?::(\d{2}))?\s*(am|pm)?\b", re.IGNORECASE), "before"),
    (re.compile(r"\bbefore\s+noon\b", re.IGNORECASE), "before_noon"),
    (re.compile(r"\bafter\s+noon\b", re.IGNORECASE), "after_noon"),
    (re.compile(r"\b(morning|afternoon|evening|night)\b", re.IGNORECASE), "time_of_day"),
]


def _parse_hour(hour: int, minute: int, meridiem: Optional[str]) -> float:
    """Convert hour/minute/optional am/pm to fractional hour (0-24)."""
    if meridiem:
        mer = meridiem.lower()
        if mer == "pm" and hour < 12:
            hour += 12
        elif mer == "am" and hour == 12:
            hour = 0
    return hour + minute / 60.0


def _day_bounds(reference: datetime, day_offset: int) -> Tuple[float, float]:
    """Return UTC epoch bounds for a calendar day relative to reference."""
    day = (reference + timedelta(days=day_offset)).date()
    start = datetime(day.year, day.month, day.day, tzinfo=timezone.utc)
    end = start + timedelta(days=1)
    return start.timestamp(), end.timestamp()


def _time_of_day_filter(
    label: str,
    day_start: Optional[float],
    day_end: Optional[float],
) -> Dict[str, Any]:
    """Build Chroma filters for a named time-of-day window."""
    start_hour, end_hour = _TIME_OF_DAY_RANGES[label.lower()]
    filters: Dict[str, Any] = {"time_of_day": label.lower()}

    if day_start is not None and day_end is not None:
        day = datetime.fromtimestamp(day_start, tz=timezone.utc).date()
        if start_hour < end_hour:
            window_start = datetime(
                day.year, day.month, day.day, start_hour, 0, tzinfo=timezone.utc
            ).timestamp()
            window_end = datetime(
                day.year, day.month, day.day, end_hour, 0, tzinfo=timezone.utc
            ).timestamp()
        else:
            window_start = datetime(
                day.year, day.month, day.day, start_hour, 0, tzinfo=timezone.utc
            ).timestamp()
            window_end = datetime(
                day.year, day.month, day.day, 23, 59, 59, tzinfo=timezone.utc
            ).timestamp()
        filters["camera_timestamp_gte"] = max(day_start, window_start)
        filters["camera_timestamp_lt"] = min(day_end, window_end + 1)

    return filters


def parse_query(
    query: str,
    reference_time: Optional[datetime] = None,
) -> ParsedQuery:
    """Split a natural-language query into semantic text and metadata filters.

    Temporal constraints are converted to structured filters — never embedded.
    """
    reference = reference_time or datetime.now(tz=timezone.utc)
    remaining = query.strip()
    filters: Dict[str, Any] = {}
    matched_spans: List[Tuple[int, int]] = []

    for pattern, day_offset in _RELATIVE_DATE_PATTERNS:
        match = pattern.search(remaining)
        if match:
            day_start, day_end = _day_bounds(reference, day_offset)
            filters["camera_timestamp_gte"] = day_start
            filters["camera_timestamp_lt"] = day_end
            matched_spans.append(match.span())

    for pattern in _CAMERA_PATTERNS:
        match = pattern.search(remaining)
        if match:
            camera_ref = match.group(1)
            filters["camera_id"] = _normalize_camera_id(camera_ref)
            matched_spans.append(match.span())

    for pattern, kind in _TIME_PATTERNS:
        match = pattern.search(remaining)
        if match is None:
            continue

        if kind == "last_minutes":
            minutes = int(match.group(1))
            filters["camera_timestamp_gte"] = (reference - timedelta(minutes=minutes)).timestamp()
            filters["camera_timestamp_lt"] = reference.timestamp() + 1
        elif kind == "last_hours":
            hours = int(match.group(1))
            filters["camera_timestamp_gte"] = (reference - timedelta(hours=hours)).timestamp()
            filters["camera_timestamp_lt"] = reference.timestamp() + 1
        elif kind == "between":
            start = _parse_hour(int(match.group(1)), int(match.group(2) or 0), match.group(5))
            end = _parse_hour(int(match.group(3)), int(match.group(4) or 0), match.group(5))
            day_start, day_end = _day_bounds(reference, 0)
            day = datetime.fromtimestamp(day_start, tz=timezone.utc).date()
            filters["camera_timestamp_gte"] = datetime(
                day.year, day.month, day.day, int(start), int((start % 1) * 60), tzinfo=timezone.utc
            ).timestamp()
            filters["camera_timestamp_lt"] = (
                datetime(
                    day.year, day.month, day.day, int(end), int((end % 1) * 60), tzinfo=timezone.utc
                ).timestamp()
                + 1
            )
        elif kind == "after":
            hour = _parse_hour(int(match.group(1)), int(match.group(2) or 0), match.group(3))
            day_start, _ = _day_bounds(reference, 0)
            day = datetime.fromtimestamp(day_start, tz=timezone.utc).date()
            filters["camera_timestamp_gte"] = datetime(
                day.year, day.month, day.day, int(hour), int((hour % 1) * 60), tzinfo=timezone.utc
            ).timestamp()
        elif kind == "before":
            hour = _parse_hour(int(match.group(1)), int(match.group(2) or 0), match.group(3))
            day_start, _ = _day_bounds(reference, 0)
            day = datetime.fromtimestamp(day_start, tz=timezone.utc).date()
            filters["camera_timestamp_lt"] = datetime(
                day.year, day.month, day.day, int(hour), int((hour % 1) * 60), tzinfo=timezone.utc
            ).timestamp()
        elif kind == "before_noon":
            day_start, _ = _day_bounds(reference, 0)
            day = datetime.fromtimestamp(day_start, tz=timezone.utc).date()
            filters["camera_timestamp_lt"] = datetime(
                day.year, day.month, day.day, 12, 0, tzinfo=timezone.utc
            ).timestamp()
        elif kind == "after_noon":
            day_start, _ = _day_bounds(reference, 0)
            day = datetime.fromtimestamp(day_start, tz=timezone.utc).date()
            filters["camera_timestamp_gte"] = datetime(
                day.year, day.month, day.day, 12, 0, tzinfo=timezone.utc
            ).timestamp()
        elif kind == "time_of_day":
            label = match.group(1).lower()
            day_start = filters.get("camera_timestamp_gte")
            day_end = filters.get("camera_timestamp_lt")
            tod_filters = _time_of_day_filter(label, day_start, day_end)
            filters.update(tod_filters)

        matched_spans.append(match.span())

    semantic = _extract_semantic_text(remaining, matched_spans)
    if not semantic:
        semantic = query.strip()

    return ParsedQuery(
        original_query=query,
        semantic_text=semantic,
        metadata_filters=filters,
    )


def _normalize_camera_id(camera_ref: str) -> str:
    """Normalize camera references like '4' -> 'cam_4', 'Gate3' -> 'gate3'."""
    ref = camera_ref.strip()
    if ref.isdigit():
        return f"cam_{ref}"
    return ref.lower().replace(" ", "_")


def _extract_semantic_text(text: str, spans: List[Tuple[int, int]]) -> str:
    """Remove matched constraint phrases and return the remaining semantic query."""
    if not spans:
        return re.sub(r"\s+", " ", text).strip()

    merged = sorted(spans)
    cleaned: List[str] = []
    cursor = 0
    for start, end in merged:
        cleaned.append(text[cursor:start])
        cursor = end
    cleaned.append(text[cursor:])

    semantic = " ".join(part.strip() for part in cleaned if part.strip())
    semantic = re.sub(
        r"\b(?:at|on|from|in|during|show|find|all|the)\b", " ", semantic, flags=re.IGNORECASE
    )
    return re.sub(r"\s+", " ", semantic).strip()
