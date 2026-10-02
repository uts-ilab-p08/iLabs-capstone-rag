"""Pull dates and times of day out of a plain-language question.

Rule-based, like the location and camera matching in `filters`. No model call.

Two kinds of answer, because they need different payload fields:

  a date      -> matched against `capture_date`
  a time range -> matched against `start_time_of_day`, seconds since midnight

A time of day cannot be a range over timestamps: "2-4pm" across two days is two
disjoint windows. Hence the separate field, and the separate result here.

Every rule errs toward returning nothing. A missed filter leaves results
slightly broad; a wrong one quietly returns the wrong day's footage.
"""

from __future__ import annotations

import re

HOUR = 3600
DAY_END = 24 * HOUR - 1

MONTHS = {
    "jan": 1, "feb": 2, "mar": 3, "apr": 4, "may": 5, "jun": 6,
    "jul": 7, "aug": 8, "sep": 9, "oct": 10, "nov": 11, "dec": 12,
}

# A clock time: 2, 2pm, 2:30, 2:30 pm, 14:00.
_CLOCK = r"(\d{1,2})(?::(\d{2}))?\s*(am|pm)?"

_RANGE_RE = re.compile(
    rf"(?:from\s+|between\s+)?{_CLOCK}\s*(?:-|–|—|to|and|until|till)\s*{_CLOCK}",
    re.IGNORECASE,
)
_AFTER_RE = re.compile(rf"(?:after|from|since)\s+{_CLOCK}", re.IGNORECASE)
_BEFORE_RE = re.compile(rf"(?:before|until|till|up to)\s+{_CLOCK}", re.IGNORECASE)
_AT_RE = re.compile(rf"(?:at|around|about)\s+{_CLOCK}", re.IGNORECASE)

_ISO_RE = re.compile(r"\b(\d{4})-(\d{1,2})-(\d{1,2})\b")
_MONTH_DAY_RE = re.compile(
    r"\b(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\s+(\d{1,2})(?:st|nd|rd|th)?\b",
    re.IGNORECASE,
)
_DAY_MONTH_RE = re.compile(
    r"\b(\d{1,2})(?:st|nd|rd|th)?\s+(?:of\s+)?"
    r"(jan|feb|mar|apr|may|jun|jul|aug|sep|oct|nov|dec)[a-z]*\.?\b",
    re.IGNORECASE,
)
_DAY_ONLY_RE = re.compile(r"\bthe\s+(\d{1,2})(?:st|nd|rd|th)\b", re.IGNORECASE)


class Ambiguous(Exception):
    """The text names a time, but not one we can pin down safely."""


def _clock_seconds(hour: str, minute: str | None, meridiem: str | None,
                   inherited: str | None = None) -> int:
    """One clock reading to seconds since midnight.

    Raises Ambiguous when am/pm is absent and the hour could mean either, e.g.
    "2" could be 02:00 or 14:00. Our footage runs 11:00-17:00 so guessing "pm"
    would usually be right, and silently wrong the rest of the time. An hour of
    13 or more is already unambiguous.
    """
    h, m = int(hour), int(minute or 0)
    if h > 23 or m > 59:
        raise Ambiguous("not a real time")

    suffix = (meridiem or inherited or "").lower()
    if suffix:
        if h > 12:
            raise Ambiguous("hour contradicts am/pm")
        if suffix == "pm" and h != 12:
            h += 12
        elif suffix == "am" and h == 12:
            h = 0
    elif h <= 12 and minute is None:
        raise Ambiguous("no am/pm given")

    return h * HOUR + m * 60


def strip_dates(text: str) -> str:
    """Blank out date expressions so the clock parser cannot misread them.

    "2018-03-07" contains "03-07", which looks exactly like the range "3 to 7",
    and "between March 5 and March 7" looks like "5 and 7". Dates must be
    removed before times are read, or every dated question reports an ambiguous
    time.
    """
    for pattern in (_ISO_RE, _MONTH_DAY_RE, _DAY_MONTH_RE, _DAY_ONLY_RE):
        text = pattern.sub(" ", text)
    return text


def time_of_day_range(text: str) -> tuple[int, int] | None:
    """Seconds-since-midnight window named by the question, if any.

    Expects date expressions to already be stripped (see `strip_dates`).
    Raises Ambiguous if a time is clearly meant but cannot be pinned down.
    """
    match = _RANGE_RE.search(text)
    if match:
        h1, m1, mer1, h2, m2, mer2 = match.groups()
        # "2-4pm": the first half inherits pm from the second.
        start = _clock_seconds(h1, m1, mer1, inherited=mer2)
        end = _clock_seconds(h2, m2, mer2, inherited=mer1)
        if end < start:
            raise Ambiguous("range appears to run backwards")
        return start, end

    match = _AFTER_RE.search(text)
    if match:
        return _clock_seconds(*match.groups()), DAY_END

    match = _BEFORE_RE.search(text)
    if match:
        return 0, _clock_seconds(*match.groups())

    match = _AT_RE.search(text)
    if match:
        start = _clock_seconds(*match.groups())
        return start, min(start + HOUR, DAY_END)

    return None


def dates(text: str, known: frozenset[str]) -> tuple[str, ...]:
    """Dates named by the question, resolved against what is indexed.

    A fully written date is used even when the index has no footage from it —
    "nothing was recorded that day" is the honest answer, and better than
    quietly returning some other day.

    A partial date ("the 7th") is resolved against the indexed dates, in the
    same spirit as rejecting an unknown camera ID. It only resolves when exactly
    one indexed date fits; several means ambiguous, and ambiguous means no
    filter.
    """
    found: set[str] = set()

    for year, month, day in _ISO_RE.findall(text):
        found.add(f"{int(year):04d}-{int(month):02d}-{int(day):02d}")

    for month_name, day in _MONTH_DAY_RE.findall(text):
        found |= _resolve(known, month=MONTHS[month_name[:3].lower()], day=int(day))
    for day, month_name in _DAY_MONTH_RE.findall(text):
        found |= _resolve(known, month=MONTHS[month_name[:3].lower()], day=int(day))

    if not found:
        for day in _DAY_ONLY_RE.findall(text):
            matches = _resolve(known, day=int(day))
            if len(matches) > 1:
                raise Ambiguous(f"'the {day}th' matches more than one indexed date")
            found |= matches

    return tuple(sorted(found))


def _resolve(known: frozenset[str], day: int, month: int | None = None) -> set[str]:
    """Indexed dates matching this day, and month when one was given."""
    out = set()
    for iso in known:
        try:
            y, m, d = (int(part) for part in iso.split("-"))
        except ValueError:
            continue
        if d == day and (month is None or m == month):
            out.add(iso)
    # A written-out date with a year we do not hold still filters, on purpose.
    if not out and month is not None:
        years = {iso.split("-")[0] for iso in known}
        if len(years) == 1:
            out.add(f"{next(iter(years))}-{month:02d}-{day:02d}")
    return out
