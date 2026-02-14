"""
Vex Memory Temporal Reasoning Module
=====================================

Robust temporal expression parsing and time-based memory search.
Handles natural language date expressions like "last Tuesday", "yesterday",
"this week", "2 days ago", "last month", etc.
"""

import re
import logging
from datetime import datetime, date, time, timedelta
from typing import List, Dict, Optional, Any, Tuple

from dateutil import parser as dateutil_parser
from dateutil.relativedelta import relativedelta, MO, TU, WE, TH, FR, SA, SU

import db

logger = logging.getLogger(__name__)

DAY_MAP = {
    'monday': MO, 'tuesday': TU, 'wednesday': WE, 'thursday': TH,
    'friday': FR, 'saturday': SA, 'sunday': SU,
}

# Patterns ordered by specificity (most specific first)
RELATIVE_PATTERNS = [
    # "N days/weeks/months/years ago"
    (r'(\d+)\s+(day|week|month|year)s?\s+ago', 'relative_ago'),
    # "last/this/next week/month/year"
    (r'(last|this|next)\s+(week|month|year)', 'relative_period'),
    # "last/next Monday/Tuesday/..."
    (r'(last|next)\s+(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', 'relative_day'),
    # Simple keywords
    (r'\byesterday\b', 'yesterday'),
    (r'\btoday\b', 'today'),
    (r'\btomorrow\b', 'tomorrow'),
    # "past N days/weeks"
    (r'(?:the\s+)?(?:past|last)\s+(\d+)\s+(day|week|month|year)s?', 'past_n'),
    # "since DATE"
    (r'since\s+(.+?)(?:\s*$|[,.])', 'since'),
    # "in January", "in February", etc.
    (r'\bin\s+(january|february|march|april|may|june|july|august|september|october|november|december)\b', 'in_month'),
]

COMPILED_PATTERNS = [(re.compile(p, re.IGNORECASE), tag) for p, tag in RELATIVE_PATTERNS]


def parse_temporal_expression(text: str, reference_date: Optional[date] = None) -> Optional[Dict[str, Any]]:
    """
    Parse temporal expressions from natural language text.

    Returns:
        dict with keys: start (datetime), end (datetime), type ("exact"|"range"|"relative"), expression (str)
        or None if no temporal expression found
    """
    ref = reference_date or date.today()
    ref_dt = datetime.combine(ref, time.min)
    text_lower = text.lower().strip()

    for pattern, tag in COMPILED_PATTERNS:
        m = pattern.search(text_lower)
        if not m:
            continue

        if tag == 'yesterday':
            d = ref - timedelta(days=1)
            return _day_result(d, "relative", "yesterday")

        elif tag == 'today':
            return _day_result(ref, "relative", "today")

        elif tag == 'tomorrow':
            d = ref + timedelta(days=1)
            return _day_result(d, "relative", "tomorrow")

        elif tag == 'relative_ago':
            n = int(m.group(1))
            unit = m.group(2)
            delta = _make_delta(n, unit)
            d = ref - delta if isinstance(delta, timedelta) else ref - delta
            if unit in ('day',):
                return _day_result(d, "relative", m.group(0))
            else:
                start, end = _period_bounds(d, unit)
                return {"start": datetime.combine(start, time.min),
                        "end": datetime.combine(end, time(23, 59, 59)),
                        "type": "range", "expression": m.group(0)}

        elif tag == 'relative_period':
            which = m.group(1)  # last/this/next
            unit = m.group(2)   # week/month/year
            start, end = _resolve_period(ref, which, unit)
            return {"start": datetime.combine(start, time.min),
                    "end": datetime.combine(end, time(23, 59, 59)),
                    "type": "range", "expression": m.group(0)}

        elif tag == 'relative_day':
            which = m.group(1)
            day_name = m.group(2)
            weekday = DAY_MAP[day_name]
            if which == 'last':
                d = ref + relativedelta(weekday=weekday(-1))
                if d >= ref:
                    d -= timedelta(weeks=1)
            else:  # next
                d = ref + relativedelta(weekday=weekday(+1))
                if d <= ref:
                    d += timedelta(weeks=1)
            return _day_result(d, "relative", m.group(0))

        elif tag == 'past_n':
            n = int(m.group(1))
            unit = m.group(2)
            delta = _make_delta(n, unit)
            start = ref - delta if isinstance(delta, timedelta) else ref - delta
            return {"start": datetime.combine(start, time.min),
                    "end": datetime.combine(ref, time(23, 59, 59)),
                    "type": "range", "expression": m.group(0)}

        elif tag == 'since':
            date_str = m.group(1).strip()
            parsed = _try_parse_date(date_str, ref)
            if parsed:
                return {"start": datetime.combine(parsed, time.min),
                        "end": datetime.combine(ref, time(23, 59, 59)),
                        "type": "range", "expression": m.group(0)}

        elif tag == 'in_month':
            month_name = m.group(1)
            month_num = _month_number(month_name)
            # Use current year, or last year if month is in the future
            year = ref.year
            if month_num > ref.month:
                year -= 1
            start = date(year, month_num, 1)
            if month_num == 12:
                end = date(year + 1, 1, 1) - timedelta(days=1)
            else:
                end = date(year, month_num + 1, 1) - timedelta(days=1)
            return {"start": datetime.combine(start, time.min),
                    "end": datetime.combine(end, time(23, 59, 59)),
                    "type": "range", "expression": m.group(0)}

    # Try dateutil as fallback for explicit dates like "February 11" or "2026-02-11"
    parsed = _try_parse_date(text, ref)
    if parsed:
        return _day_result(parsed, "exact", text.strip())

    return None


def _day_result(d: date, type_: str, expr: str) -> Dict[str, Any]:
    return {
        "start": datetime.combine(d, time.min),
        "end": datetime.combine(d, time(23, 59, 59)),
        "type": type_,
        "expression": expr,
    }


def _make_delta(n: int, unit: str):
    if unit == 'day':
        return timedelta(days=n)
    elif unit == 'week':
        return timedelta(weeks=n)
    elif unit == 'month':
        return relativedelta(months=n)
    elif unit == 'year':
        return relativedelta(years=n)
    return timedelta(days=n)


def _period_bounds(d: date, unit: str) -> Tuple[date, date]:
    if unit == 'week':
        start = d - timedelta(days=d.weekday())
        end = start + timedelta(days=6)
    elif unit == 'month':
        start = d.replace(day=1)
        if d.month == 12:
            end = date(d.year + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(d.year, d.month + 1, 1) - timedelta(days=1)
    elif unit == 'year':
        start = date(d.year, 1, 1)
        end = date(d.year, 12, 31)
    else:
        start = end = d
    return start, end


def _resolve_period(ref: date, which: str, unit: str) -> Tuple[date, date]:
    if unit == 'week':
        monday = ref - timedelta(days=ref.weekday())
        if which == 'last':
            start = monday - timedelta(weeks=1)
        elif which == 'next':
            start = monday + timedelta(weeks=1)
        else:
            start = monday
        end = start + timedelta(days=6)
    elif unit == 'month':
        if which == 'last':
            d = ref.replace(day=1) - timedelta(days=1)
        elif which == 'next':
            if ref.month == 12:
                d = date(ref.year + 1, 1, 15)
            else:
                d = date(ref.year, ref.month + 1, 15)
        else:
            d = ref
        start = d.replace(day=1)
        if d.month == 12:
            end = date(d.year + 1, 1, 1) - timedelta(days=1)
        else:
            end = date(d.year, d.month + 1, 1) - timedelta(days=1)
    elif unit == 'year':
        if which == 'last':
            y = ref.year - 1
        elif which == 'next':
            y = ref.year + 1
        else:
            y = ref.year
        start = date(y, 1, 1)
        end = date(y, 12, 31)
    else:
        start = end = ref
    return start, end


def _month_number(name: str) -> int:
    months = ['january', 'february', 'march', 'april', 'may', 'june',
              'july', 'august', 'september', 'october', 'november', 'december']
    return months.index(name.lower()) + 1


def _try_parse_date(text: str, ref: date) -> Optional[date]:
    """Try to parse an explicit date from text using dateutil."""
    # Clean the text - extract just the date-like part
    text = text.strip()
    if len(text) < 3:
        return None
    try:
        parsed = dateutil_parser.parse(text, default=datetime(ref.year, ref.month, ref.day),
                                        fuzzy=True)
        return parsed.date()
    except (ValueError, OverflowError):
        return None


# ---------------------------------------------------------------------------
# Memory search functions
# ---------------------------------------------------------------------------

def temporal_search(query: str, memories: list, reference_date: Optional[date] = None) -> list:
    """
    Search memories by temporal expressions found in the query.

    Args:
        query: Natural language query possibly containing temporal references
        memories: List of memory objects (dicts or MemoryNode-like with event_time and content)
        reference_date: Reference date for relative expressions (default: today)

    Returns:
        Filtered list of memories matching the temporal criteria
    """
    parsed = parse_temporal_expression(query, reference_date)
    if not parsed:
        return []

    start = parsed["start"]
    end = parsed["end"]
    results = []

    for mem in memories:
        # Check event_time
        event_time = _get_event_time(mem)
        if event_time and start <= event_time <= end:
            results.append(mem)
            continue

        # Check content for date mentions
        content = _get_content(mem)
        if content and _content_mentions_date_range(content, start.date(), end.date()):
            results.append(mem)

    # Sort chronologically
    results.sort(key=lambda m: _get_event_time(m) or datetime.min)
    return results


def get_timeline(start_date, end_date) -> list:
    """
    Get all memories in a date range, ordered chronologically.
    Queries the database directly.

    Args:
        start_date: Start date (date or datetime or str)
        end_date: End date (date or datetime or str)

    Returns:
        List of memory dicts ordered by event_time
    """
    start_dt = _coerce_datetime(start_date)
    end_dt = _coerce_datetime(end_date, end_of_day=True)

    try:
        with db.get_cursor() as cur:
            cur.execute(
                """SELECT id::text, type::text, content, importance_score, decay_factor,
                          access_count, source, event_time, created_at, metadata
                   FROM memory_nodes
                   WHERE event_time IS NOT NULL
                     AND event_time >= %s AND event_time <= %s
                   ORDER BY event_time ASC""",
                (start_dt, end_dt),
            )
            return cur.fetchall()
    except Exception as e:
        logger.error(f"Timeline query failed: {e}")
        return []


def whats_changed_since(since_date) -> list:
    """
    Get memories created or modified after a given date.

    Args:
        since_date: date, datetime, or ISO string

    Returns:
        List of memory dicts ordered by created_at
    """
    since_dt = _coerce_datetime(since_date)

    try:
        with db.get_cursor() as cur:
            cur.execute(
                """SELECT id::text, type::text, content, importance_score, decay_factor,
                          access_count, source, event_time, created_at, metadata
                   FROM memory_nodes
                   WHERE created_at >= %s
                   ORDER BY created_at ASC""",
                (since_dt,),
            )
            return cur.fetchall()
    except Exception as e:
        logger.error(f"whats_changed_since query failed: {e}")
        return []


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _get_event_time(mem) -> Optional[datetime]:
    if isinstance(mem, dict):
        et = mem.get('event_time')
    else:
        et = getattr(mem, 'event_time', None)
    if et is None:
        return None
    if isinstance(et, datetime):
        return et
    if isinstance(et, date):
        return datetime.combine(et, time.min)
    if isinstance(et, str):
        try:
            return datetime.fromisoformat(et)
        except ValueError:
            return None
    return None


def _get_content(mem) -> Optional[str]:
    if isinstance(mem, dict):
        return mem.get('content', '')
    return getattr(mem, 'content', '')


def _content_mentions_date_range(content: str, start: date, end: date) -> bool:
    """Check if content text mentions any date within the range."""
    # Look for ISO dates
    iso_dates = re.findall(r'\d{4}-\d{2}-\d{2}', content)
    for ds in iso_dates:
        try:
            d = date.fromisoformat(ds)
            if start <= d <= end:
                return True
        except ValueError:
            continue

    # Look for written dates like "February 11" or "Feb 11, 2026"
    try:
        parsed = dateutil_parser.parse(content, fuzzy=True)
        if start <= parsed.date() <= end:
            return True
    except (ValueError, OverflowError):
        pass

    return False


def _coerce_datetime(d, end_of_day=False) -> datetime:
    """Coerce various date types to datetime."""
    if isinstance(d, datetime):
        return d
    if isinstance(d, date):
        t = time(23, 59, 59) if end_of_day else time.min
        return datetime.combine(d, t)
    if isinstance(d, str):
        parsed = dateutil_parser.parse(d)
        if end_of_day and parsed.hour == 0 and parsed.minute == 0:
            parsed = parsed.replace(hour=23, minute=59, second=59)
        return parsed
    raise ValueError(f"Cannot coerce {type(d)} to datetime")
