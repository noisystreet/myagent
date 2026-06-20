"""Time and date utilities."""

from datetime import datetime
from zoneinfo import ZoneInfo

from ..core.state import ToolResult

# AI-generated


def _get_local_time() -> tuple[datetime, str]:
    """Get current time in the system local timezone."""
    now = datetime.now()
    return now, str(now.astimezone().tzinfo)


def _build_time_data(now: datetime, tz_name: str) -> dict:
    """Build structured time data dict from a datetime."""
    return {
        "timezone": tz_name,
        "iso": now.isoformat(),
        "datetime": now.strftime("%Y-%m-%d %H:%M:%S"),
        "date": now.strftime("%Y-%m-%d"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": now.strftime("%A"),
        "timestamp": int(now.timestamp()),
    }


def _resolve_time(tz: str) -> tuple[datetime, str]:
    """Resolve current time for the given timezone string."""
    if not tz or tz.lower() == "local":
        return _get_local_time()
    zi = ZoneInfo(tz)
    return datetime.now(zi), tz


def get_time(tz: str = "local") -> ToolResult:
    """Get the current date and time.

    Args:
        tz: Timezone string, e.g. "UTC", "Asia/Shanghai", "America/New_York".
            Default "local" uses the system local timezone.
    """
    try:
        now, tz_name = _resolve_time(tz)
        return ToolResult("get_time", True, data=_build_time_data(now, tz_name))
    except Exception as e:
        return ToolResult(
            "get_time",
            False,
            error={"type": "invalid_timezone", "message": str(e)},
        )
