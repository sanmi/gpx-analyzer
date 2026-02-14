"""Formatting utilities for display."""


def format_duration(seconds: float) -> str:
    """Format seconds as Xh Ym string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    if hours > 0:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"


def format_duration_long(seconds: float) -> str:
    """Format seconds as Xh Ym Zs string."""
    hours = int(seconds // 3600)
    minutes = int((seconds % 3600) // 60)
    secs = int(seconds % 60)
    return f"{hours}h {minutes:02d}m {secs:02d}s"


def format_time_diff(seconds1: float, seconds2: float) -> str:
    """Format time difference as +/- Xh Ym."""
    diff = seconds1 - seconds2
    sign = "+" if diff > 0 else ""
    return sign + format_duration(abs(diff))


def format_diff(val1: float, val2: float, unit: str, decimals: int = 0) -> str:
    """Format numeric difference with sign and unit."""
    diff = val1 - val2
    sign = "+" if diff > 0 else ""
    if decimals == 0:
        return f"{sign}{diff:.0f} {unit}"
    return f"{sign}{diff:.{decimals}f} {unit}"


def format_pct_diff(val1: float, val2: float) -> str:
    """Format percentage point difference."""
    diff = val1 - val2
    sign = "+" if diff > 0 else ""
    return f"{sign}{diff:.1f}%"
