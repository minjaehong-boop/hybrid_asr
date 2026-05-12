from __future__ import annotations


def format_ts(seconds: float | None) -> str:
    """Format seconds to MM:SS.mmm subtitle timestamp."""
    if seconds is None:
        return "--:--.---"
    total_ms = max(0, int(seconds * 1000))
    minutes = total_ms // 60000
    sec = (total_ms % 60000) // 1000
    ms = total_ms % 1000
    return f"{minutes:02d}:{sec:02d}.{ms:03d}"
