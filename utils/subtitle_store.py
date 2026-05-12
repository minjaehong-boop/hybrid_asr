from __future__ import annotations

from dataclasses import dataclass
from difflib import SequenceMatcher


@dataclass(slots=True)
class SubtitleLine:
    start_sec: float
    end_sec: float
    text: str
    source: str


class SubtitleStore:
    """Accumulate REFINE confirmed text and overlay onto LIVE output.

    Strategy: REFINE chunks (already deduped by CC) are concatenated into
    ``confirmed_text``.  When a LIVE update arrives, we find where confirmed
    coverage ends within LIVE via compact SequenceMatcher, then display
    ``confirmed + live_suffix``.
    """

    def __init__(self) -> None:
        self.live_line: SubtitleLine | None = None
        self.live_display_line: SubtitleLine | None = None
        self.last_refine_display_text: str | None = None

        # Growing confirmed text from REFINE chunks
        self.confirmed_text: str = ""
        self._confirmed_compact: str = ""

    # ------------------------------------------------------------------
    # LIVE
    # ------------------------------------------------------------------

    def update_live(self, line: SubtitleLine) -> SubtitleLine:
        """Return display line: confirmed prefix + live suffix."""
        self.live_line = line

        if not self.confirmed_text:
            self.live_display_line = line
            return line

        display_text = self._overlay(line.text)
        self.live_display_line = SubtitleLine(
            start_sec=line.start_sec,
            end_sec=line.end_sec,
            text=display_text,
            source=line.source,
        )
        return self.live_display_line

    # ------------------------------------------------------------------
    # REFINE
    # ------------------------------------------------------------------

    def apply_refine(self, line: SubtitleLine, now_sec: float, is_final: bool = False) -> bool:
        """Append REFINE text to confirmed and return True if accepted."""
        if not line.text.strip():
            return False

        new_text = line.text.strip()

        if self.confirmed_text:
            merged = self.confirmed_text + " " + new_text
        else:
            merged = new_text

        self.confirmed_text = merged
        self._confirmed_compact = merged.replace(" ", "")

        if self.live_line is not None and self.live_line.text:
            self.last_refine_display_text = self._overlay(self.live_line.text)
        else:
            self.last_refine_display_text = self.confirmed_text
        return True

    # ------------------------------------------------------------------
    # Core: overlay confirmed onto live via SequenceMatcher
    # ------------------------------------------------------------------

    def _overlay(self, live_text: str) -> str:
        """Replace the portion of live_text covered by confirmed_text.

        Uses SequenceMatcher on compact (space-removed) text to find the
        rightmost position in live that confirmed covers, then appends
        the remaining live suffix onto confirmed.
        """
        if not self._confirmed_compact:
            return live_text

        live_compact = live_text.replace(" ", "")
        if not live_compact:
            return self.confirmed_text

        # Find where confirmed ends within live using matching blocks
        live_cut_compact = self._find_coverage_end(self._confirmed_compact, live_compact)

        if live_cut_compact == 0:
            # No meaningful match — show confirmed + full live as suffix
            return self._join(self.confirmed_text, live_text)

        # Convert compact index back to original live_text position
        live_cut = self._compact_index_to_original(live_text, live_cut_compact)
        suffix = live_text[live_cut:]
        return self._join(self.confirmed_text, suffix)

    @staticmethod
    def _find_coverage_end(confirmed_c: str, live_c: str) -> int:
        """Find rightmost position in live_compact covered by confirmed_compact.

        Uses SequenceMatcher matching blocks to find overlap between the two
        compact texts, and returns the end position in live_c.
        """
        sm = SequenceMatcher(None, confirmed_c, live_c, autojunk=False)
        if sm.ratio() < 0.3:
            return 0

        # Find the rightmost position in live_c covered by any matching block
        live_end = 0
        for _conf_start, live_start, length in sm.get_matching_blocks():
            if length > 0:
                live_end = max(live_end, live_start + length)
        return live_end

    @staticmethod
    def _compact_index_to_original(text: str, compact_count: int) -> int:
        """Convert compact char count back to position in the original (spaced) text."""
        seen = 0
        for i, ch in enumerate(text):
            if not ch.isspace():
                seen += 1
                if seen == compact_count:
                    return i + 1
        return len(text)

    @staticmethod
    def _join(confirmed: str, suffix: str) -> str:
        """Join confirmed text and live suffix, removing overlap and fixing spacing."""
        if not suffix:
            return confirmed
        suffix = suffix.lstrip()
        if not suffix:
            return confirmed

        # Remove suffix-prefix overlap
        max_k = min(len(confirmed), len(suffix))
        overlap = 0
        for k in range(max_k, 0, -1):
            if confirmed.endswith(suffix[:k]):
                overlap = k
                break
        if overlap > 0:
            suffix = suffix[overlap:]
            if not suffix:
                return confirmed

        _PUNCT = frozenset('.,!?;:…')
        if confirmed and suffix and not confirmed[-1].isspace() and not suffix[0].isspace() and suffix[0] not in _PUNCT:
            return confirmed + " " + suffix
        return confirmed + suffix

