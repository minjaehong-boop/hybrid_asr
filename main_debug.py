"""Debug runner: shows step-by-step overlay process of REFINE onto LIVE.

Usage:
    python main_debug.py --audio data/utterances/utt_0001.wav --no-realtime
"""
from __future__ import annotations

import sys
from difflib import SequenceMatcher

from config.settings import build_config, parse_args
from asr.pipeline import StreamingSubtitlePipeline
from utils.subtitle_store import SubtitleLine, SubtitleStore
from utils.time_utils import format_ts


# ANSI colors
C_RESET  = "\033[0m"
C_CYAN   = "\033[36m"
C_YELLOW = "\033[33m"
C_GREEN  = "\033[32m"
C_RED    = "\033[31m"
C_MAGENTA= "\033[35m"
C_DIM    = "\033[2m"
C_BOLD   = "\033[1m"


class DebugSubtitleStore(SubtitleStore):
    """SubtitleStore with verbose debug output for overlay operations."""

    def __init__(self) -> None:
        super().__init__()
        self._refine_count = 0
        self._overlay_count = 0

    def apply_refine(self, line: SubtitleLine, now_sec: float, is_final: bool = False) -> bool:
        if not line.text.strip():
            return False

        self._refine_count += 1
        new_text = line.text.strip()
        tag = "REFINE-FINAL" if is_final else "REFINE"

        print(f"\n{'='*80}")
        print(f"{C_YELLOW}{C_BOLD}[{tag} #{self._refine_count}] {format_ts(line.start_sec)} ~ {format_ts(line.end_sec)}{C_RESET}")
        print(f"{'='*80}")

        print(f"\n{C_BOLD}1) 새 REFINE 텍스트:{C_RESET}")
        print(f"   \"{C_GREEN}{new_text}{C_RESET}\"")

        prev = self.confirmed_text
        if prev:
            print(f"\n{C_BOLD}2) 기존 confirmed_text:{C_RESET}")
            print(f"   \"{C_CYAN}{prev}{C_RESET}\"")
            merged = prev + " " + new_text
            print(f"\n{C_BOLD}3) 단순 연결 결과:{C_RESET}")
            print(f"   \"{C_GREEN}{merged}{C_RESET}\"")
        else:
            merged = new_text
            print(f"\n{C_BOLD}2) confirmed_text 없음 → 그대로 사용{C_RESET}")

        self.confirmed_text = merged
        self._confirmed_compact = merged.replace(" ", "")

        print(f"\n{C_BOLD}4) confirmed_text 갱신 완료:{C_RESET}")
        print(f"   원본:   \"{C_CYAN}{self.confirmed_text}{C_RESET}\"")
        print(f"   compact: \"{C_DIM}{self._confirmed_compact}{C_RESET}\"")

        # Overlay onto current LIVE
        if self.live_line is not None and self.live_line.text:
            print(f"\n{C_BOLD}5) 현재 LIVE에 오버레이 적용:{C_RESET}")
            self.last_refine_display_text = self._debug_overlay(self.live_line.text, context=f"{tag} #{self._refine_count}")
        else:
            self.last_refine_display_text = self.confirmed_text
            print(f"\n{C_DIM}5) 현재 LIVE 없음 → confirmed_text 그대로 표시{C_RESET}")

        print()
        return True

    def update_live(self, line: SubtitleLine) -> SubtitleLine:
        self.live_line = line

        if not self.confirmed_text:
            self.live_display_line = line
            return line

        self._overlay_count += 1
        show_debug = (self._overlay_count % 20 == 1)

        if show_debug:
            print(f"\n{C_DIM}--- LIVE overlay #{self._overlay_count} ---{C_RESET}")
            display_text = self._debug_overlay(line.text, context=f"LIVE #{self._overlay_count}", compact=True)
        else:
            display_text = self._overlay(line.text)

        self.live_display_line = SubtitleLine(
            start_sec=line.start_sec,
            end_sec=line.end_sec,
            text=display_text,
            source=line.source,
        )
        return self.live_display_line

    def _debug_overlay(self, live_text: str, context: str = "", compact: bool = False) -> str:
        """_overlay() with step-by-step debug output.

        Mirrors SubtitleStore._overlay / _find_coverage_end logic exactly.
        """
        indent = "   "

        if not self._confirmed_compact:
            print(f"{indent}{C_DIM}confirmed 비어있음 → LIVE 그대로{C_RESET}")
            return live_text

        live_compact = live_text.replace(" ", "")
        if not live_compact:
            print(f"{indent}{C_DIM}LIVE 비어있음 → confirmed 반환{C_RESET}")
            return self.confirmed_text

        confirmed_c = self._confirmed_compact

        # Step A: Show compact texts
        if not compact:
            print(f"\n{indent}{C_BOLD}[Step A] 공백 제거 (compact 변환){C_RESET}")
            print(f"{indent}confirmed_compact: \"{C_CYAN}{confirmed_c}{C_RESET}\"")
            print(f"{indent}live_compact:      \"{C_MAGENTA}{live_compact}{C_RESET}\"")

        # Step B: SequenceMatcher — mirrors SubtitleStore._find_coverage_end
        sm = SequenceMatcher(None, confirmed_c, live_compact, autojunk=False)
        ratio = sm.ratio()
        blocks = [(a, b, n) for a, b, n in sm.get_matching_blocks() if n > 0]

        if not compact:
            print(f"\n{indent}{C_BOLD}[Step B] SequenceMatcher 분석{C_RESET}")
            print(f"{indent}유사도 ratio: {ratio:.4f}  (임계값: 0.3)")

        if ratio < 0.3:
            if not compact:
                print(f"{indent}{C_RED}ratio < 0.3 → 매칭 실패! confirmed + LIVE 전체를 suffix로{C_RESET}")
            result = self._join(self.confirmed_text, live_text)
            if not compact:
                print(f"\n{indent}{C_BOLD}[결과]{C_RESET} \"{C_GREEN}{result}{C_RESET}\"")
            return result

        # Show matching blocks
        if not compact:
            print(f"\n{indent}{C_BOLD}[Step C] matching_blocks (겹치는 구간들){C_RESET}")
            for i, (conf_start, live_start, length) in enumerate(blocks):
                conf_frag = confirmed_c[conf_start:conf_start+length]
                print(f"{indent}  블록 {i+1}: confirmed[{conf_start}:{conf_start+length}] ↔ live[{live_start}:{live_start+length}]")
                print(f"{indent}         \"{C_GREEN}{conf_frag}{C_RESET}\" ({length}글자)")

        # Compute live_cut_compact — same as _find_coverage_end's live_end
        live_end = 0
        for _, live_start, length in blocks:
            live_end = max(live_end, live_start + length)
        live_cut_compact = live_end

        if not compact:
            print(f"\n{indent}{C_BOLD}[Step D] 커버리지 끝 위치 (live_cut_compact){C_RESET}")
            print(f"{indent}live_compact에서 confirmed가 커버하는 끝 = {C_BOLD}{live_cut_compact}{C_RESET}")
            print(f"{indent}live_compact: \"{live_compact}\"")
            if live_cut_compact <= len(live_compact):
                covered = live_compact[:live_cut_compact]
                remaining = live_compact[live_cut_compact:]
                print(f"{indent}              \"{C_CYAN}{covered}{C_RESET}{C_RED}{remaining}{C_RESET}\"")
                print(f"{indent}               {'─'*live_cut_compact}^ cut here (pos {live_cut_compact})")

        if live_cut_compact == 0:
            result = self._join(self.confirmed_text, live_text)
            if not compact:
                print(f"\n{indent}{C_RED}커버리지 0 → confirmed + LIVE 전체{C_RESET}")
                print(f"{indent}{C_BOLD}[결과]{C_RESET} \"{C_GREEN}{result}{C_RESET}\"")
            return result

        # Step E: Convert compact index to original
        live_cut = self._compact_index_to_original(live_text, live_cut_compact)
        suffix = live_text[live_cut:]

        if not compact:
            print(f"\n{indent}{C_BOLD}[Step E] compact → 원본 인덱스 변환{C_RESET}")
            print(f"{indent}live_text:     \"{live_text}\"")
            print(f"{indent}compact {live_cut_compact}글자 → 원본 위치 {live_cut}")
            if live_cut <= len(live_text):
                covered_orig = live_text[:live_cut]
                print(f"{indent}               \"{C_CYAN}{covered_orig}{C_RESET}{C_RED}{suffix}{C_RESET}\"")
                print(f"{indent}               {'─'*len(covered_orig)}^ cut (pos {live_cut})")

            print(f"\n{indent}{C_BOLD}[Step F] suffix 추출{C_RESET}")
            print(f"{indent}suffix (LIVE의 남은 부분): \"{C_RED}{suffix}{C_RESET}\"")

        # Step G: Join
        result = self._debug_join(self.confirmed_text, suffix, compact=compact, indent=indent)
        return result

    def _debug_join(self, confirmed: str, suffix: str, compact: bool = False, indent: str = "   ") -> str:
        """_join() with debug output — mirrors SubtitleStore._join exactly."""
        if not suffix:
            if not compact:
                print(f"\n{indent}{C_BOLD}[Step G] _join: suffix 없음 → confirmed 그대로{C_RESET}")
                print(f"{indent}{C_BOLD}[결과]{C_RESET} \"{C_GREEN}{confirmed}{C_RESET}\"")
            return confirmed

        suffix_stripped = suffix.lstrip()
        if not suffix_stripped:
            if not compact:
                print(f"\n{indent}{C_BOLD}[Step G] _join: suffix 공백만 → confirmed 그대로{C_RESET}")
                print(f"{indent}{C_BOLD}[결과]{C_RESET} \"{C_GREEN}{confirmed}{C_RESET}\"")
            return confirmed

        # Remove suffix-prefix overlap
        max_k = min(len(confirmed), len(suffix_stripped))
        overlap = 0
        for k in range(max_k, 0, -1):
            if confirmed.endswith(suffix_stripped[:k]):
                overlap = k
                break

        if not compact:
            print(f"\n{indent}{C_BOLD}[Step G] _join 결합{C_RESET}")
            print(f"{indent}confirmed 끝: \"...{confirmed[-20:]}\"")
            print(f"{indent}suffix:        \"{suffix_stripped}\"")

        if overlap > 0:
            overlap_text = suffix_stripped[:overlap]
            suffix_stripped = suffix_stripped[overlap:]
            if not compact:
                print(f"{indent}{C_YELLOW}suffix-prefix 중복 발견: \"{overlap_text}\" ({overlap}글자) → 제거{C_RESET}")

            if not suffix_stripped:
                if not compact:
                    print(f"{indent}{C_BOLD}[결과]{C_RESET} \"{C_GREEN}{confirmed}{C_RESET}\"")
                return confirmed

        _PUNCT = frozenset('.,!?;:…')
        if (confirmed and suffix_stripped
                and not confirmed[-1].isspace()
                and not suffix_stripped[0].isspace()
                and suffix_stripped[0] not in _PUNCT):
            result = confirmed + " " + suffix_stripped
            if not compact:
                print(f"{indent}공백 삽입하여 결합")
        else:
            result = confirmed + suffix_stripped
            if not compact and suffix_stripped and suffix_stripped[0] in _PUNCT:
                print(f"{indent}구두점 앞 공백 없이 결합")

        if not compact:
            print(f"\n{indent}{C_BOLD}[최종 결과]{C_RESET}")
            print(f"{indent}\"{C_GREEN}{result}{C_RESET}\"")

        return result


class DebugPipeline(StreamingSubtitlePipeline):
    """Pipeline with debug subtitle store."""

    def __init__(self, cfg) -> None:
        super().__init__(cfg)
        self.store = DebugSubtitleStore()


def main() -> int:
    args = parse_args()
    cfg = build_config(args)
    pipeline = DebugPipeline(cfg)
    pipeline.run()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
