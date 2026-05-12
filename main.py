from __future__ import annotations

import sys

from config.settings import build_config, parse_args
from asr.pipeline import StreamingSubtitlePipeline


def main() -> int:
    """Parse config and run the streaming subtitle pipeline."""
    args = parse_args()
    cfg = build_config(args)
    pipeline = StreamingSubtitlePipeline(cfg)
    pipeline.run()
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(1)
