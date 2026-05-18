#!/usr/bin/env python3
# Copyright    2024
# See ../../../../LICENSE
# Licensed under the Apache License, Version 2.0

import argparse
from pathlib import Path

from lhotse import CutSet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--cuts",
        type=Path,
        nargs="+",
        required=True,
        help="One or more cut manifests (jsonl.gz).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        required=True,
        help="Output transcript_words.txt path.",
    )
    return parser.parse_args()


def main():
    args = get_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w", encoding="utf-8") as f:
        for cuts_path in args.cuts:
            cuts = CutSet.from_file(cuts_path)
            for cut in cuts:
                for sup in cut.supervisions:
                    if sup.text:
                        f.write(sup.text)
                        f.write("\n")


if __name__ == "__main__":
    main()
