#!/usr/bin/env python3
# Copyright    2024
# See ../../../../LICENSE
# Licensed under the Apache License, Version 2.0

"""Prepare Zeroth-Korean manifests for Lhotse.

Expected layout:
  zeroth_korean/
    train_data_01/
      .../*.flac and *.trans.txt
    test_data_01/
      .../*.flac and *.trans.txt

Each *.trans.txt line format:
  <utt_id> <text>
"""

import argparse
import logging
from pathlib import Path
from typing import Dict, Iterable, Tuple

from lhotse import Recording, RecordingSet, SupervisionSegment, SupervisionSet


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--zeroth-dir",
        type=Path,
        required=True,
        help="Path to zeroth_korean root dir.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        required=True,
        help="Output directory for manifests.",
    )
    return parser.parse_args()


def iter_transcripts(trans_path: Path) -> Iterable[Tuple[str, str]]:
    with trans_path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split(maxsplit=1)
            if len(parts) != 2:
                continue
            yield parts[0], parts[1]


def prepare_split(split_dir: Path) -> Tuple[RecordingSet, SupervisionSet]:
    recordings = []
    supervisions = []

    trans_files = list(split_dir.rglob("*.trans.txt"))
    for trans_path in trans_files:
        for utt_id, text in iter_transcripts(trans_path):
            audio_path = trans_path.parent / f"{utt_id}.flac"
            if not audio_path.is_file():
                wav_path = trans_path.parent / f"{utt_id}.wav"
                if wav_path.is_file():
                    audio_path = wav_path
                else:
                    logging.warning("Missing audio for %s", utt_id)
                    continue

            recording = Recording.from_file(audio_path, recording_id=utt_id)
            supervision = SupervisionSegment(
                id=utt_id,
                recording_id=utt_id,
                start=0.0,
                duration=recording.duration,
                text=text,
                language="kor",
            )

            recordings.append(recording)
            supervisions.append(supervision)

    return RecordingSet.from_recordings(recordings), SupervisionSet.from_segments(
        supervisions
    )


def main():
    args = get_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    for split_name in ("train", "test"):
        split_dir = args.zeroth_dir / f"{split_name}_data_01"
        if not split_dir.is_dir():
            logging.warning("Missing split dir: %s", split_dir)
            continue

        logging.info("Preparing %s", split_dir)
        recordings, supervisions = prepare_split(split_dir)

        recordings_path = args.output_dir / f"zeroth_recordings_{split_name}.jsonl.gz"
        supervisions_path = (
            args.output_dir / f"zeroth_supervisions_{split_name}.jsonl.gz"
        )

        recordings.to_file(recordings_path)
        supervisions.to_file(supervisions_path)
        logging.info("Wrote %s and %s", recordings_path, supervisions_path)


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    main()
