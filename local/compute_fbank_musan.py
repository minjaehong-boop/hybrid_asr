#!/usr/bin/env python3
# Copyright    2021  Xiaomi Corp.        (authors: Fangjun Kuang)
#
# See ../../../../LICENSE for clarification regarding multiple authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
Computes fbank features for the MUSAN dataset.
Looks for manifests in data/manifests. Saves features to data/fbank.

Uses CUDA batch extraction (same approach as compute_fbank_ksponspeech.py)
to avoid multiprocessing deadlocks on CUDA systems.
"""
import argparse
import logging
from pathlib import Path

import torch
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter, MonoCut, combine
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import str2bool

torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def is_cut_long(c: MonoCut) -> bool:
    return c.duration > 5


def compute_fbank_musan(
    num_mel_bins: int = 80,
    output_dir: str = "data/fbank",
):
    src_dir = Path("data/manifests")
    output_dir = Path(output_dir)

    dataset_parts = ("music", "speech", "noise")
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix="musan",
        suffix="jsonl.gz",
    )
    assert manifests is not None
    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        list(manifests.keys()),
    )

    musan_cuts_path = output_dir / "musan_cuts.jsonl.gz"
    if musan_cuts_path.is_file():
        logging.info(f"{musan_cuts_path} already exists - skipping")
        return

    logging.info("Extracting features for MUSAN")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    logging.info(f"Device: {device}")
    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins, device=device))

    cut_set = (
        CutSet.from_manifests(
            recordings=combine(part["recordings"] for part in manifests.values())
        )
        .cut_into_windows(10.0)
        .filter(is_cut_long)
    )

    if device == "cuda":
        cut_set = cut_set.compute_and_store_features_batch(
            extractor=extractor,
            storage_path=f"{output_dir}/musan_feats",
            num_workers=4,
            storage_type=LilcomChunkyWriter,
        )
    else:
        cut_set = cut_set.compute_and_store_features(
            extractor=extractor,
            storage_path=f"{output_dir}/musan_feats",
            num_jobs=1,
            storage_type=LilcomChunkyWriter,
        )

    cut_set.to_file(musan_cuts_path)


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-mel-bins", type=int, default=80)
    parser.add_argument("--output-dir", type=str, default="data/fbank")
    return parser.parse_args()


if __name__ == "__main__":
    logging.basicConfig(
        format="%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s",
        level=logging.INFO,
    )
    args = get_args()
    compute_fbank_musan(
        num_mel_bins=args.num_mel_bins,
        output_dir=args.output_dir,
    )
