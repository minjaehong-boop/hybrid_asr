#!/usr/bin/env python3
# Copyright    2024
# See ../../../../LICENSE
# Licensed under the Apache License, Version 2.0

import argparse
import logging
import os
from pathlib import Path
from typing import Optional

import sentencepiece as spm
import torch
from filter_cuts import filter_cuts
from lhotse import CutSet, Fbank, FbankConfig, LilcomChunkyWriter
from lhotse.recipes.utils import read_manifests_if_cached

from icefall.utils import get_executor, str2bool

# Torch's multithreaded behavior needs to be disabled or
# it wastes a lot of CPU and slow things down.
torch.set_num_threads(1)
torch.set_num_interop_threads(1)


def get_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--bpe-model",
        type=str,
        help="""Path to the bpe.model. If not None, we will remove short and
        long utterances before extracting features""",
    )

    parser.add_argument(
        "--dataset",
        type=str,
        help="""Dataset parts to compute fbank. If None, we will use all""",
    )

    parser.add_argument(
        "--perturb-speed",
        type=str2bool,
        default=True,
        help="""Perturb speed with factor 0.9 and 1.1 on train subset.""",
    )
    parser.add_argument(
        "--data-dir",
        type=str,
        default="data",
        help="""Path of data directory""",
    )

    return parser.parse_args()


def compute_fbank_zeroth(
    bpe_model: Optional[str] = None,
    dataset: Optional[str] = None,
    perturb_speed: Optional[bool] = False,
    data_dir: Optional[str] = "data",
):
    src_dir = Path(data_dir) / "manifests"
    output_dir = Path(data_dir) / "fbank"
    num_jobs = min(4, os.cpu_count())
    num_mel_bins = 80

    if bpe_model:
        logging.info("Loading %s", bpe_model)
        sp = spm.SentencePieceProcessor()
        sp.load(bpe_model)

    if dataset is None:
        dataset_parts = (
            "train",
            "test",
        )
    else:
        dataset_parts = dataset.split(" ", -1)

    prefix = "zeroth"
    suffix = "jsonl.gz"
    logging.info("Read manifests...")
    manifests = read_manifests_if_cached(
        dataset_parts=dataset_parts,
        output_dir=src_dir,
        prefix=prefix,
        suffix=suffix,
    )
    assert manifests is not None

    assert len(manifests) == len(dataset_parts), (
        len(manifests),
        len(dataset_parts),
        list(manifests.keys()),
        dataset_parts,
    )

    if torch.cuda.is_available():
        device = "cuda"
    else:
        device = "cpu"
    logging.info("Device: %s", device)

    extractor = Fbank(FbankConfig(num_mel_bins=num_mel_bins, device=device))

    with get_executor() as ex:
        logging.info("Executor: %s", ex)
        for partition, m in manifests.items():
            cuts_filename = f"{prefix}_cuts_{partition}.{suffix}"
            if (output_dir / cuts_filename).is_file():
                logging.info("%s already exists - skipping.", partition)
                continue
            logging.info("Processing %s", partition)
            cut_set = CutSet.from_manifests(
                recordings=m["recordings"],
                supervisions=m["supervisions"],
            )

            cut_set = cut_set.filter(
                lambda x: x.duration > 1 and x.sampling_rate == 16000
            )

            if "train" in partition:
                if bpe_model:
                    cut_set = filter_cuts(cut_set, sp)
                if perturb_speed:
                    logging.info("Doing speed perturb")
                    cut_set = (
                        cut_set
                        + cut_set.perturb_speed(0.9)
                        + cut_set.perturb_speed(1.1)
                    )

            logging.info("Compute & Store features...")
            if device == "cuda":
                cut_set = cut_set.compute_and_store_features_batch(
                    extractor=extractor,
                    storage_path=f"{output_dir}/{prefix}_feats_{partition}",
                    num_workers=4,
                    storage_type=LilcomChunkyWriter,
                )
            else:
                cut_set = cut_set.compute_and_store_features(
                    extractor=extractor,
                    storage_path=f"{output_dir}/{prefix}_feats_{partition}",
                    num_jobs=num_jobs if ex is None else 80,
                    executor=ex,
                    storage_type=LilcomChunkyWriter,
                )

            cut_set.to_file(output_dir / cuts_filename)


if __name__ == "__main__":
    formatter = "%(asctime)s %(levelname)s [%(filename)s:%(lineno)d] %(message)s"

    logging.basicConfig(format=formatter, level=logging.INFO)
    args = get_args()
    logging.info(vars(args))
    compute_fbank_zeroth(
        bpe_model=args.bpe_model,
        dataset=args.dataset,
        perturb_speed=args.perturb_speed,
        data_dir=args.data_dir,
    )
