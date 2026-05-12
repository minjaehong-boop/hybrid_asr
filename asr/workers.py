from __future__ import annotations

import traceback

from config.settings import SenseVoiceConfig, TransducerConfig
from asr.transducer_stage import TransducerStage
from asr.refine_stage import SenseVoiceRefineStage
from asr.protocol import RefineEvent, RefineTask, TransducerEvent, TransducerTask


def run_transducer_worker(cfg: TransducerConfig, task_queue, event_queue) -> None:
    print("run_transducer_worker")
    """Run the sherpa-onnx transducer loop in a dedicated process."""
    try:
        stage = TransducerStage(cfg)
        while True:
            task: TransducerTask = task_queue.get()
            if task.action == "feed":
                if task.samples is None:
                    continue
                update = stage.feed(task.samples, task.end_sec)
                if update is not None:
                    event_queue.put(TransducerEvent(kind="update", update=update))
                continue

            if task.action == "finalize":
                update = stage.finalize(task.end_sec)
                if update is not None:
                    event_queue.put(TransducerEvent(kind="update", update=update))
                event_queue.put(TransducerEvent(kind="done"))
                return

            raise ValueError(f"Unknown transducer action: {task.action}")
    except Exception:
        event_queue.put(TransducerEvent(kind="error", error=traceback.format_exc()))


def run_refine_worker(cfg: SenseVoiceConfig, task_queue, event_queue) -> None:
    print("run_refine_worker")
    """Run the SenseVoice chunk refine loop in a dedicated process."""
    try:
        stage = SenseVoiceRefineStage(cfg)
        while True:
            task: RefineTask = task_queue.get()
            if task.action == "feed":
                if task.samples is None:
                    continue
                update = stage.feed(task.samples)
                if update is not None:
                    event_queue.put(RefineEvent(kind="update", update=update))
                continue

            if task.action == "finalize":
                update = stage.finalize()
                if update is not None:
                    event_queue.put(RefineEvent(kind="update", update=update))
                event_queue.put(RefineEvent(kind="done"))
                return

            raise ValueError(f"Unknown refine action: {task.action}")
    except Exception:
        event_queue.put(RefineEvent(kind="error", error=traceback.format_exc()))
