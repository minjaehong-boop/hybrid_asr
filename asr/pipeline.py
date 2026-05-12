from __future__ import annotations

import multiprocessing as mp
import queue
import time

import numpy as np

from config.settings import PipelineConfig
from asr.transducer_stage import TransducerUpdate
from asr.refine_stage import RefineUpdate
from asr.protocol import RefineEvent, RefineTask, TransducerEvent, TransducerTask
from asr.workers import run_refine_worker, run_transducer_worker
from utils.audio_source import MicChunkSource, WavChunkSource
from utils.subtitle_store import SubtitleLine, SubtitleStore
from utils.time_utils import format_ts

from quickspacer import Spacer

class StreamingSubtitlePipeline:
    """Coordinate audio streaming and subtitle merge while ASR runs in parallel workers."""

    def __init__(self, cfg: PipelineConfig) -> None:
        """Initialize audio source, subtitle store, and worker runtime state."""
        self.cfg = cfg
        self.store = SubtitleStore()
        self.spacer = Spacer(level=3)
        if cfg.use_mic:
            self.source = MicChunkSource(
                frame_sec=cfg.frame_sec,
                target_sr=cfg.transducer.sample_rate,
                max_seconds=cfg.max_seconds,
                vad_model=cfg.vad_model,
                silence_timeout=cfg.silence_timeout,
            )
        else:
            self.source = WavChunkSource(
                path=cfg.audio_path,
                frame_sec=cfg.frame_sec,
                target_sr=cfg.transducer.sample_rate,
                realtime=cfg.realtime,
                max_seconds=cfg.max_seconds,
            )
        sr = cfg.transducer.sample_rate
        self.refine_chunk_samples = int(cfg.refine_chunk_sec * sr)
        self.refine_buffer: list[np.ndarray] = []
        self.refine_buffer_samples = 0
        self.transducer_task_q = None
        self.transducer_event_q = None
        self.refine_task_q = None
        self.refine_event_q = None
        self.transducer_proc: mp.Process | None = None
        self.refine_proc: mp.Process | None = None
        self.transducer_done = False
        self.refine_done = False

    def _emit_live(self, update: TransducerUpdate) -> None:
        #print("_emit_live")
        #print("update: ", update)
        """Apply live text replacement and print live subtitle output."""

        display_line = self.store.update_live(
            SubtitleLine(
                start_sec=update.start_sec,
                end_sec=update.end_sec,
                text=update.text,
                source="live",
            )
        )
        #print("start_sec: ",update.start_sec)
        #print("end_sec: ",update.end_sec)
        display_text = display_line.text
        if display_text.strip():
            display_text = self.spacer.space([display_text.replace(" ", "")])[0]
        tag = "LIVE-FINAL" if update.is_final else "LIVE"
        print(
            f"[{tag}] {format_ts(display_line.start_sec)} ~ {format_ts(display_line.end_sec)} | {display_text}",
            flush=True,
        )

    def _emit_refine(self, update: RefineUpdate) -> None:
        """Apply spacer to refine text, then store and print."""
        # Fix spacing before entering store
        now_sec = update.end_sec
        if self.store.live_line is not None:
            now_sec = max(now_sec, self.store.live_line.end_sec)

        refine_text = update.text
        if refine_text.strip():
            refine_text = self.spacer.space([refine_text.replace(" ", "")])[0]
        applied = self.store.apply_refine(
            SubtitleLine(
                start_sec=update.start_sec,
                end_sec=update.end_sec,
                text=refine_text,
                source="refine",
            ),
            now_sec=now_sec,
            is_final=update.is_final,
        )
        if not applied and not update.is_final:
            return
        tag = "REFINE-FINAL" if update.is_final else "REFINE"
        display_text = self.store.last_refine_display_text or update.text
        if display_text.strip():
            display_text = self.spacer.space([display_text.replace(" ", "")])[0]
        if display_text:
            print(
                f"[{tag}] {format_ts(update.start_sec)} ~ {format_ts(update.end_sec)} | {display_text}",
                flush=True,
            )

    def _drain_transducer_events(self, max_items: int | None = None) -> None:
        #print("_drain_transducer_events")
        """Consume transducer events, optionally limiting batch size for fairness."""
        if self.transducer_event_q is None:
            return
        count = 0
        while True:
            try:
                event: TransducerEvent = self.transducer_event_q.get_nowait()
            except queue.Empty:
                return
            if event.kind == "update" and event.update is not None:
                self._emit_live(event.update)
                #print(self._emit_live)
            elif event.kind == "done":
                self.transducer_done = True
                #print("done")
            elif event.kind == "error":
                raise RuntimeError(f"Transducer worker failed:\n{event.error}")
            count += 1
            if max_items is not None and count >= max_items:
                return

    def _drain_refine_events(self, max_items: int | None = None) -> None:
        #print("_drain_refine_events")
        """Consume refine events, optionally limiting batch size for fairness."""
        if self.refine_event_q is None:
            return
        count = 0
        while True:
            try:
                event: RefineEvent = self.refine_event_q.get_nowait()
            except queue.Empty:
                return
            if event.kind == "update" and event.update is not None:
                self._emit_refine(event.update)
            elif event.kind == "done":
                self.refine_done = True
            elif event.kind == "error":
                raise RuntimeError(f"Refine worker failed:\n{event.error}")
            count += 1
            if max_items is not None and count >= max_items:
                return

    def _drain_worker_events(self) -> None:
        #print("_drain_worker_events")
        """Drain both worker event queues."""
        self._drain_refine_events(max_items=8)
        self._drain_transducer_events(max_items=16)
        self._drain_refine_events(max_items=8)

    def _check_worker_health(self) -> None:
        #print("_check_worker_health")
        """Raise an error if a worker exits unexpectedly before done event."""
        if self.transducer_proc is not None and (not self.transducer_done):
            if not self.transducer_proc.is_alive():
                if self.transducer_proc.exitcode == 0:
                    self.transducer_done = True
                else:
                    raise RuntimeError(f"Transducer worker exited early (code={self.transducer_proc.exitcode}).")
        if self.refine_proc is not None and (not self.refine_done):
            if not self.refine_proc.is_alive():
                if self.refine_proc.exitcode == 0:
                    self.refine_done = True
                else:
                    raise RuntimeError(f"Refine worker exited early (code={self.refine_proc.exitcode}).")

    def _put_task(self, task_queue, task) -> None:
        #print("_put_task")
        """Put a task with backpressure handling while draining output queues."""
        while True:
            try:
                task_queue.put(task, timeout=0.05)
                return
            except queue.Full:
                self._drain_worker_events()
                self._check_worker_health()

    def _flush_refine_buffer(self) -> None:
        #print("_flush_refine_buffer")
        """Send accumulated refine chunk to the SenseVoice worker."""
        if self.refine_buffer_samples == 0:
            return
        chunk = np.concatenate(self.refine_buffer)
        self.refine_buffer.clear()
        self.refine_buffer_samples = 0
        if self.refine_task_q is None:
            return
        self._put_task(
            self.refine_task_q,
            RefineTask(action="feed", samples=chunk),
        )

    def _start_workers(self) -> None:
        #print("_start_workers")
        """Create queues and start transducer/refine worker processes."""
        ctx = mp.get_context("spawn")
        self.transducer_task_q = ctx.Queue(maxsize=32)
        self.transducer_event_q = ctx.Queue()
        self.refine_task_q = ctx.Queue(maxsize=1)
        self.refine_event_q = ctx.Queue()

        self.transducer_proc = ctx.Process(
            target=run_transducer_worker,
            args=(self.cfg.transducer, self.transducer_task_q, self.transducer_event_q),
            name="transducer-worker",
        )
        self.refine_proc = ctx.Process(
            target=run_refine_worker,
            args=(self.cfg.refine, self.refine_task_q, self.refine_event_q),
            name="refine-worker",
        )
        self.transducer_proc.start()
        self.refine_proc.start()
        self.transducer_done = False
        self.refine_done = False

    def _stop_workers(self) -> None:
        """Join workers and force terminate only if still alive."""
        for proc in (self.transducer_proc, self.refine_proc):
            if proc is None:
                continue
            proc.join(timeout=0.2)
            if proc.is_alive():
                proc.terminate()
                proc.join(timeout=1.0)

    def _finalize_workers(self) -> None:
        #print("_finalize_workers")
        """Request finalization from both workers."""
        if self.transducer_task_q is not None:
            self._put_task(
                self.transducer_task_q,
                TransducerTask(action="finalize", end_sec=self.source.duration_sec),
            )
        if self.refine_task_q is not None:
            self._put_task(
                self.refine_task_q,
                RefineTask(action="finalize"),
            )

    def _drain_until_done(self) -> None:
        #print("_drain_until_done")
        """Block until both workers report done while continuously draining events."""
        while not (self.transducer_done and self.refine_done):
            self._drain_worker_events()
            self._check_worker_health()
            time.sleep(0.005)
        self._drain_worker_events()

    def run(self) -> None:
        #print("run")
        """Run the streaming pipeline with transducer/refine in parallel processes."""
        source_label = "mic" if self.cfg.use_mic else str(self.cfg.audio_path)
        print(
            f"Start pipeline: source={source_label} frame={self.cfg.frame_sec}s refine={self.cfg.refine_chunk_sec}s",
            flush=True,
        )
        self._start_workers()
        try:
            for frame in self.source.iter_frames():
                if self.transducer_task_q is not None:
                    self._put_task(
                        self.transducer_task_q,
                        TransducerTask(action="feed", end_sec=frame.end_sec, samples=frame.samples),
                    )

                self.refine_buffer.append(frame.samples)
                self.refine_buffer_samples += len(frame.samples)
                if self.refine_buffer_samples >= self.refine_chunk_samples:
                    self._flush_refine_buffer()

                self._drain_worker_events()
                self._check_worker_health()

            self._flush_refine_buffer()
            self._finalize_workers()
            self._drain_until_done()
        finally:
            self._stop_workers()
