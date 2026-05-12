"""Web UI server for hybrid1 real-time Korean ASR subtitle pipeline.

Provides a modern web interface with:
- WAV file upload or sample file selection
- Real-time subtitle display via WebSocket (LIVE + REFINE)
- Pipeline status and audio timeline visualization
"""

from __future__ import annotations

import asyncio
import json
import multiprocessing as mp
import queue
import tempfile
import threading
import time
import uuid
from pathlib import Path

import numpy as np
import uvicorn
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse

from config.settings import PipelineConfig, SenseVoiceConfig, TransducerConfig, build_config
from asr.protocol import RefineEvent, RefineTask, TransducerEvent, TransducerTask
from asr.transducer_stage import TransducerUpdate
from asr.refine_stage import RefineUpdate
from utils.audio_source import MicChunkSource, WavChunkSource
from utils.subtitle_store import SubtitleLine, SubtitleStore
from utils.time_utils import format_ts

from quickspacer import Spacer


@asynccontextmanager
async def lifespan(_app: FastAPI):
    loop = asyncio.get_event_loop()
    await loop.run_in_executor(None, pool.start)
    yield


app = FastAPI(title="Hybrid1 ASR", lifespan=lifespan)

# Active sessions: session_id -> thread-safe queue
sessions: dict[str, queue.Queue] = {}


# ------------------------------------------------------------------
# Persistent worker functions (model loaded once, reset between runs)
# ------------------------------------------------------------------

def _persistent_transducer_worker(
    cfg: TransducerConfig, task_queue, event_queue,
) -> None:
    """Transducer worker that loads the model once and resets between runs."""
    from asr.transducer_stage import TransducerStage
    import traceback
    try:
        stage = TransducerStage(cfg)
        event_queue.put(TransducerEvent(kind="done"))  # signal: model ready

        while True:
            task: TransducerTask = task_queue.get()
            if task.action == "feed":
                if task.samples is None:
                    continue
                update = stage.feed(task.samples, task.end_sec)
                if update is not None:
                    event_queue.put(TransducerEvent(kind="update", update=update))
            elif task.action == "finalize":
                update = stage.finalize(task.end_sec)
                if update is not None:
                    event_queue.put(TransducerEvent(kind="update", update=update))
                event_queue.put(TransducerEvent(kind="done"))
                # Reset state for next run (model stays loaded)
                stage.stream = stage.recognizer.create_stream()
                stage.prev_text = ""
                stage.prev_tokens = []
    except Exception:
        event_queue.put(TransducerEvent(kind="error", error=traceback.format_exc()))


def _persistent_refine_worker(
    cfg: SenseVoiceConfig, task_queue, event_queue,
) -> None:
    """SenseVoice refine worker that loads the model once and resets between runs."""
    from asr.refine_stage import SenseVoiceRefineStage
    import traceback
    try:
        stage = SenseVoiceRefineStage(cfg)
        event_queue.put(RefineEvent(kind="done"))  # signal: model ready

        while True:
            task: RefineTask = task_queue.get()
            if task.action == "configure":
                if task.context_mode is not None:
                    stage.context_mode = task.context_mode
            elif task.action == "feed":
                if task.samples is None:
                    continue
                update = stage.feed(task.samples)
                if update is not None:
                    event_queue.put(RefineEvent(kind="update", update=update))
            elif task.action == "finalize":
                update = stage.finalize()
                if update is not None:
                    event_queue.put(RefineEvent(kind="update", update=update))
                event_queue.put(RefineEvent(kind="done"))
                # Reset state for next run (model stays loaded)
                stage.chunk_offset_sec = 0.0
                stage._chunk_index = 0
                stage._chunk_buffers = []
                stage._context_buffer = None
                stage._prev_committed_text = ""
                stage._last_result_tokens = []
                stage._last_result_timestamps = []
                stage._last_result_text = ""
                stage._last_chunk_dur = 0.0
                stage._last_right_trim = 0.0
                stage._overlap_buffer = None
                stage._last_raw_result = None
    except Exception:
        event_queue.put(RefineEvent(kind="error", error=traceback.format_exc()))


# ------------------------------------------------------------------
# Worker pool (started once at server boot)
# ------------------------------------------------------------------

class WorkerPool:
    """Pre-loads ASR models in persistent worker processes."""

    def __init__(self) -> None:
        self.cfg: PipelineConfig | None = None
        self.transducer_task_q = None
        self.transducer_event_q = None
        self.refine_task_q = None
        self.refine_event_q = None
        self.lock = threading.Lock()  # one pipeline at a time
        self.ready = False

    def start(self) -> None:
        """Build config, spawn workers, wait for model load."""
        import argparse
        args = argparse.Namespace(
            audio=Path("data/utterances/utt_0001.wav"),
            mic=False, config=None, no_realtime=False, max_seconds=None,
        )
        self.cfg = build_config(args)

        ctx = mp.get_context("spawn")
        self.transducer_task_q = ctx.Queue(maxsize=32)
        self.transducer_event_q = ctx.Queue()
        self.refine_task_q = ctx.Queue(maxsize=1)
        self.refine_event_q = ctx.Queue()

        tp = ctx.Process(
            target=_persistent_transducer_worker,
            args=(self.cfg.transducer, self.transducer_task_q, self.transducer_event_q),
            name="transducer-worker", daemon=True,
        )
        rp = ctx.Process(
            target=_persistent_refine_worker,
            args=(self.cfg.refine, self.refine_task_q, self.refine_event_q),
            name="refine-worker", daemon=True,
        )

        print("[WorkerPool] Starting workers, loading models...", flush=True)
        tp.start()
        rp.start()

        # Wait for "done" (= model ready) from each worker
        self.transducer_event_q.get(timeout=120)
        print("[WorkerPool] Transducer model loaded", flush=True)
        self.refine_event_q.get(timeout=120)
        print("[WorkerPool] Refine model loaded", flush=True)

        self.ready = True
        print("[WorkerPool] Ready", flush=True)


pool = WorkerPool()


# ------------------------------------------------------------------
# Config builder (reuses existing YAML)
# ------------------------------------------------------------------

def _build_config_for_audio(audio_path: Path, realtime: bool = True) -> PipelineConfig:
    """Build pipeline config for a given audio file."""
    import argparse
    args = argparse.Namespace(
        audio=audio_path,
        mic=False,
        config=None,
        no_realtime=not realtime,
        max_seconds=None,
    )
    return build_config(args)


# ------------------------------------------------------------------
# Pipeline runner (uses pre-loaded worker pool)
# ------------------------------------------------------------------

def _run_pipeline_thread(
    audio_path: Path | None,
    session_id: str,
    out_queue: queue.Queue,
    use_mic: bool = False,
    stop_event: threading.Event | None = None,
    context_mode: str = "accumulate",
) -> None:
    """Run the ASR pipeline using the persistent worker pool."""

    def send(msg: dict):
        out_queue.put_nowait(msg)

    if not pool.ready:
        send({"type": "error", "message": "Models not loaded yet"})
        send({"type": "status", "status": "error"})
        return

    if not pool.lock.acquire(blocking=False):
        send({"type": "error", "message": "Another pipeline is already running"})
        send({"type": "status", "status": "error"})
        return

    try:
        cfg = pool.cfg
        if use_mic:
            source = MicChunkSource(
                frame_sec=cfg.frame_sec,
                target_sr=cfg.transducer.sample_rate,
                max_seconds=cfg.max_seconds,
                vad_model=cfg.vad_model,
                silence_timeout=cfg.silence_timeout,
            )
        else:
            cfg = _build_config_for_audio(audio_path, realtime=True)
            source = WavChunkSource(
                path=cfg.audio_path,
                frame_sec=cfg.frame_sec,
                target_sr=cfg.transducer.sample_rate,
                realtime=cfg.realtime,
                max_seconds=cfg.max_seconds,
            )

        store = SubtitleStore()
        spacer = Spacer(level=3)

        send({
            "type": "status",
            "status": "running",
            "duration": source.duration_sec,
        })

        transducer_task_q = pool.transducer_task_q
        transducer_event_q = pool.transducer_event_q
        refine_task_q = pool.refine_task_q
        refine_event_q = pool.refine_event_q

        # Apply context mode before this run starts
        refine_task_q.put(RefineTask(action="configure", context_mode=context_mode))

        sr = cfg.transducer.sample_rate
        refine_chunk_samples = int(cfg.refine_chunk_sec * sr)
        print(f"Refine chunk: {cfg.refine_chunk_sec} seconds = {refine_chunk_samples} samples at {sr} Hz")
        refine_buffer: list[np.ndarray] = []
        refine_buffer_samples = 0
        transducer_done = False
        refine_done = False

        def drain_events():
            nonlocal transducer_done, refine_done

            for _ in range(16):
                try:
                    event: TransducerEvent = transducer_event_q.get_nowait()
                except queue.Empty:
                    break
                if event.kind == "update" and event.update is not None:
                    u = event.update
                    display_line = store.update_live(
                        SubtitleLine(start_sec=u.start_sec, end_sec=u.end_sec, text=u.text, source="live")
                    )
                    dt = display_line.text
                    if dt.strip():
                        dt = spacer.space([dt.replace(" ", "")])[0]
                    tag = "LIVE-FINAL" if u.is_final else "LIVE"
                    send({
                        "type": "subtitle",
                        "tag": tag,
                        "start": display_line.start_sec,
                        "end": display_line.end_sec,
                        "text": dt,
                        "confirmed": store.confirmed_text,
                    })
                elif event.kind == "done":
                    transducer_done = True
                elif event.kind == "error":
                    send({"type": "error", "message": event.error})

            for _ in range(8):
                try:
                    event: RefineEvent = refine_event_q.get_nowait()
                except queue.Empty:
                    break
                if event.kind == "update" and event.update is not None:
                    u = event.update
                    spaced = u.text
                    if spaced.strip():
                        spaced = spacer.space([spaced.replace(" ", "")])[0]

                    now_sec = u.end_sec
                    if store.live_line is not None:
                        now_sec = max(now_sec, store.live_line.end_sec)

                    applied = store.apply_refine(
                        SubtitleLine(start_sec=u.start_sec, end_sec=u.end_sec, text=spaced, source="refine"),
                        now_sec=now_sec,
                        is_final=u.is_final,
                    )
                    if applied or u.is_final:
                        tag = "REFINE-FINAL" if u.is_final else "REFINE"
                        display_text = store.last_refine_display_text or spaced
                        if display_text and display_text.strip():
                            display_text = spacer.space([display_text.replace(" ", "")])[0]
                        if display_text:
                            send({
                                "type": "subtitle",
                                "tag": tag,
                                "start": u.start_sec,
                                "end": u.end_sec,
                                "text": display_text,
                                "confirmed": store.confirmed_text,
                            })
                elif event.kind == "done":
                    refine_done = True
                elif event.kind == "error":
                    send({"type": "error", "message": event.error})

        def put_task(task_queue, task):
            while True:
                try:
                    task_queue.put(task, timeout=0.05)
                    return
                except queue.Full:
                    drain_events()

        def flush_refine():
            nonlocal refine_buffer_samples
            if refine_buffer_samples == 0:
                return
            chunk = np.concatenate(refine_buffer)
            refine_buffer.clear()
            refine_buffer_samples = 0
            put_task(refine_task_q, RefineTask(action="feed", samples=chunk))

        # Main loop
        for frame in source.iter_frames():
            if stop_event is not None and stop_event.is_set():
                if hasattr(source, '_stopped'):
                    source._stopped = True
                break
            put_task(
                transducer_task_q,
                TransducerTask(action="feed", end_sec=frame.end_sec, samples=frame.samples),
            )

            refine_buffer.append(frame.samples)
            refine_buffer_samples += len(frame.samples)
            if refine_buffer_samples >= refine_chunk_samples:
                flush_refine()

            drain_events()

            send({
                "type": "progress",
                "current": frame.end_sec,
                "duration": source.duration_sec,
            })

        # Finalize
        flush_refine()
        put_task(transducer_task_q, TransducerTask(action="finalize", end_sec=source.duration_sec))
        put_task(refine_task_q, RefineTask(action="finalize"))

        while not (transducer_done and refine_done):
            drain_events()
            time.sleep(0.01)
        drain_events()

        final = store.confirmed_text
        if final.strip():
            final = spacer.space([final.replace(" ", "")])[0]
        send({"type": "status", "status": "done", "final_text": final})

    except Exception as e:
        send({"type": "error", "message": str(e)})
        send({"type": "status", "status": "error"})
    finally:
        pool.lock.release()


# ------------------------------------------------------------------
# API endpoints
# ------------------------------------------------------------------

@app.get("/api/samples")
async def list_samples():
    """List available sample WAV files."""
    samples_dir = Path("data/utterances")
    if not samples_dir.exists():
        return {"files": []}
    files = sorted(f.name for f in samples_dir.glob("*.wav"))
    return {"files": files}


@app.post("/api/upload")
async def upload_audio(file: UploadFile):
    """Upload a WAV file and return a temp path."""
    suffix = Path(file.filename or "audio.wav").suffix
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=suffix, dir="/tmp")
    content = await file.read()
    tmp.write(content)
    tmp.close()
    return {"path": tmp.name, "filename": file.filename}


@app.websocket("/ws/{session_id}")
async def websocket_endpoint(websocket: WebSocket, session_id: str):
    """WebSocket endpoint for real-time subtitle streaming."""
    await websocket.accept()
    out_queue: queue.Queue = queue.Queue()
    sessions[session_id] = out_queue
    stop_event = threading.Event()

    try:
        while True:
            data = await websocket.receive_text()
            msg = json.loads(data)

            if msg.get("action") in ("start", "start_mic"):
                use_mic = msg.get("action") == "start_mic"
                audio_path = msg.get("path") if not use_mic else None
                if audio_path and not audio_path.startswith("/"):
                    audio_path = str(Path("data/utterances") / audio_path)
                context_mode = msg.get("context_mode", "accumulate")

                stop_event.clear()
                thread = threading.Thread(
                    target=_run_pipeline_thread,
                    args=(Path(audio_path) if audio_path else None, session_id, out_queue),
                    kwargs={"use_mic": use_mic, "stop_event": stop_event, "context_mode": context_mode},
                    daemon=True,
                )
                thread.start()

                loop = asyncio.get_event_loop()

                async def poll_events():
                    while True:
                        try:
                            event = await loop.run_in_executor(
                                None, lambda: out_queue.get(timeout=0.1)
                            )
                        except queue.Empty:
                            continue
                        await websocket.send_text(json.dumps(event, ensure_ascii=False))
                        if event.get("type") == "status" and event.get("status") in ("done", "error"):
                            return

                async def listen_commands():
                    while True:
                        data = await websocket.receive_text()
                        cmd = json.loads(data)
                        if cmd.get("action") == "stop":
                            stop_event.set()
                            return

                poll_task = asyncio.create_task(poll_events())
                listen_task = asyncio.create_task(listen_commands())

                done, pending = await asyncio.wait(
                    {poll_task, listen_task},
                    return_when=asyncio.FIRST_COMPLETED,
                )
                for task in pending:
                    task.cancel()
                    try:
                        await task
                    except asyncio.CancelledError:
                        pass

    except WebSocketDisconnect:
        stop_event.set()
    finally:
        sessions.pop(session_id, None)


# ------------------------------------------------------------------
# Web UI
# ------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def index():
    return HTML_PAGE


HTML_PAGE = """<!DOCTYPE html>
<html lang="ko">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Hybrid1 — 실시간 한국어 자막</title>
<style>
  :root {
    --bg: #0a0a0f;
    --surface: #12121a;
    --surface2: #1a1a26;
    --border: #2a2a3a;
    --text: #e4e4ef;
    --text-dim: #8888a0;
    --accent: #6c5ce7;
    --accent-glow: rgba(108,92,231,0.3);
    --live: #00cec9;
    --live-glow: rgba(0,206,201,0.2);
    --refine: #fd79a8;
    --refine-glow: rgba(253,121,168,0.2);
    --success: #00b894;
    --error: #ff6b6b;
    --radius: 16px;
    --radius-sm: 10px;
  }

  * { margin:0; padding:0; box-sizing:border-box; }

  body {
    font-family: 'Pretendard', -apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif;
    background: var(--bg);
    color: var(--text);
    min-height: 100vh;
    overflow-x: hidden;
  }

  /* Ambient background glow */
  body::before {
    content: '';
    position: fixed;
    top: -50%;
    left: -50%;
    width: 200%;
    height: 200%;
    background: radial-gradient(ellipse at 30% 20%, var(--accent-glow) 0%, transparent 50%),
                radial-gradient(ellipse at 70% 80%, var(--live-glow) 0%, transparent 50%);
    pointer-events: none;
    z-index: 0;
  }

  .container {
    max-width: 960px;
    margin: 0 auto;
    padding: 32px 24px;
    position: relative;
    z-index: 1;
  }

  /* Header */
  header {
    text-align: center;
    margin-bottom: 40px;
  }

  header h1 {
    font-size: 2.2rem;
    font-weight: 700;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, var(--accent), var(--live));
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    background-clip: text;
  }

  header p {
    color: var(--text-dim);
    font-size: 0.95rem;
    margin-top: 6px;
  }

  /* Cards */
  .card {
    background: var(--surface);
    border: 1px solid var(--border);
    border-radius: var(--radius);
    padding: 28px;
    margin-bottom: 20px;
    backdrop-filter: blur(10px);
    transition: border-color 0.3s;
  }

  .card:hover {
    border-color: color-mix(in srgb, var(--accent) 40%, var(--border));
  }

  .card-title {
    font-size: 0.8rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: var(--text-dim);
    margin-bottom: 16px;
  }

  /* Audio source selector */
  .source-tabs {
    display: flex;
    gap: 8px;
    margin-bottom: 20px;
  }

  .source-tab {
    flex: 1;
    padding: 12px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-dim);
    font-size: 0.9rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    text-align: center;
  }

  .source-tab:hover {
    border-color: var(--accent);
    color: var(--text);
  }

  .source-tab.active {
    background: color-mix(in srgb, var(--accent) 15%, var(--surface2));
    border-color: var(--accent);
    color: var(--text);
    box-shadow: 0 0 20px var(--accent-glow);
  }

  /* Sample file grid */
  .sample-grid {
    display: grid;
    grid-template-columns: repeat(auto-fill, minmax(120px, 1fr));
    gap: 8px;
    max-height: 200px;
    overflow-y: auto;
    padding-right: 4px;
  }

  .sample-grid::-webkit-scrollbar { width: 4px; }
  .sample-grid::-webkit-scrollbar-track { background: transparent; }
  .sample-grid::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  .sample-btn {
    padding: 10px 12px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text);
    font-size: 0.82rem;
    cursor: pointer;
    transition: all 0.2s;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
  }

  .sample-btn:hover {
    border-color: var(--live);
    background: color-mix(in srgb, var(--live) 10%, var(--surface2));
    transform: translateY(-1px);
  }

  .sample-btn.selected {
    border-color: var(--live);
    background: color-mix(in srgb, var(--live) 15%, var(--surface2));
    box-shadow: 0 0 12px var(--live-glow);
  }

  /* Upload zone */
  .upload-zone {
    border: 2px dashed var(--border);
    border-radius: var(--radius-sm);
    padding: 32px;
    text-align: center;
    color: var(--text-dim);
    cursor: pointer;
    transition: all 0.3s;
    display: none;
  }

  .upload-zone.active { display: block; }

  .upload-zone:hover, .upload-zone.dragover {
    border-color: var(--accent);
    color: var(--text);
    background: color-mix(in srgb, var(--accent) 5%, transparent);
  }

  .upload-zone input { display: none; }

  /* Start button */
  .btn-start {
    width: 100%;
    padding: 16px;
    background: linear-gradient(135deg, var(--accent), color-mix(in srgb, var(--accent) 70%, var(--live)));
    border: none;
    border-radius: var(--radius-sm);
    color: white;
    font-size: 1.05rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s;
    letter-spacing: 0.02em;
    margin-top: 16px;
  }

  .btn-start:hover:not(:disabled) {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px var(--accent-glow);
  }

  .btn-start:disabled {
    opacity: 0.5;
    cursor: not-allowed;
  }

  /* Progress bar */
  .progress-wrap {
    margin-top: 16px;
    display: none;
  }

  .progress-wrap.visible { display: block; }

  .progress-bar-bg {
    height: 6px;
    background: var(--surface2);
    border-radius: 3px;
    overflow: hidden;
  }

  .progress-bar {
    height: 100%;
    width: 0%;
    background: linear-gradient(90deg, var(--accent), var(--live));
    border-radius: 3px;
    transition: width 0.3s ease;
  }

  .progress-info {
    display: flex;
    justify-content: space-between;
    font-size: 0.78rem;
    color: var(--text-dim);
    margin-top: 6px;
  }

  /* Subtitle display */
  .subtitle-area {
    min-height: 200px;
    position: relative;
  }

  .subtitle-main {
    font-size: 1.5rem;
    font-weight: 500;
    line-height: 1.7;
    color: var(--text);
    word-break: keep-all;
    overflow-wrap: break-word;
    min-height: 80px;
    padding: 8px 0;
  }

  .subtitle-main .confirmed {
    color: var(--text);
  }

  .subtitle-main .live-suffix {
    color: var(--live);
    opacity: 0.85;
  }

  /* Status indicator */
  .status-bar {
    display: flex;
    align-items: center;
    gap: 10px;
    margin-bottom: 16px;
    padding: 10px 16px;
    background: var(--surface2);
    border-radius: var(--radius-sm);
    font-size: 0.85rem;
  }

  .status-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    background: var(--text-dim);
    flex-shrink: 0;
  }

  .status-dot.running {
    background: var(--live);
    box-shadow: 0 0 8px var(--live-glow);
    animation: pulse 1.5s ease infinite;
  }

  .status-dot.done { background: var(--success); }
  .status-dot.error { background: var(--error); }

  @keyframes pulse {
    0%, 100% { opacity: 1; }
    50% { opacity: 0.4; }
  }

  /* Event log */
  .event-log {
    max-height: 260px;
    overflow-y: auto;
    font-family: 'JetBrains Mono', 'Fira Code', monospace;
    font-size: 0.78rem;
    line-height: 1.8;
    padding-right: 4px;
  }

  .event-log::-webkit-scrollbar { width: 4px; }
  .event-log::-webkit-scrollbar-track { background: transparent; }
  .event-log::-webkit-scrollbar-thumb { background: var(--border); border-radius: 4px; }

  .log-entry {
    padding: 2px 0;
    border-bottom: 1px solid color-mix(in srgb, var(--border) 30%, transparent);
  }

  .log-tag {
    display: inline-block;
    padding: 1px 8px;
    border-radius: 4px;
    font-size: 0.72rem;
    font-weight: 600;
    margin-right: 6px;
  }

  .log-tag.LIVE {
    background: color-mix(in srgb, var(--live) 20%, transparent);
    color: var(--live);
  }

  .log-tag.REFINE {
    background: color-mix(in srgb, var(--refine) 20%, transparent);
    color: var(--refine);
  }

  .log-tag.LIVE-FINAL { background: color-mix(in srgb, var(--live) 30%, transparent); color: var(--live); }
  .log-tag.REFINE-FINAL { background: color-mix(in srgb, var(--refine) 30%, transparent); color: var(--refine); }

  .log-time {
    color: var(--text-dim);
    margin-right: 6px;
  }

  .log-text { color: var(--text); }

  /* Empty state */
  .empty-state {
    text-align: center;
    padding: 48px 20px;
    color: var(--text-dim);
  }

  .empty-state .icon {
    font-size: 3rem;
    margin-bottom: 12px;
    opacity: 0.4;
  }

  /* Responsive */
  @media (max-width: 640px) {
    .container { padding: 20px 16px; }
    header h1 { font-size: 1.6rem; }
    .subtitle-main { font-size: 1.2rem; }
    .sample-grid { grid-template-columns: repeat(auto-fill, minmax(100px, 1fr)); }
  }

  /* Mic panel */
  .mic-panel {
    display: none;
    text-align: center;
    padding: 40px 20px;
  }

  .mic-panel.active { display: block; }

  .mic-circle {
    width: 100px;
    height: 100px;
    border-radius: 50%;
    background: var(--surface2);
    border: 2px solid var(--border);
    display: inline-flex;
    align-items: center;
    justify-content: center;
    margin-bottom: 16px;
    transition: all 0.3s;
  }

  .mic-circle svg {
    width: 40px;
    height: 40px;
    fill: var(--text-dim);
    transition: fill 0.3s;
  }

  .mic-circle.recording {
    border-color: var(--error);
    background: color-mix(in srgb, var(--error) 10%, var(--surface2));
    box-shadow: 0 0 0 0 rgba(255,107,107,0.4);
    animation: mic-pulse 1.5s ease infinite;
  }

  .mic-circle.recording svg { fill: var(--error); }

  @keyframes mic-pulse {
    0% { box-shadow: 0 0 0 0 rgba(255,107,107,0.4); }
    70% { box-shadow: 0 0 0 20px rgba(255,107,107,0); }
    100% { box-shadow: 0 0 0 0 rgba(255,107,107,0); }
  }

  .mic-label {
    color: var(--text-dim);
    font-size: 0.9rem;
  }

  .mic-label.recording { color: var(--error); font-weight: 500; }

  .btn-stop {
    width: 100%;
    padding: 16px;
    background: linear-gradient(135deg, var(--error), color-mix(in srgb, var(--error) 70%, var(--refine)));
    border: none;
    border-radius: var(--radius-sm);
    color: white;
    font-size: 1.05rem;
    font-weight: 600;
    cursor: pointer;
    transition: all 0.3s;
    letter-spacing: 0.02em;
    margin-top: 16px;
    display: none;
  }

  .btn-stop:hover {
    transform: translateY(-2px);
    box-shadow: 0 8px 30px rgba(255,107,107,0.3);
  }

  /* Smooth scroll for log */
  .event-log { scroll-behavior: smooth; }

  /* Context mode toggle */
  .mode-toggle {
    display: flex;
    gap: 6px;
    margin-top: 16px;
    margin-bottom: 0;
  }

  .mode-tab {
    flex: 1;
    padding: 9px 12px;
    background: var(--surface2);
    border: 1px solid var(--border);
    border-radius: var(--radius-sm);
    color: var(--text-dim);
    font-size: 0.82rem;
    font-weight: 500;
    cursor: pointer;
    transition: all 0.2s;
    text-align: center;
  }

  .mode-tab:hover { border-color: var(--refine); color: var(--text); }

  .mode-tab.active {
    background: color-mix(in srgb, var(--refine) 15%, var(--surface2));
    border-color: var(--refine);
    color: var(--text);
    box-shadow: 0 0 14px var(--refine-glow);
  }

  .mode-label {
    font-size: 0.75rem;
    font-weight: 600;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: var(--text-dim);
    margin-top: 16px;
    margin-bottom: 6px;
  }
</style>
</head>
<body>

<div class="container">
  <header>
    <h1>Hybrid1 ASR</h1>
    <p>실시간 한국어 음성인식 자막 파이프라인</p>
  </header>

  <!-- Source Selection -->
  <div class="card">
    <div class="card-title">음성 소스</div>
    <div class="source-tabs">
      <div class="source-tab active" data-tab="samples" onclick="switchTab('samples')">샘플 파일</div>
      <div class="source-tab" data-tab="upload" onclick="switchTab('upload')">파일 업로드</div>
      <div class="source-tab" data-tab="mic" onclick="switchTab('mic')">마이크 입력</div>
    </div>

    <div id="samples-panel">
      <div class="sample-grid" id="sample-grid"></div>
    </div>

    <div id="upload-panel" class="upload-zone" onclick="document.getElementById('file-input').click()">
      <div style="font-size:2rem; margin-bottom:8px; opacity:0.5;">&#8593;</div>
      <div>WAV 파일을 드래그하거나 클릭하여 업로드</div>
      <div style="font-size:0.8rem; margin-top:6px; opacity:0.6;">16kHz mono 권장</div>
      <input type="file" id="file-input" accept=".wav,.mp3,.flac" onchange="handleFileSelect(event)">
    </div>

    <div id="mic-panel" class="mic-panel">
      <div class="mic-circle" id="mic-circle">
        <svg viewBox="0 0 24 24"><path d="M12 14c1.66 0 3-1.34 3-3V5c0-1.66-1.34-3-3-3S9 3.34 9 5v6c0 1.66 1.34 3 3 3zm-1-9c0-.55.45-1 1-1s1 .45 1 1v6c0 .55-.45 1-1 1s-1-.45-1-1V5zm6 6c0 2.76-2.24 5-5 5s-5-2.24-5-5H5c0 3.53 2.61 6.43 6 6.92V21h2v-3.08c3.39-.49 6-3.39 6-6.92h-2z"/></svg>
      </div>
      <div class="mic-label" id="mic-label">마이크를 사용하여 실시간 음성인식</div>
      <div style="color:var(--text-dim); font-size:0.78rem; margin-top:8px;">VAD로 자동 감지 &middot; 무음 시 자동 종료</div>
    </div>

    <div id="upload-info" style="display:none; margin-top:12px; padding:10px 16px; background:var(--surface2); border-radius:var(--radius-sm); font-size:0.85rem;"></div>

    <div class="mode-label">컨텍스트 모드</div>
    <div class="mode-toggle">
      <div class="mode-tab active" data-mode="accumulate" onclick="selectMode('accumulate')" title="최대 4개 청크를 누적하여 디코딩 (기본값)">누적청크</div>
      <div class="mode-tab" data-mode="overlap" onclick="selectMode('overlap')" title="고정 크기(2×trim_sec) 오버랩 버퍼만 유지">오버랩</div>
    </div>

    <button class="btn-start" id="btn-start" onclick="startPipeline()" disabled>
      파이프라인 시작
    </button>
    <button class="btn-stop" id="btn-stop" onclick="stopPipeline()">
      녹음 중지
    </button>

    <div class="progress-wrap" id="progress-wrap">
      <div class="progress-bar-bg"><div class="progress-bar" id="progress-bar"></div></div>
      <div class="progress-info">
        <span id="progress-time">0:00</span>
        <span id="progress-duration">0:00</span>
      </div>
    </div>
  </div>

  <!-- Subtitle Display -->
  <div class="card">
    <div class="card-title">자막 출력</div>
    <div class="status-bar">
      <div class="status-dot" id="status-dot"></div>
      <span id="status-text">대기 중</span>
    </div>
    <div class="subtitle-area">
      <div class="subtitle-main" id="subtitle-main">
        <div class="empty-state" id="empty-state">
          <div class="icon">&#127908;</div>
          <div>음성 파일을 선택하고 시작하세요</div>
        </div>
      </div>
    </div>
  </div>

  <!-- Event Log -->
  <div class="card">
    <div class="card-title">이벤트 로그</div>
    <div class="event-log" id="event-log">
      <div style="color:var(--text-dim); text-align:center; padding:20px; font-size:0.85rem;">
        로그가 여기에 표시됩니다
      </div>
    </div>
  </div>
</div>

<script>
let selectedFile = null;
let uploadedPath = null;
let ws = null;
let logStarted = false;
let currentTab = 'samples';
let isRecording = false;
let selectedMode = 'accumulate';

function selectMode(mode) {
  selectedMode = mode;
  document.querySelectorAll('.mode-tab').forEach(t => t.classList.toggle('active', t.dataset.mode === mode));
}

// Load sample files
async function loadSamples() {
  const res = await fetch('/api/samples');
  const data = await res.json();
  const grid = document.getElementById('sample-grid');
  grid.innerHTML = '';
  data.files.forEach(f => {
    const btn = document.createElement('div');
    btn.className = 'sample-btn';
    btn.textContent = f;
    btn.onclick = () => selectSample(f, btn);
    grid.appendChild(btn);
  });
}

function selectSample(filename, btn) {
  document.querySelectorAll('.sample-btn').forEach(b => b.classList.remove('selected'));
  btn.classList.add('selected');
  selectedFile = filename;
  uploadedPath = null;
  document.getElementById('btn-start').disabled = false;
}

function switchTab(tab) {
  currentTab = tab;
  document.querySelectorAll('.source-tab').forEach(t => t.classList.remove('active'));
  document.querySelector(`[data-tab="${tab}"]`).classList.add('active');

  if (isRecording) { stopPipeline(); setMicRecording(false); }

  document.getElementById('samples-panel').style.display = tab === 'samples' ? '' : 'none';
  document.getElementById('upload-panel').classList.toggle('active', tab === 'upload');
  document.getElementById('mic-panel').classList.toggle('active', tab === 'mic');

  const btn = document.getElementById('btn-start');
  if (tab === 'mic') {
    btn.disabled = false;
    btn.textContent = '녹음 시작';
  } else {
    btn.textContent = '파이프라인 시작';
    btn.disabled = !(selectedFile || uploadedPath);
  }
}

async function handleFileSelect(e) {
  const file = e.target.files[0];
  if (!file) return;

  const formData = new FormData();
  formData.append('file', file);

  const info = document.getElementById('upload-info');
  info.style.display = 'block';
  info.textContent = `업로드 중: ${file.name}...`;

  const res = await fetch('/api/upload', { method: 'POST', body: formData });
  const data = await res.json();

  uploadedPath = data.path;
  selectedFile = null;
  info.innerHTML = `<span style="color:var(--success);">&#10003;</span> ${data.filename} (업로드 완료)`;
  document.getElementById('btn-start').disabled = false;
}

// Drag & drop
const uploadZone = document.getElementById('upload-panel');
uploadZone.addEventListener('dragover', e => { e.preventDefault(); uploadZone.classList.add('dragover'); });
uploadZone.addEventListener('dragleave', () => uploadZone.classList.remove('dragover'));
uploadZone.addEventListener('drop', e => {
  e.preventDefault();
  uploadZone.classList.remove('dragover');
  const file = e.dataTransfer.files[0];
  if (file) {
    const input = document.getElementById('file-input');
    const dt = new DataTransfer();
    dt.items.add(file);
    input.files = dt.files;
    handleFileSelect({ target: input });
  }
});

function formatTime(sec) {
  const m = Math.floor(sec / 60);
  const s = Math.floor(sec % 60);
  return `${m}:${s.toString().padStart(2, '0')}`;
}

function addLogEntry(tag, time, text) {
  const log = document.getElementById('event-log');
  if (!logStarted) {
    log.innerHTML = '';
    logStarted = true;
  }

  const entry = document.createElement('div');
  entry.className = 'log-entry';

  const tagClass = tag.replace('-', '-');
  entry.innerHTML = `<span class="log-tag ${tag}">${tag}</span><span class="log-time">${time}</span><span class="log-text">${escapeHtml(text)}</span>`;

  log.appendChild(entry);

  // Keep only last 200 entries
  while (log.children.length > 200) log.removeChild(log.firstChild);

  log.scrollTop = log.scrollHeight;
}

function escapeHtml(text) {
  const div = document.createElement('div');
  div.textContent = text;
  return div.innerHTML;
}

function setStatus(status, text) {
  const dot = document.getElementById('status-dot');
  const txt = document.getElementById('status-text');
  dot.className = 'status-dot ' + status;
  txt.textContent = text;
}

function stopPipeline() {
  if (ws && ws.readyState === WebSocket.OPEN) {
    ws.send(JSON.stringify({ action: 'stop' }));
  }
}

function setMicRecording(on) {
  isRecording = on;
  document.getElementById('mic-circle').classList.toggle('recording', on);
  const label = document.getElementById('mic-label');
  label.textContent = on ? '녹음 중...' : '마이크를 사용하여 실시간 음성인식';
  label.classList.toggle('recording', on);
  document.getElementById('btn-stop').style.display = on ? 'block' : 'none';
}

function onPipelineEnd(btn) {
  btn.disabled = false;
  btn.textContent = currentTab === 'mic' ? '녹음 시작' : '파이프라인 시작';
  if (currentTab === 'mic') setMicRecording(false);
}

function startPipeline() {
  const isMic = currentTab === 'mic';
  const path = isMic ? null : (uploadedPath || selectedFile);
  if (!isMic && !path) return;

  const sessionId = Date.now().toString(36) + Math.random().toString(36).slice(2);
  const proto = location.protocol === 'https:' ? 'wss' : 'ws';
  ws = new WebSocket(`${proto}://${location.host}/ws/${sessionId}`);

  const btn = document.getElementById('btn-start');
  btn.disabled = true;
  btn.textContent = '처리 중...';
  if (isMic) setMicRecording(true);

  const subtitleMain = document.getElementById('subtitle-main');
  const emptyState = document.getElementById('empty-state');
  if (emptyState) emptyState.style.display = 'none';
  subtitleMain.innerHTML = '<span style="color:var(--text-dim)">준비 중...</span>';

  logStarted = false;
  document.getElementById('event-log').innerHTML = '<div style="color:var(--text-dim); text-align:center; padding:10px;">파이프라인 시작 중...</div>';

  ws.onopen = () => {
    if (isMic) {
      ws.send(JSON.stringify({ action: 'start_mic', context_mode: selectedMode }));
      setStatus('running', '녹음 중');
    } else {
      ws.send(JSON.stringify({ action: 'start', path: path, context_mode: selectedMode }));
      setStatus('running', '처리 중');
    }
  };

  ws.onmessage = (e) => {
    const msg = JSON.parse(e.data);

    if (msg.type === 'subtitle') {
      const confirmed = msg.confirmed || '';
      const fullText = msg.text || '';

      let html = '';
      if (confirmed && fullText.startsWith(confirmed)) {
        const suffix = fullText.slice(confirmed.length);
        html = `<span class="confirmed">${escapeHtml(confirmed)}</span><span class="live-suffix">${escapeHtml(suffix)}</span>`;
      } else if (confirmed) {
        html = `<span class="confirmed">${escapeHtml(fullText)}</span>`;
      } else {
        html = `<span class="live-suffix">${escapeHtml(fullText)}</span>`;
      }
      subtitleMain.innerHTML = html;

      const timeStr = formatTime(msg.start) + '~' + formatTime(msg.end);
      addLogEntry(msg.tag, timeStr, msg.text);
    }

    else if (msg.type === 'progress') {
      const wrap = document.getElementById('progress-wrap');
      wrap.classList.add('visible');
      const pct = (msg.current / msg.duration * 100).toFixed(1);
      document.getElementById('progress-bar').style.width = pct + '%';
      document.getElementById('progress-time').textContent = formatTime(msg.current);
      document.getElementById('progress-duration').textContent = formatTime(msg.duration);
    }

    else if (msg.type === 'status') {
      if (msg.status === 'done') {
        setStatus('done', '완료');
        onPipelineEnd(btn);
        document.getElementById('progress-bar').style.width = '100%';
        if (msg.final_text) {
          subtitleMain.innerHTML = `<span class="confirmed">${escapeHtml(msg.final_text)}</span>`;
        }
      } else if (msg.status === 'error') {
        setStatus('error', '오류 발생');
        onPipelineEnd(btn);
      } else if (msg.status === 'loading') {
        setStatus('running', '모델 로딩 중...');
        if (msg.duration) {
          document.getElementById('progress-duration').textContent = formatTime(msg.duration);
        }
      }
    }

    else if (msg.type === 'error') {
      addLogEntry('ERROR', '', msg.message);
    }
  };

  ws.onclose = () => {
    if (btn.textContent === '처리 중...') {
      onPipelineEnd(btn);
      setStatus('', '연결 종료');
    }
  };
}

// Init
loadSamples();
</script>
</body>
</html>"""

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8080, log_level="info")
