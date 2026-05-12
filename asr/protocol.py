from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from asr.transducer_stage import TransducerUpdate
from asr.refine_stage import RefineUpdate


@dataclass(slots=True)
class TransducerTask:
    """Message for the transducer worker."""

    action: Literal["feed", "finalize"]
    end_sec: float
    samples: np.ndarray | None = None


@dataclass(slots=True)
class RefineTask:
    """Message for the SenseVoice refine worker."""

    action: Literal["feed", "finalize", "configure"]
    samples: np.ndarray | None = None
    context_mode: str | None = None


@dataclass(slots=True)
class TransducerEvent:
    """Result message emitted by the transducer worker."""

    kind: Literal["update", "done", "error"]
    update: TransducerUpdate | None = None
    error: str | None = None


@dataclass(slots=True)
class RefineEvent:
    """Result message emitted by the SenseVoice refine worker."""

    kind: Literal["update", "done", "error"]
    update: RefineUpdate | None = None
    error: str | None = None
