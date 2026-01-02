from dataclasses import dataclass
import numpy as np
from typing import Optional

@dataclass(frozen=True)
class FrameEvent:
    data: np.ndarray
    width: int
    height: int
    seq: int        # generation / render sequence number

@dataclass(frozen=True)
class TileEvent:
    x: int
    y: int
    w: int
    h: int
    data: np.ndarray
    seq: int            # generation / render sequence number
    frame_w: int
    frame_h: int

@dataclass(frozen=True)
class LogEvent:
    message: str
    level: Optional[int]
