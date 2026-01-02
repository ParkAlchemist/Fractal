from abc import ABC, abstractmethod
from typing import Optional, Callable

import numpy as np

from backend.manager import BackendManager
from fractals.base import Fractal, RenderSettings, Viewport


class BaseRenderEngine(ABC):
    """
    Base class for rendering engines.
    """
    def __init__(self, on_tile: Optional[Callable[[int, int, np.ndarray], None]] = None):
        self.on_tile = on_tile

    @abstractmethod
    def render(self,
               fractal: Fractal,
               manager: BackendManager,
               settings: RenderSettings,
               vp: Viewport,
               cancel_cb: Optional[Callable[[], bool]] = None,
               backend: Optional[str] = None,
               device: Optional[int] = None) -> np.ndarray:
        ...
