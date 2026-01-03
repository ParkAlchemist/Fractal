from typing import Optional, Callable

import numpy as np

from fractals.base import Fractal, RenderSettings, Viewport
from rendering.engines.base import BaseRenderEngine
from backend.manager import BackendManager


class FullFrameEngine(BaseRenderEngine):
    """
    Rendering engine for full-frame rendering.
    """
    def render(self, fractal: Fractal, manager: BackendManager, settings: RenderSettings,
               vp: Viewport,
               cancel_cb: Optional[Callable[[], bool]] = None,
               backend: Optional[str] = None,
               device: Optional[int] = None) -> np.ndarray:
        return manager.render_async(fractal, vp, settings, backend=backend, device=device).result
