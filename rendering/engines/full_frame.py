from __future__ import annotations

from typing import Optional, Callable
import numpy as np

from fractals.base import Fractal, Viewport, RenderSettings
from rendering.engines.base import BaseRenderEngine
from rendering.executor import RenderExecutor


class FullFrameEngine(BaseRenderEngine):
    """
    Full-frame rendering strategy:
      - Delegates a single blocking render to the provided executor.
      - Returns the iteration canvas (H x W, dtype=settings.precision).
      - No tiles are emitted (on_tile is unused here).
    """

    @property
    def supports_async(self) -> bool:
        # We perform one blocking call; nothing to stream.
        return False

    def render(
        self,
        fractal: Fractal,
        executor: RenderExecutor,
        settings: RenderSettings,
        viewport: Viewport,
        cancel_cb: Optional[Callable[[], bool]] = None,
        *,
        backend: Optional[str] = None,
        device: Optional[int] = None,
    ) -> np.ndarray:
        # If the caller already cancelled (e.g., a new render started), return an empty canvas.
        if cancel_cb is not None and cancel_cb():
            return np.zeros((viewport.height, viewport.width), dtype=settings.precision)

        # Delegate to the execution layer
        canvas = executor.render(
            fractal, viewport, settings, backend=backend, device=device
        )

        # (Optional defensive cast if a backend drifts; normally unnecessary.)
        if canvas.dtype != settings.precision:
            canvas = canvas.astype(settings.precision, copy=False)

        return canvas
