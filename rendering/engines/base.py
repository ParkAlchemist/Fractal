from __future__ import annotations

from typing import Optional, Callable
import numpy as np

from fractals.base import Fractal, Viewport, RenderSettings
from rendering.executor import RenderExecutor


class BaseRenderEngine:
    """
    Base class for all render engines (full-frame, tiled, adaptive).

    Responsibilities:
      - Decide *how* to decompose a viewport into work (strategy),
      - Emit partial results via on_tile (if applicable),
      - Delegate *execution* to an executor.
    """

    def __init__(self, on_tile: Optional[Callable[[int, int, np.ndarray], None]] = None) -> None:
        # Engines may stream tiles; service composes + colorizes.  (Service uses this to raise TileEvent)
        self.on_tile: Optional[Callable[[int, int, np.ndarray], None]] = on_tile
        # Hint for future parallel execution (N tiles in flight). Optional for engines to use.
        self._in_flight_hint: int = 1

    # ---- Optional capabilities -----------------------------------------

    @property
    def supports_async(self) -> bool:
        """
        Engines that internally submit async work (e.g., through executor.render_async)
        can override this to True. Default False keeps expectations simple for now.
        """
        return False

    def configure_concurrency(self, in_flight_tiles: int = 1) -> None:
        """
        Optional hint: how many tiles this engine would like to keep in flight.
        The executor may use this later to size its queues/workers.
        """
        self._in_flight_hint = max(1, int(in_flight_tiles))

    # ---- Convenience for tile emitters ---------------------------------

    def emit_tile(self, x0: int, y0: int, tile_iters: np.ndarray) -> None:
        """
        Call this from subclasses when a tile finishes computing.
        RenderService subscribes to on_tile and composes into the canvas.
        """
        cb = self.on_tile
        if callable(cb):
            cb(int(x0), int(y0), tile_iters)

    # ---- Abstract entry point ------------------------------------------

    def render(
        self,
        fractal: Fractal,
        exec_or_mgr: RenderExecutor,
        settings: RenderSettings,
        viewport: Viewport,
        cancel_cb: Optional[Callable[[], bool]] = None,
        *,
        backend: Optional[str] = None,
        device: Optional[int] = None,
    ) -> np.ndarray:
        """
        Subclasses must implement the strategy:
          - FULL-FRAME: call exec_or_mgr.render(...)
          - TILED/ADAPTIVE: loop, submit tiles (sync or async), call self.emit_tile(...),
            and assemble/return the full iteration canvas.

        Must return an iteration buffer of shape (H, W) in settings.precision.
        """
        raise NotImplementedError("BaseRenderEngine.render() must be implemented by subclasses.")
