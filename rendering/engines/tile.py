from __future__ import annotations

from typing import Optional, Callable, List, Tuple
import numpy as np

from fractals.base import Fractal, Viewport, RenderSettings
from rendering.engines.base import BaseRenderEngine
from rendering.executor import RenderExecutor


class TileEngine(BaseRenderEngine):
    """
    Fixed-grid tiled rendering:
      - Splits the viewport into (tile_w × tile_h) tiles.
      - Submits each tile via exec_or_mgr.render_async(...).
      - Waits for completion batch-by-batch and emits tiles as they finish.
      - Assembles and returns the full iteration canvas (H x W).

    Backwards compatible:
      - `exec_or_mgr` may be a BackendManager today (Renderer passes it),
        or a RenderExecutor later; both expose .render_async(...) with the same signature.
    """

    def __init__(
        self,
        tile_w: int = 256,
        tile_h: int = 256,
        order: str = "scanline",   # "scanline" | "center-first"
        on_tile: Optional[Callable[[int, int, np.ndarray], None]] = None
    ) -> None:
        super().__init__(on_tile=on_tile)
        self.tile_w = int(tile_w)
        self.tile_h = int(tile_h)
        self.order = order

        # Hint to allow simple pipelining (N tiles in flight).
        # Can be tuned from outside via configure_concurrency(...).
        self._in_flight_hint = 2

    @property
    def supports_async(self) -> bool:
        # We make async submissions but return a fully assembled frame.
        return True

    # ---- Public API -----------------------------------------------------

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
        W, H = int(viewport.width), int(viewport.height)
        tw, th = max(1, int(self.tile_w)), max(1, int(self.tile_h))

        # Output canvas (iteration counts), assembled progressively
        canvas = np.zeros((H, W), dtype=settings.precision)

        # Build tile list
        tiles = self._compute_tiles(W, H, tw, th)
        if self.order == "center-first":
            tiles = self._order_center_first(tiles, W, H)

        # Simple in-flight batching: submit up to N tiles, then drain results
        n = max(1, int(self._in_flight_hint))
        idx = 0
        while idx < len(tiles):
            if cancel_cb is not None and cancel_cb():
                break

            batch = tiles[idx:idx + n]
            idx += len(batch)

            # Submit
            in_flight: List[Tuple[Tuple[int, int, int, int], any]] = []
            for (x0, y0, w, h) in batch:
                sub_vp = self._make_sub_viewport(viewport, x0, y0, w, h)  # same math as AdaptiveTileEngine
                handle = executor.render_async(
                    fractal, sub_vp, settings, backend=backend, device=device
                )
                in_flight.append(((x0, y0, w, h), handle))

            # Gather in submission order (RenderHandle doesn’t expose futures; .result will block)
            for (x0, y0, w, h), handle in in_flight:
                if cancel_cb is not None and cancel_cb():
                    break
                tile = handle.result  # waits for GPU/CL to complete (RenderHandle API)
                # Commit & emit
                canvas[y0:y0 + h, x0:x0 + w] = tile
                self.emit_tile(x0, y0, tile)

            if cancel_cb is not None and cancel_cb():
                break

        # Ensure dtype consistency
        if canvas.dtype != settings.precision:
            canvas = canvas.astype(settings.precision, copy=False)

        return canvas

    # ---- Helpers --------------------------------------------------------

    @staticmethod
    def _compute_tiles(W: int, H: int, tw: int, th: int) -> List[Tuple[int, int, int, int]]:
        tiles: List[Tuple[int, int, int, int]] = []
        for y0 in range(0, H, th):
            h = min(th, H - y0)
            if h <= 0: break
            for x0 in range(0, W, tw):
                w = min(tw, W - x0)
                if w <= 0: break
                tiles.append((x0, y0, w, h))
        return tiles

    @staticmethod
    def _order_center_first(tiles: List[Tuple[int, int, int, int]], W: int, H: int) -> List[Tuple[int, int, int, int]]:
        cx, cy = (W - 1) * 0.5, (H - 1) * 0.5
        # sort by tile center distance to image center
        def key(t):
            x0, y0, w, h = t
            tx, ty = x0 + 0.5 * w, y0 + 0.5 * h
            dx, dy = tx - cx, ty - cy
            return dx * dx + dy * dy
        return sorted(tiles, key=key)

    @staticmethod
    def _make_sub_viewport(vp: Viewport, x0: int, y0: int, w: int, h: int) -> Viewport:
        """
        maps pixel rect (x0..x0+w, y0..y0+h) to fractal-space sub-viewport.
        """
        W, H = int(vp.width), int(vp.height)
        sx, ex = x0 / W, (x0 + w) / W
        sy, ey = y0 / H, (y0 + h) / H
        sub_min_x = vp.min_x + (vp.max_x - vp.min_x) * sx
        sub_max_x = vp.min_x + (vp.max_x - vp.min_x) * ex
        sub_min_y = vp.min_y + (vp.max_y - vp.min_y) * sy
        sub_max_y = vp.min_y + (vp.max_y - vp.min_y) * ey
        return Viewport(sub_min_x, sub_max_x, sub_min_y, sub_max_y, w, h)
