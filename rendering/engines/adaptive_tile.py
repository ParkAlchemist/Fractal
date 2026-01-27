from __future__ import annotations

import time
from typing import Optional, Callable, List
import numpy as np

from fractals.base import Viewport, RenderSettings, Fractal
from rendering.engines.base import BaseRenderEngine
from rendering.executor import RenderExecutor
from rendering.scheduling.tilescheduler import TileScheduler, TileInfo
from rendering.scheduling.tilescorer import TileScorer


class AdaptiveTileEngine(BaseRenderEngine):
    """
    Dynamic tiling via quadtree and priority queue.

    - Starts with coarse tiles (max_tile).
    - Measures compute time and variance, then splits expensive tiles (2x2).
    - Prioritization handled by TileScheduler (center/visibility/variance/age).
    - Streams tiles via `self.emit_tile(x0, y0, tile)` as they finish.
    - Returns the assembled iteration canvas (H x W, dtype=settings.precision).
    """

    def __init__(
        self,
        min_tile: int = 64,
        max_tile: int = 512,
        target_ms: float = 12.0,
        max_depth: int = 4,
        sample_stride: int = 8,
        on_tile: Optional[Callable[[int, int, np.ndarray], None]] = None
    ):
        super().__init__(on_tile=on_tile)
        self.min_tile = int(min_tile)
        self.max_tile = int(max_tile)
        self.target_ms = float(target_ms)
        self.max_depth = int(max_depth)
        self.sample_stride = int(sample_stride)

        # Motion state (used by scheduler/scorer)
        self._last_viewport: Optional[Viewport] = None

        # Weights (kept from your original engine)
        weights_motion = {
            'vis': 1.0, 'center': 0.5, 'area': 0.7, 'motion': 1.0, 'age': 0.2, 'var': 0.2, 'bnd': 0.2
        }
        weights_idle = {
            'vis': 1.0, 'center': 0.7, 'area': 0.6, 'motion': 0.1, 'age': 0.3, 'var': 0.8, 'bnd': 0.6
        }
        self.scorer = TileScorer(weights_motion=weights_motion, weights_idle=weights_idle)
        self.scheduler = TileScheduler(self.scorer)

        # Allow simple parallelism later (N tiles in flight) without changing API
        self._in_flight_hint = 1

    @property
    def supports_async(self) -> bool:
        return True

    # ---------------------------------------------------------------------
    # Public API
    # ---------------------------------------------------------------------

    def render(
        self,
        fractal: Fractal,
        executor: RenderExecutor,
        settings: RenderSettings,
        viewport: Viewport,
        cancel_cb: Optional[Callable[[], bool]] = None,
        *,
        backend: Optional[str] = None,
        device: int = 0
    ) -> np.ndarray:
        W, H = int(viewport.width), int(viewport.height)
        canvas = np.zeros((H, W), dtype=settings.precision)

        # Update scheduler with current/previous view (your original intent)
        self.scheduler.update_view(W, H, viewport, self._last_viewport)
        self._last_viewport = viewport

        # Seed root tiles (aligned to 32 like before)
        roots = self._seed_root_tiles(W, H, self.max_tile)   # preserved behavior
        self.scheduler.clear()
        now = time.perf_counter()
        for x0, y0, w, h, depth in roots:
            ti = TileInfo(x0=x0, y0=y0, w=w, h=h, depth=depth, enqueue_time=now)
            self.scheduler.enqueue(ti, visible=True)

        # Local helper to make a RenderSettings clone (allows per-tile tweaks if needed)
        def make_settings(base: RenderSettings, max_iter: int) -> RenderSettings:
            return RenderSettings(max_iter=max_iter, samples=base.samples, precision=base.precision)

        # Main loop: pop by priority, compute, maybe split, enqueue children
        try:
            while True:
                # Cancellation check
                if cancel_cb is not None and cancel_cb():
                    self.scheduler.clear()
                    break

                popped = self.scheduler.pop_next()
                if popped is None:
                    break  # done

                phase, ti = popped

                # Clamp to bounds
                ti.w = min(ti.w, W - ti.x0)
                ti.h = min(ti.h, H - ti.y0)
                if ti.w <= 0 or ti.h <= 0:
                    continue

                # Sub-viewport & per-tile settings (kept equal to global max_iter for now)
                sub_vp = self._make_sub_viewport(viewport, ti.x0, ti.y0, ti.w, ti.h)
                st = make_settings(settings, settings.max_iter)

                # Submit & WAIT (measure *actual* compute time)
                t0 = time.perf_counter()
                handle = executor.render_async(fractal, sub_vp, st, backend=backend, device=device)  # BackendManager path today
                tile_iters = handle.result  # blocks until tile completes (RenderHandle API)
                t_ms = (time.perf_counter() - t0) * 1000.0

                # Commit tile & notify
                canvas[ti.y0:ti.y0 + ti.h, ti.x0:ti.x0 + ti.w] = tile_iters
                self.emit_tile(ti.x0, ti.y0, tile_iters)

                # Update stats used by scheduler/scorer
                ti.iteration_variance = self._compute_iteration_variance(tile_iters)
                ti.boundary_likelihood = self._compute_boundary_likelihood(tile_iters)

                # Decide whether to split
                should_split = (ti.depth < self.max_depth) and self._should_split(tile_iters, t_ms, ti.w, ti.h)
                if should_split:
                    for child in self._split((ti.x0, ti.y0, ti.w, ti.h, ti.depth), W, H):
                        cti = TileInfo(
                            x0=child[0], y0=child[1],
                            w=child[2],  h=child[3],
                            depth=child[4],
                            enqueue_time=ti.enqueue_time
                        )
                        self.scheduler.enqueue(cti, visible=self._visible(cti, W, H))

        except Exception as e:
            # Keep same failure mode as before—surface to service logs
            print(f"Error in adaptive render loop: {e}")

        # Ensure dtype consistency
        if canvas.dtype != settings.precision:
            canvas = canvas.astype(settings.precision, copy=False)

        return canvas

    # ---------------------------------------------------------------------
    # Helpers
    # ---------------------------------------------------------------------

    def _seed_root_tiles(self, W: int, H: int, max_tile: int) -> List[tuple[int, int, int, int, int]]:
        """
        Cover the viewport with large blocks aligned to multiples of 32.
        """
        step = max(self._align_32(max_tile), 32)
        tiles: List[tuple[int, int, int, int, int]] = []
        for y in range(0, H, step):
            for x in range(0, W, step):
                w = min(step, W - x)
                h = min(step, H - y)
                tiles.append((x, y, w, h, 0))
        return tiles

    @staticmethod
    def _align_32(n: int) -> int:
        return max((n // 32) * 32, 32)

    def _should_split(self, tile_iters: np.ndarray, t_ms: float, w: int, h: int) -> bool:
        """
        Decide whether to split a tile further based on time and variance.
        """
        if w <= self.min_tile and h <= self.min_tile:
            return False
        time_heavy = t_ms > self.target_ms
        var = float(np.var(tile_iters[::self.sample_stride, ::self.sample_stride]))
        return time_heavy or var > 0.5

    def _split(self, tile: tuple[int, int, int, int, int], W: int, H: int) -> List[tuple[int, int, int, int, int]]:
        x0, y0, w, h, depth = tile
        w2 = max(self.min_tile, self._align_32(w // 2))
        h2 = max(self.min_tile, self._align_32(h // 2))
        children = [
            (x0,         y0,         w2,        h2,        depth + 1),
            (x0 + w2,    y0,         w - w2,    h2,        depth + 1),
            (x0,         y0 + h2,    w2,        h - h2,    depth + 1),
            (x0 + w2,    y0 + h2,    w - w2,    h - h2,    depth + 1),
        ]
        clamped: List[tuple[int, int, int, int, int]] = []
        for cx, cy, cw, ch, d in children:
            cw = min(cw, W - cx)
            ch = min(ch, H - cy)
            if cw > 0 and ch > 0:
                clamped.append((cx, cy, cw, ch, d))
        return clamped

    @staticmethod
    def _make_sub_viewport(vp: Viewport, x0: int, y0: int, w: int, h: int) -> Viewport:
        W, H = int(vp.width), int(vp.height)
        sx, ex = x0 / W, (x0 + w) / W
        sy, ey = y0 / H, (y0 + h) / H
        sub_min_x = vp.min_x + (vp.max_x - vp.min_x) * sx
        sub_max_x = vp.min_x + (vp.max_x - vp.min_x) * ex
        sub_min_y = vp.min_y + (vp.max_y - vp.min_y) * sy
        sub_max_y = vp.min_y + (vp.max_y - vp.min_y) * ey
        return Viewport(sub_min_x, sub_max_x, sub_min_y, sub_max_y, w, h)

    @staticmethod
    def _visible(ti: TileInfo, W: int, H: int) -> bool:
        return ti.x0 < W and ti.y0 < H and ti.w > 0 and ti.h > 0

    def _compute_iteration_variance(self, tile_iters: np.ndarray) -> float:
        sample = tile_iters[::self.sample_stride, ::self.sample_stride]
        v = float(np.var(sample))
        return max(0.0, min(1.0, v))

    @staticmethod
    def _compute_boundary_likelihood(tile_iters: np.ndarray) -> float:
        total = tile_iters.size
        if total == 0:
            return 0.0
        low, high = 0.3, 0.7
        mid = int(np.count_nonzero((tile_iters >= low) & (tile_iters <= high)))
        return max(0.0, min(1.0, mid / total))
