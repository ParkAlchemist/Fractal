import time
from typing import Optional, Callable, Tuple, Any

import numpy as np

from fractals.base import Viewport, RenderSettings
from rendering.engines.base import BaseRenderEngine
from rendering.scheduling.tilescheduler import TileScheduler, TileInfo
from rendering.scheduling.tilescorer import TileScorer
from backend.manager import BackendManager


class AdaptiveTileEngine(BaseRenderEngine):
    """
    Dynamic tiling via quadtree and priority queue.
    - Starts with coarse tiles (max_tile).
    - Measures cost and variance, then splits expensive tiles (2x2).
    - Prioritization handled by TileScheduler
    - Can run sequentially or with a thread pool for parallel speed.

    Signals:
      on_tile(x0, y0, tile_iters) is called whenever a tile finishes.
    """

    def __init__(self,
                 min_tile: int = 64,
                 max_tile: int = 512,
                 target_ms: float = 12.0,
                 max_depth: int = 4,
                 sample_stride: int = 8,
                 on_tile: Optional[Callable[[int, int, np.ndarray], None]] = None
                 ):
        super().__init__(on_tile)
        self.min_tile = int(min_tile)
        self.max_tile = int(max_tile)
        self.target_ms = float(target_ms)
        self.max_depth = int(max_depth)
        self.sample_stride = int(sample_stride)

        # Motion state
        self._last_viewport: Optional[Viewport] = None

        # Weights
        weights_motion = {
            'vis': 1.0,
            'center': 0.5,
            'area': 0.7,
            'motion': 1.0,
            'age': 0.2,
            'var': 0.2,
            'bnd': 0.2
        }
        weights_idle = {
            'vis': 1.0,
            'center': 0.7,
            'area': 0.6,
            'motion': 0.1,
            'age': 0.3,
            'var': 0.8,
            'bnd': 0.6
        }

        self.scorer = TileScorer(weights_motion=weights_motion,
                                 weights_idle=weights_idle)
        self.scheduler = TileScheduler(self.scorer)


    # ----------- Render entry point -----------------------------------------
    def render(self,
               fractal,
               manager: BackendManager,
               settings: RenderSettings,
               viewport: Viewport,
               cancel_cb: Optional[Callable[[], bool]] = None,
               backend: Optional[str] = None,
               device: int = 0) -> np.ndarray:
        W, H = viewport.width, viewport.height
        canvas = np.zeros((H, W), dtype=settings.precision)

        self.scheduler.update_view(W, H, viewport, self._last_viewport)
        self._last_viewport = viewport

        # Root tiles: choose coarse tiling (e.g., 1x1 or 2x2 of max_tile-ish blocks)
        roots = self._seed_root_tiles(W, H, self.max_tile)

        self.scheduler.clear()
        now = time.perf_counter()
        for x0, y0, w, h, depth in roots:
            ti = TileInfo(x0=x0, y0=y0, w=w, h=h, depth=depth, enqueue_time=now)
            self.scheduler.enqueue(ti, visible=True)

        # Local settings
        def make_settings(base: RenderSettings, max_iter: int) -> RenderSettings:
            return RenderSettings(max_iter=max_iter,
                               samples=base.samples,
                               precision=base.precision)

        def compute_tile(ti: TileInfo, phase: str = None) -> Tuple[TileInfo, Any, bool | None]:
            # Clamp image to view bounds
            ti.w = min(ti.w, W - ti.x0)
            ti.h = min(ti.h, H - ti.y0)
            if ti.w <= 0 or ti.h <= 0: return ti, None, None

            # Sub-viewport and settings for this tile
            sub_vp = self._make_sub_viewport(viewport, ti.x0, ti.y0, ti.w, ti.h)
            st = make_settings(settings, settings.max_iter)

            # Render
            t0 = time.perf_counter()
            handle = manager.render_async(fractal, sub_vp, st, backend=backend, device=device)
            t_ms = (time.perf_counter() - t0) * 1000.0

            # Commit tile to canvas
            canvas[ti.y0:ti.y0+ti.h, ti.x0:ti.x0+ti.w] = tile_iters

            if callable(self.on_tile):
                self.on_tile(ti.x0, ti.y0, tile_iters)

            ti.iteration_variance = self._compute_iteration_variance(tile_iters)
            ti.boundary_likelihood = self._compute_boundary_likelihood(tile_iters)
            should_split = (ti.depth < self.max_depth) and self._should_split(tile_iters, t_ms, ti.w, ti.h)
            return ti, tile_iters, should_split

        try:
            while True:
                if cancel_cb is not None and cancel_cb():
                    # The current operation has become obsolete -> return
                    self.scheduler.clear()
                    break

                popped = self.scheduler.pop_next()
                if popped is None:
                    # Queue empty -> return
                    break

                phase, ti = popped
                ti, tile_iters, should_split = compute_tile(ti, phase)

                if should_split:
                    for child in self._split((ti.x0, ti.y0, ti.w, ti.h, ti.depth), W, H):
                        cti = TileInfo(x0=child[0], y0=child[1],
                                       w=child[2], h=child[3],
                                       depth=child[4],
                                       enqueue_time=ti.enqueue_time)
                        self.scheduler.enqueue(cti, visible=self._visible(cti, W, H))
        except Exception as e:
            print(f"Error in render loop: {e}")

        return canvas


    # ------- Helpers -------------------------------------
    def _seed_root_tiles(self, W: int, H: int,
                         max_tile: int) -> list[tuple[int, int, int, int, int]]:
        """
        Cover the viewport with large blocks aligned to multiples of 32
        """
        step = max(self._align_32(max_tile), 32)
        tiles = []
        for y in range(0, H, step):
            for x in range(0, W, step):
                w = min(step, W - x)
                h = min(step, H - y)
                tiles.append((x, y, w, h, 0))
        return tiles

    @staticmethod
    def _align_32(n):
        # Round down to the nearest multiple of 32
        return max((n // 32) * 32, 32)

    def _should_split(self, tile_iters: np.ndarray,
                      t_ms: float, w: int, h: int) -> bool:
        """
        Check whether the given tile should be split or not
        :return: True or False
        """
        if w <= self.min_tile and h <= self.min_tile:
            return False

        time_heavy = t_ms > self.target_ms
        var = float(np.var(tile_iters[::self.sample_stride, ::self.sample_stride]))

        return time_heavy or var > 0.5

    def _split(self,
               tile: tuple[int, int, int, int, int],
               W: int, H: int) -> list[tuple[int, int, int, int, int]]:
        """
        Splits given tile into 4 tiles
        :return: split tiles
        """
        x0, y0, w, h, depth = tile
        w2 = max(self.min_tile, self._align_32(w // 2))
        h2 = max(self.min_tile, self._align_32(h // 2))
        # 2x2 quad children
        children = [
            (x0,      y0,      w2,     h2,     depth + 1),
            (x0 + w2, y0,      w - w2, h2,     depth + 1),
            (x0,      y0 + h2, w2,     h - h2, depth + 1),
            (x0 + w2, y0 + h2, w - w2, h - h2, depth + 1),
        ]
        # Clamp to image bounds
        clamped = []
        for cx, cy, cw, ch, d in children:
            cw = min(cw, W - cx)
            ch = min(ch, H - cy)
            if cw > 0 and ch > 0:
                clamped.append((cx, cy, cw, ch, d))
        return clamped

    @staticmethod
    def _make_sub_viewport(vp: Viewport, x0: int, y0: int,
                           w: int, h: int) -> Viewport:
        """
        Creates a sub_viewport on given coordinates from the given viewport
        :return: sub_viewport
        """
        W, H = vp.width, vp.height
        sx, ex = x0 / W, (x0 + w) / W
        sy, ey = y0 / H, (y0 + h) / H
        sub_min_x = vp.min_x + (vp.max_x - vp.min_x) * sx
        sub_max_x = vp.min_x + (vp.max_x - vp.min_x) * ex
        sub_min_y = vp.min_y + (vp.max_y - vp.min_y) * sy
        sub_max_y = vp.min_y + (vp.max_y - vp.min_y) * ey
        return Viewport(sub_min_x, sub_max_x, sub_min_y, sub_max_y, w, h)

    @staticmethod
    def _visible(ti: TileInfo, W: int, H: int) -> bool:
        """
        Checks whether the given tile is visible in the viewport
        :return: True or False
        """
        return ti.x0 < W and ti.y0 < H and ti.w > 0 and ti.h > 0

    def _compute_iteration_variance(self, tile_iters: np.ndarray) -> float:
        """
        Computes the iteration variance for a given tile
        :return: iteration variance
        """
        sample = tile_iters[::self.sample_stride, ::self.sample_stride]
        v = float(np.var(sample))
        return max(0.0, min(1.0, v))

    @staticmethod
    def _compute_boundary_likelihood(tile_iters: np.ndarray) -> float:
        """
        Computes the likelihood of the given tile being near the set boundary
        :return: likelihood [0.0, 1.0]
        """
        total = tile_iters.size
        if total == 0:
            return 0.0
        low, high = 0.3, 0.7
        mid = int(np.count_nonzero((tile_iters >= low) & (tile_iters <= high)))
        return max(0.0, min(1.0, mid / total))
