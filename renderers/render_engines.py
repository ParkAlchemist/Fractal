import os
import time
import math
import heapq
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any, Callable
import numpy as np

from fractals.fractal_base import Fractal, Viewport, RenderSettings

class BaseRenderEngine(ABC):
    def __init__(self, on_tile: Optional[Callable[[int, int, np.ndarray], None]] = None):
        self.on_tile = on_tile

    @abstractmethod
    def render(self, fractal: Fractal, backend, settings: RenderSettings,
               vp: Viewport, reference: Optional[Dict[str, Any]] = None,
               cancel_cb: Optional[Callable[[], bool]] = None) -> np.ndarray:
        ...

class FullFrameEngine(BaseRenderEngine):
    def render(self, fractal: Fractal, backend, settings: RenderSettings,
               vp: Viewport, reference: Optional[Dict[str, Any]] = None,
               cancel_cb: Optional[Callable[[], bool]] = None) -> np.ndarray:
        return backend.render(fractal, vp, settings, reference)

class TileEngine(BaseRenderEngine):
    def __init__(self, tile_w: int = 256, tile_h: int = 256,
                 per_tile_reference: bool = True,
                 on_tile: Optional[Callable[[int, int, np.ndarray], None]] = None):
        super().__init__(on_tile)
        self.tile_w = int(tile_w)
        self.tile_h = int(tile_h)
        self.per_tile_reference = per_tile_reference
        self.center_out = True

    def _tile_viewports(self, vp: Viewport):
        W, H = vp.width, vp.height
        cx, cy = W / 2.0, H / 2.0
        tiles = []
        for y0 in range(0, H, self.tile_h):
            for x0 in range(0, W, self.tile_w):
                tw = min(self.tile_w, W - x0)
                th = min(self.tile_h, H - y0)
                txc, tyc = x0 + tw/2.0, y0 + th/2.0
                dist2 = (txc - cx)**2 + (tyc - cy)**2
                sx, ex = x0 / W, (x0 + tw) / W
                sy, ey = y0 / H, (y0 + th) / H
                sub_min_x = vp.min_x + (vp.max_x - vp.min_x) * sx
                sub_max_x = vp.min_x + (vp.max_x - vp.min_x) * ex
                sub_min_y = vp.min_y + (vp.max_y - vp.min_y) * sy
                sub_max_y = vp.min_y + (vp.max_y - vp.min_y) * ey
                sub_vp = Viewport(sub_min_x, sub_max_x, sub_min_y, sub_max_y, tw, th)
                tiles.append((x0, y0, sub_vp, dist2))
        if self.center_out:
            tiles.sort(key=lambda t: t[3])
        for x0, y0, sub_vp, _ in tiles:
            yield x0, y0, sub_vp

    def render(self, fractal: Fractal, backend, settings: RenderSettings,
               vp: Viewport, reference: Optional[Dict[str, Any]] = None,
               cancel_cb: Optional[Callable[[], bool]] = None) -> np.ndarray:
        canvas = np.zeros((vp.height, vp.width), dtype=settings.precision)
        for x0, y0, sub_vp in self._tile_viewports(vp):
            if cancel_cb is not None and cancel_cb(): break
            ref = None
            if settings.use_perturb and self.per_tile_reference and fractal.supports_perturbation():
                ref = fractal.build_reference(sub_vp, settings)
            tile = backend.render(fractal, sub_vp, settings, ref)
            canvas[y0:y0+sub_vp.height, x0:x0+sub_vp.width] = tile
            if self.on_tile is not None:
                self.on_tile(x0, y0, tile)
        return canvas

class AdaptiveTileEngine(BaseRenderEngine):
    """
    Dynamic tiling via quadtree + priority queue.
    - Starts with coarse tiles (max_tile).
    - Measures cost & variance, then splits expensive tiles (2x2).
    - Prioritizes center-first by default.
    - Can run sequentially or with a thread pool for parallel speed.

    Signals:
      on_tile(x0, y0, tile_iters) is called whenever a tile finishes.
    """

    def __init__(self, min_tile: int = 64, max_tile: int = 512,
                 target_ms: float = 12.0, max_depth: int = 4,
                 priority="center-first", sample_stride: int = 8,
                 on_tile: Optional[Callable[[int, int, np.ndarray], None]] = None,
                 parallel: bool = True, max_workers: int = 0,
                 parallel_for_cpu_only: bool = True,
                 per_tile_reference: bool = False):
        super().__init__(on_tile)
        self.min_tile = min_tile
        self.max_tile = max_tile
        self.target_ms = target_ms
        self.max_depth = max_depth
        self.priority = priority
        self.sample_stride = sample_stride

        # Concurrency controls
        self.parallel = parallel
        self.per_tile_reference = per_tile_reference
        self.max_workers = max_workers
        self.parallel_for_cpu_only = parallel_for_cpu_only

    # ----------- Render entry point -----------------------------------------
    def render(self, fractal, backend, settings: RenderSettings,
               viewport: Viewport, reference: Optional[Dict[str, Any]] = None,
               cancel_cb: Optional[Callable[[], bool]] = None) -> np.ndarray:
        W, H = viewport.width, viewport.height
        canvas = np.zeros((H, W), dtype=settings.precision)
        canvas_lock = threading.Lock()

        # Root tiles: choose a coarse tiling (e.g., 1x1 or 2x2 of max_tile-ish blocks)
        roots = self._seed_root_tiles(W, H, self.max_tile)

        # Priority queue of tiles: (-score, tile_index, tile_info)
        pq = []
        for idx, t in enumerate(roots):
            score = self._tile_priority(t, W, H)
            heapq.heappush(pq, (-score, idx, t))

        # Decide concurrency
        backend_name = getattr(backend, "name", "").upper()
        allow_parallel = self.parallel and (not self.parallel_for_cpu_only or backend_name == "CPU")
        workers = (self.max_workers if self.max_workers > 0 else max(1, (os.cpu_count() or 4))) if allow_parallel else 1

        # Thread pool
        executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        inflight = set() # track futures

        try:
            while pq or inflight:
                if cancel_cb is not None and cancel_cb(): break

                while pq and (executor is None or len(inflight) < workers):
                    _, _, tile = heapq.heappop(pq)
                    x0, y0, w, h, depth = tile

                    # Clamp to image bounds
                    w = min(w, W - x0)
                    h = min(h, H - y0)
                    if w <= 0 or h <= 0:
                        continue

                    sub_vp = self._make_sub_viewport(viewport, x0, y0, w, h)

                    # Pick reference
                    ref = None
                    if settings.use_perturb and fractal.supports_perturbation():
                        if self.per_tile_reference:
                            # Accurate: per-tile reference
                            ref = fractal.build_reference(sub_vp, settings)
                        else:
                            # Fast: reuse frame level reference
                            ref = reference

                    def _compute_tile(fr=fractal, be=backend, st=settings, sv=sub_vp, X=x0, Y=y0, Wt=w, Ht=h, D=depth, Re=ref):
                        t0 = time.perf_counter()
                        tile_iters = be.render(fr, sv, st, Re)
                        t_ms = (time.perf_counter() - t0) * 1000.0

                        with canvas_lock:
                            canvas[Y:Y+Ht, X:X+Wt] = tile_iters
                        # Emit progressive tile
                        if callable(self.on_tile):
                            self.on_tile(X, Y, tile_iters)

                        # Decide refinement
                        split = (D < self.max_depth) and self._should_split(tile_iters, t_ms, Wt, Ht)

                        return X, Y, Wt, Ht, D, split

                    if executor is None:
                        # Sequential
                        X, Y, Wt, Ht, D, split = _compute_tile()
                        if split:
                            for child in self._split((X, Y, Wt, Ht, D), W, H):
                                score2 = self._tile_priority(child, W, H)
                                heapq.heappush(pq, (-score2, id(child), child))
                    else:
                        fut = executor.submit(_compute_tile)
                        inflight.add(fut)

                if inflight:
                    done, _ = self._wait_some(inflight, timeout=0.05)
                    for fut in done:
                        inflight.discard(fut)
                        try:
                            X, Y, Wt, Ht, D, split = fut.result()
                        except Exception as e:
                            print("Failed to compute tile:", e)
                            continue
                        if split:
                            for child in self._split((X, Y, Wt, Ht, D), W, H):
                                score2 = self._tile_priority(child, W, H)
                                heapq.heappush(pq, (-score2, id(child), child))
                else:
                    pass
        finally:
            if executor is not None:
                executor.shutdown(wait=False)
            return canvas

    # ------- Helpers -------------------------------------
    @staticmethod
    def _wait_some(inflight, timeout: float):
        """Wait for at least one future to complete"""
        done = set()
        if not inflight:
            return done, inflight
        try:
            for fut in as_completed(inflight, timeout=timeout):
                done.add(fut)
                break
        except Exception:
            pass
        return done, inflight

    def _seed_root_tiles(self, W, H, max_tile):
        # Cover the viewport with large blocks aligned to multiples of 32
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
        # Round down to nearest multiple of 32
        return max((n // 32) * 32, 32)

    @staticmethod
    def _tile_priority(tile, W, H):
        x0, y0, w, h, depth = tile
        # Center first by default
        cx, cy = W / 2.0, H / 2.0
        tx = x0 + w / 2.0
        ty = y0 + h / 2.0
        dist = math.hypot(tx - cx, ty - cy)
        base = 1.0 / (1.0 + dist)
        # Prefer larger tiles
        size_bonus = (w * h) / (W * H)
        # Shallower depth first
        depth_bonus = 1.0 / (1.0 + depth)
        return base * 0.7 + size_bonus * 0.2 + depth_bonus * 0.1

    def _should_split(self, tile_iters, t_ms, w, h):
        if w <= self.min_tile and h <= self.min_tile:
            return False

        # Time budget heuristic
        time_heavy = t_ms > self.target_ms
        # Variance heuristic
        stride = self.sample_stride
        sample = tile_iters[::stride, ::stride]
        var = float(np.var(sample))
        # Combine
        return time_heavy or var > 0.5

    def _split(self, tile, W, H):
        x0, y0, w, h, depth = tile
        w2 = max(self.min_tile, self._align_32(w // 2))
        h2 = max(self.min_tile, self._align_32(h // 2))
        # 2x2 quad children
        children = [
            (x0, y0, w2, h2, depth + 1),
            (x0 + w2, y0, w - w2, h2, depth + 1),
            (x0, y0 + h2, w2, h - h2, depth + 1),
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
    def _make_sub_viewport(vp: Viewport, x0: int, y0: int, w: int, h: int) -> Viewport:
        W, H = vp.width, vp.height
        sx, ex = x0 / W, (x0 + w) / W
        sy, ey = y0 / H, (y0 + h) / H
        sub_min_x = vp.min_x + (vp.max_x - vp.min_x) * sx
        sub_max_x = vp.min_x + (vp.max_x - vp.min_x) * ex
        sub_min_y = vp.min_y + (vp.max_y - vp.min_y) * sy
        sub_max_y = vp.min_y + (vp.max_y - vp.min_y) * ey
        return Viewport(sub_min_x, sub_max_x, sub_min_y, sub_max_y, w, h)
