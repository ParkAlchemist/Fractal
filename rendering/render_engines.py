import os
import time
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed, Future
from abc import ABC, abstractmethod
from typing import Optional, Callable, Tuple, Generator, Set, Any
import numpy as np

from fractals.fractal_base import Fractal, Viewport, RenderSettings
from rendering.tilescheduler import TileScheduler, TileInfo
from rendering.tilescorer import TileScorer


class BaseRenderEngine(ABC):
    """
    Base class for rendering engines.
    """
    def __init__(self, on_tile: Optional[Callable[[int, int, np.ndarray], None]] = None):
        self.on_tile = on_tile

    @abstractmethod
    def render(self, fractal: Fractal, backend, settings: RenderSettings,
               vp: Viewport,
               cancel_cb: Optional[Callable[[], bool]] = None) -> np.ndarray:
        ...

class FullFrameEngine(BaseRenderEngine):
    """
    Rendering engine for full-frame rendering.
    """
    def render(self, fractal: Fractal, backend, settings: RenderSettings,
               vp: Viewport,
               cancel_cb: Optional[Callable[[], bool]] = None) -> np.ndarray:
        return backend.render(fractal, vp, settings)

class TileEngine(BaseRenderEngine):
    """
    Simple tile-based rendering engine. Divides viewport into given sized tiles
    and renders them sequentially.
    """
    def __init__(self, tile_w: int = 256, tile_h: int = 256,
                 on_tile: Optional[Callable[[int, int, np.ndarray], None]] = None):
        super().__init__(on_tile)
        self.tile_w = int(tile_w)
        self.tile_h = int(tile_h)
        self.center_out = True

    def _tile_viewports(self, vp: Viewport) -> Generator[Tuple[int, int, Viewport]]:
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
               vp: Viewport,
               cancel_cb: Optional[Callable[[], bool]] = None) -> np.ndarray:
        canvas = np.zeros((vp.height, vp.width), dtype=settings.precision)
        for x0, y0, sub_vp in self._tile_viewports(vp):
            if cancel_cb is not None and cancel_cb(): break
            tile = backend.render(fractal, sub_vp, settings)
            canvas[y0:y0+sub_vp.height, x0:x0+sub_vp.width] = tile
            if callable(self.on_tile):
                self.on_tile(x0, y0, tile)
        return canvas

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
                 on_tile: Optional[Callable[[int, int, np.ndarray], None]] = None,
                 parallel: bool = True,
                 max_workers: int = 0):
        super().__init__(on_tile)
        self.min_tile = int(min_tile)
        self.max_tile = int(max_tile)
        self.target_ms = float(target_ms)
        self.max_depth = int(max_depth)
        self.sample_stride = int(sample_stride)

        # Concurrency controls
        self.parallel = bool(parallel)
        self.max_workers = int(max_workers)

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
    def render(self, fractal, backend, settings: RenderSettings,
               viewport: Viewport,
               cancel_cb: Optional[Callable[[], bool]] = None) -> np.ndarray:
        W, H = viewport.width, viewport.height
        canvas = np.zeros((H, W), dtype=settings.precision)
        canvas_lock = threading.Lock()

        self.scheduler.update_view(W, H, viewport, self._last_viewport)
        self._last_viewport = viewport

        # Root tiles: choose coarse tiling (e.g., 1x1 or 2x2 of max_tile-ish blocks)
        roots = self._seed_root_tiles(W, H, self.max_tile)

        self.scheduler.clear()
        now = time.perf_counter()
        for x0, y0, w, h, depth in roots:
            ti = TileInfo(x0=x0, y0=y0, w=w, h=h, depth=depth, enqueue_time=now)
            self.scheduler.enqueue(ti, visible=True)

        # Decide concurrency
        backend_name = getattr(backend, "name", "").upper()
        allow_parallel = self.parallel and (not backend_name == "CPU")
        workers = (self.max_workers if self.max_workers > 0 else max(1, (os.cpu_count() or 4))) if allow_parallel else 1

        # Thread pool
        executor: Optional[ThreadPoolExecutor] = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None
        inflight = set() # track futures

        # Local settings
        def make_settings(base: RenderSettings, max_iter: int) -> RenderSettings:
            s = RenderSettings(max_iter=max_iter,
                               samples=base.samples,
                               precision=base.precision)
            return s

        def compute_tile(ti: TileInfo, phase: str = None) -> Tuple[TileInfo, None, None] | \
                                                             Tuple[TileInfo, Any, bool | Any]:
            ti.w = min(ti.w, W - ti.x0)
            ti.h = min(ti.h, H - ti.y0)
            if ti.w <= 0 or ti.h <= 0: return ti, None, None

            sub_vp = self._make_sub_viewport(viewport, ti.x0, ti.y0, ti.w, ti.h)

            st = make_settings(settings, settings.max_iter)

            # Render
            t0 = time.perf_counter()
            if hasattr(backend, "render_async") and allow_parallel:
                tile_iters, done_evt = backend.render_async(fractal, sub_vp, st)
                try:
                    done_evt.synchronize()  # CUDA
                except AttributeError:
                    done_evt.wait()     # OpenCL
            else:
                tile_iters = backend.render(fractal, sub_vp, st)
            t_ms = (time.perf_counter() - t0) * 1000.0

            with canvas_lock:
                canvas[ti.y0: ti.y0 + ti.h, ti.x0: ti.x0 + ti.w] = tile_iters
                if callable(self.on_tile):
                    self.on_tile(ti.x0, ti.y0, tile_iters) # emit

            ti.iteration_variance = self._compute_iteration_variance(tile_iters)
            ti.boundary_likelihood = self._compute_boundary_likelihood(tile_iters)

            # decide wether tile should be split
            split = (ti.depth < self.max_depth) and self._should_split(tile_iters, t_ms, ti.w, ti.h)
            return ti, tile_iters, split

        try:
            while True:

                if cancel_cb is not None and cancel_cb():
                    # The current operation has become obsolete -> return
                    self.scheduler.clear()
                    break

                popped = self.scheduler.pop_next()
                if popped is None and not inflight:
                    # Queue empties, and all tasks done -> return
                    break

                while popped is not None:

                    phase, ti = popped
                    if executor is None:
                        # Sequential path
                        ti, _, split = compute_tile(ti, phase)
                        if split:
                            for child in self._split((ti.x0, ti.y0, ti.w, ti.h, ti.depth), W, H):
                                cti = TileInfo(x0=child[0], y0=child[1],
                                               w=child[2], h=child[3],
                                               depth=child[4],
                                               enqueue_time=time.perf_counter())
                                self.scheduler.enqueue(cti, visible=self._visible(cti, W, H))
                    else:
                        # Parallel path
                        fut = executor.submit(compute_tile, ti, phase)
                        inflight.add(fut)
                    popped = self.scheduler.pop_next()

                if inflight:
                    done, _ = self._wait_some(inflight, timeout=0.05)
                    for fut in done:
                        inflight.discard(fut)
                        try:
                            ti, _, split = fut.result()
                        except Exception as e:
                            print(f"Failed to compute tile: {e}")
                            continue
                        if split:
                            for child in self._split((ti.x0, ti.y0, ti.w, ti.h, ti.depth), W, H):
                                cti = TileInfo(x0=child[0], y0=child[1],
                                               w=child[2], h=child[3],
                                               depth=child[4],
                                               enqueue_time=time.perf_counter())
                                self.scheduler.enqueue(cti, visible=self._visible(cti, W, H))
        except Exception as e:
            print(f"Error in render loop: {e}")
        finally:
            if executor is not None:
                executor.shutdown(wait=False)
        return canvas


    # ------- Helpers -------------------------------------
    @staticmethod
    def _wait_some(inflight, timeout: float) -> Tuple[Set[Future], Set[Future]]:
        """
        Wait for at least one future to complete
        """
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
        Check wether given tile should be split or not
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
    def _make_sub_viewport(vp: Viewport, x0: int, y0: int,
                           w: int, h: int) -> Viewport:
        """
        Creates a subviewport on given coordinates from given viewport
        :return: subviewport
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
        Checks wether given tile is visible in the viewport
        :return: True or False
        """
        return ti.x0 < W and ti.y0 < H and ti.w > 0 and ti.h > 0

    def _compute_iteration_variance(self, tile_iters: np.ndarray) -> float:
        """
        Computes the iteration variance for given tile
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
        mid = np.count_nonzero((tile_iters >= low) & (tile_iters <= high))
        return max(0.0, min(1.0, mid / total))
