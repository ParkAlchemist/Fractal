from __future__ import annotations
import threading
import time
import numpy as np
from typing import Callable, Optional, Tuple, Dict

# Fractal imports
from fractals.base import Viewport, RenderSettings
from fractals.mandelbrot import MandelbrotFractal

# Coloring imports
from coloring.palettes import palettes
from coloring.smooth_escape import SmoothEscapeColoring

# Rendering imports
from rendering.core import Renderer
from rendering.engines.full_frame import FullFrameEngine
from rendering.engines.tile import TileEngine
from rendering.engines.adaptive_tile import AdaptiveTileEngine
from rendering.events import FrameEvent, TileEvent, LogEvent

# Utils imports
from utils.enums import EngineMode, ColoringMode, PrecisionMode, BackendType
from utils.backend_helpers import available_backends

# Backend imports
from backend.model.base import Backend
from backend.model.opencl import OpenClBackend
from backend.model.cuda import CudaBackend
from backend.model.cpu import CpuBackend


class RenderService:
    def __init__(self, width: int, height: int, palette,
                 kernel: BackendType = BackendType.AUTO, max_iter: int = 200, samples: int = 2,
                 coloring_mode: ColoringMode = ColoringMode.EXTERIOR, precision: np.dtype = np.float32
                 ) -> None:
        self.width = width
        self.height = height
        self.kernel = kernel
        self.max_iter = max_iter
        self.samples = samples
        self.precision = precision

        # Palettes as numpy arrays
        self.exter_palette = np.array(palette, dtype=np.uint8)
        self.inter_palette = self.exter_palette
        self.inter_color = (100, 100, 100)
        self.coloring_mode = coloring_mode
        self.coloring = SmoothEscapeColoring()

        self.fractal = MandelbrotFractal()
        self.backend = self._select_backend(kernel)
        self.settings = RenderSettings(max_iter=max_iter, samples=samples, precision=precision)
        self.renderer = Renderer(self.fractal, self.settings, engine=FullFrameEngine())

        self.engine_mode = EngineMode.FULL_FRAME
        self.tile_w = 256
        self.tile_h = 256

        self._viewport: Optional[Tuple[float, float, float, float]] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._render_seq = 0
        self._start_time: Optional[float] = None

        # Callbacks
        self.on_frame: Optional[Callable[[FrameEvent], None]] = None
        self.on_tile: Optional[Callable[[TileEvent], None]] = None
        self.on_log: Optional[Callable[[LogEvent], None]] = None

    @staticmethod
    def _select_backend(kernel: BackendType) -> Backend:
        if kernel == BackendType.OPENCL: return OpenClBackend()
        if kernel == BackendType.CUDA:   return CudaBackend()
        if kernel == BackendType.CPU:    return CpuBackend()
        backs = available_backends()
        if BackendType.OPENCL.name in backs: return OpenClBackend()
        if BackendType.CUDA.name   in backs: return CudaBackend()
        return CpuBackend()

    # -------------- Configuration --------------------------
    def set_palettes(self, exterior: str, interior: str) -> None:
        self.exter_palette = np.array(palettes[exterior], dtype=np.uint8)
        self.inter_palette = np.array(palettes[interior], dtype=np.uint8)

    def set_coloring_mode(self, mode: ColoringMode):
        self.coloring_mode = mode

    def set_precision(self, precision: PrecisionMode) -> None:
        if precision == PrecisionMode.Double:
            self.precision = np.float64
        elif precision == PrecisionMode.Single:
            self.precision = np.float32
        self.settings.precision = self.precision
        self.renderer.settings = self.settings

    def set_image_size(self, width: int, height: int) -> None:
        self.width, self.height = int(width), int(height)

    def set_max_iter(self, new_max: int) -> None:
        self.max_iter = new_max
        self.settings.max_iter = new_max
        self.renderer.settings = self.settings

    def set_samples(self, new_samples: int) -> None:
        self.samples = new_samples
        self.settings.samples = new_samples
        self.renderer.settings = self.settings

    def set_engine_mode(self, mode: EngineMode, adaptive_opts: Optional[Dict] = None,
                        tile_w: int | None = None,
                        tile_h: int | None = None) -> None:
        self.engine_mode = mode
        if tile_w: self.tile_w = tile_w
        if tile_h: self.tile_h = tile_h

        if mode == EngineMode.FULL_FRAME:
            self.renderer.set_engine(FullFrameEngine())
        elif mode == EngineMode.TILED:
            self.renderer.set_engine(TileEngine(tile_w=self.tile_w, tile_h=self.tile_h))
        elif mode == EngineMode.ADAPTIVE and adaptive_opts:
            self.renderer.set_engine(AdaptiveTileEngine(
                min_tile=adaptive_opts.get("min_tile", 64),
                max_tile=adaptive_opts.get("max_tile", self.tile_w),
                target_ms=adaptive_opts.get("target_ms", 12.0),
                max_depth=adaptive_opts.get("max_depth", 4),
                sample_stride=adaptive_opts.get("sample_stride", 8),
                on_tile=None))


    def set_view(self, center_x: float, center_y: float, zoom: float) -> None:
        scale = 1.0 / zoom
        min_x = center_x - (self.width / 2) * scale
        min_y = center_y - (self.height / 2) * scale
        max_x = center_x + (self.width / 2) * scale
        max_y = center_y + (self.height / 2) * scale
        self._viewport = (min_x, max_x, min_y, max_y)

    # ------------- Lifecycle ------------------------
    def start_render(self) -> None:
        if self._viewport is None:
            self.set_view(center_x=-0.5, center_y=0.0, zoom=250)
        self.stop()
        self._render_seq += 1
        self._start_time = time.time()
        self._stop_flag.clear()

        if self.engine_mode == EngineMode.FULL_FRAME:
            self._worker_thread = threading.Thread(target=self._run_full_frame, daemon=True)
        else:
            self._worker_thread = threading.Thread(target=self._run_tiled, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        self._worker_thread = None

    # ----------- Worker routines -----------------------
    def _run_full_frame(self) -> None:
        min_x, max_x, min_y, max_y = self._viewport
        vp = Viewport(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, width=self.width, height=self.height)
        self._iter_canvas = self.renderer.render(vp)
        rgb_img = self.coloring.apply(self._iter_canvas.astype(self.precision),
                                      self.coloring_mode,
                                      self.exter_palette, self.inter_palette,
                                      interior_color=self.inter_color)
        if self.on_frame:
            evt = FrameEvent(rgb_img, int(vp.width), int(vp.height), self._render_seq)
            self.on_frame(evt)
        if self.on_log:
            elapsed = round(time.time() - self._start_time, 3)
            evt = LogEvent(f"Render time: {elapsed}s", level=None)
            self.on_log(evt)

    def _run_tiled(self) -> None:
        min_x, max_x, min_y, max_y = self._viewport
        vp = Viewport(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, width=self.width, height=self.height)
        engine = self.renderer.engine
        manager = self.renderer.manager
        settings = self.renderer.settings
        seq = self._render_seq

        canvas = np.zeros((vp.height, vp.width), dtype=self.precision)  # Iteration canvas

        def cancel_cb() -> bool:
            return self._stop_flag.is_set()

        def on_tile(x0, y0, tile_iter) -> None:
            if cancel_cb(): return
            h, w = tile_iter.shape
            canvas[y0:y0+h, x0:x0+w] = tile_iter
            if self.on_tile:
                evt = TileEvent(x0, y0, h, w, tile_iter, seq, int(vp.width), int(vp.height))
                self.on_tile(evt)

        engine.on_tile = on_tile
        engine.render(self.fractal, manager, settings, vp, cancel_cb=cancel_cb)

        if not cancel_cb():
            rgb_img = self.coloring.apply(canvas.astype(self.precision),
                                          self.coloring_mode,
                                          self.exter_palette, self.inter_palette,
                                          interior_color=self.inter_color)
            if self.on_frame:
                evt = FrameEvent(rgb_img, int(vp.width), int(vp.height), seq)
                self.on_frame(evt)
            if self.on_log:
                elapsed = round(time.time() - self._start_time, 3)
                evt = LogEvent(f"Render time: {elapsed}s", level=None)
                self.on_log(evt)
