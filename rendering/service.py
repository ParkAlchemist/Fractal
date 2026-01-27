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
from coloring.base import ColoringStrategy

# Rendering imports
from rendering.core import Renderer
from rendering.engines.full_frame import FullFrameEngine
from rendering.engines.tile import TileEngine
from rendering.engines.adaptive_tile import AdaptiveTileEngine
from rendering.executor import RenderExecutor
from rendering.events import FrameEvent, TileEvent, LogEvent

# Utils imports
from utils.enums import EngineMode, ColoringMode, PrecisionMode, BackendType


class RenderService:
    """
    UI-facing facade that owns:
      - configuration (palettes, precision, image size),
      - lifecycle (start/stop),
      - event dispatch (frame/tile/log),
      - worker threads and cancellation.

    It composes a Renderer (engine + backend manager) and keeps the UI layer
    decoupled from device/runtime details.
    """

    def __init__(
        self,
        width: int,
        height: int,
        palette,
        kernel: BackendType = BackendType.AUTO,
        max_iter: int = 200,
        samples: int = 2,
        coloring_mode: ColoringMode = ColoringMode.EXTERIOR,
        precision: np.dtype = np.float32
    ) -> None:
        # ----- Image & Compute config -----
        self.width = int(width)
        self.height = int(height)
        self.kernel = kernel
        self.max_iter = int(max_iter)
        self.samples = int(samples)
        self.precision = precision

        # ----- Coloring config -----
        self.exter_palette = np.array(palette, dtype=np.uint8)
        self.inter_palette = self.exter_palette
        self.inter_color = (100, 100, 100)
        self.coloring_mode = coloring_mode
        self.coloring: ColoringStrategy = SmoothEscapeColoring()

        # ----- Fractal & Settings -----
        self.fractal = MandelbrotFractal()
        self.settings = RenderSettings(max_iter=self.max_iter,
                                       samples=self.samples,
                                       precision=self.precision)

        # ----- Rendering facade -----
        self.renderer = Renderer(self.fractal,
                                 self.settings,
                                 engine=FullFrameEngine(),
                                 executor=RenderExecutor())
        self.engine_mode = EngineMode.FULL_FRAME
        self.tile_w = 256
        self.tile_h = 256

        # ----- Viewport & threading -----
        self._viewport: Optional[Tuple[float, float, float, float]] = None
        self._worker_thread: Optional[threading.Thread] = None
        self._stop_flag = threading.Event()
        self._render_seq = 0
        self._start_time: Optional[float] = None
        self._lock = threading.Lock()

        self._base_canvas: Optional[np.ndarray] = None

        # Callbacks
        self.on_frame: Optional[Callable[[FrameEvent], None]] = None
        self.on_tile: Optional[Callable[[TileEvent], None]] = None
        self.on_log: Optional[Callable[[LogEvent], None]] = None

    # ---------------------------------------------------------------------
    # Configuration
    # ---------------------------------------------------------------------

    def set_palettes(self, exterior: str, interior: str) -> None:
        self.exter_palette = np.array(palettes[exterior], dtype=np.uint8)
        self.inter_palette = np.array(palettes[interior], dtype=np.uint8)

    def set_coloring_mode(self, mode: ColoringMode) -> None:
        self.coloring_mode = mode

    def set_coloring_strategy(self, strategy: ColoringStrategy) -> None:
        self.coloring = strategy

    def set_precision(self, precision: PrecisionMode) -> None:
        if precision == PrecisionMode.Double:
            self.precision = np.float64
        elif precision == PrecisionMode.Single:
            self.precision = np.float32
        self.settings.precision = self.precision
        self.renderer.executor.compile(self.fractal, self.settings)

    def set_image_size(self, width: int, height: int) -> None:
        self.width, self.height = int(width), int(height)

    def set_max_iter(self, new_max: int) -> None:
        self.max_iter = new_max
        self.settings.max_iter = new_max
        self.renderer.executor.compile(self.fractal, self.settings)

    def set_samples(self, new_samples: int) -> None:
        self.samples = new_samples
        self.settings.samples = new_samples
        self.renderer.executor.compile(self.fractal, self.settings)

    def set_kernel(self, backend: BackendType) -> None:
        self.kernel = backend
        self.renderer.default_backend = self.kernel.name

    def set_engine_mode(
        self,
        mode: EngineMode,
        adaptive_opts: Optional[Dict] = None,
        tile_w: int | None = None,
        tile_h: int | None = None
    ) -> None:
        self.engine_mode = mode
        if tile_w:
            self.tile_w = int(tile_w)
        if tile_h:
            self.tile_h = int(tile_h)

        if mode == EngineMode.FULL_FRAME:
            self.renderer.set_engine(FullFrameEngine())
        elif mode == EngineMode.TILED:
            self.renderer.set_engine(TileEngine(tile_w=self.tile_w, tile_h=self.tile_h))
        elif mode == EngineMode.ADAPTIVE:
            opts = adaptive_opts or {}
            self.renderer.set_engine(
                AdaptiveTileEngine(
                    min_tile=opts.get("min_tile", 64),
                    max_tile=opts.get("max_tile", self.tile_w),
                    target_ms=opts.get("target_ms", 12.0),
                    max_depth=opts.get("max_depth", 4),
                    sample_stride=opts.get("sample_stride", 8),
                    on_tile=None,
                )
            )


    def set_view(self, center_x: float, center_y: float, zoom: float) -> None:
        scale = 1.0 / zoom
        min_x = center_x - (self.width / 2) * scale
        min_y = center_y - (self.height / 2) * scale
        max_x = center_x + (self.width / 2) * scale
        max_y = center_y + (self.height / 2) * scale
        self._viewport = (min_x, max_x, min_y, max_y)

    def set_base_canvas(self, base: Optional[np.ndarray]) -> None:
        self._base_canvas = None if base is None else np.array(base, copy=True)

    # ---------------------------------------------------------------------
    # Lifecycle
    # ---------------------------------------------------------------------

    def start_render(self) -> None:
        if self._viewport is None:
            self.set_view(center_x=-0.5, center_y=0.0, zoom=250)

        self.stop()
        self._render_seq += 1
        self._start_time = time.time()
        self._stop_flag.clear()

        if self.engine_mode == EngineMode.FULL_FRAME:
            target = self._run_full_frame
        else:
            target = self._run_tiled

        self._worker_thread = threading.Thread(target=target, daemon=True)
        self._worker_thread.start()

    def stop(self) -> None:
        self._stop_flag.set()
        if self._worker_thread and self._worker_thread.is_alive():
            self._worker_thread.join(timeout=1.0)
        self._worker_thread = None

    def shutdown(self) -> None:
        """Stop and release backend resources."""
        try:
            self.stop()
        finally:
            self.renderer.executor.close()

    # ---------------------------------------------------------------------
    # Worker routines
    # ---------------------------------------------------------------------

    def _run_full_frame(self) -> None:
        min_x, max_x, min_y, max_y = self._viewport
        vp = Viewport(min_x=min_x, max_x=max_x,
                      min_y=min_y, max_y=max_y,
                      width=self.width, height=self.height)
        try:
            iter_canvas = self.renderer.render(vp)
            rgb_img = self.coloring.apply(
                iter_canvas.astype(self.precision),
                self.coloring_mode,
                self.exter_palette,
                self.inter_palette,
                interior_color=self.inter_color)

            if self.on_frame:
                evt = FrameEvent(rgb_img, int(vp.width), int(vp.height), self._render_seq)
                self.on_frame(evt)
            if self.on_log and self._start_time is not None:
                elapsed = round(time.time() - self._start_time, 3)
                self.on_log(LogEvent(f"Render time: {elapsed}s", level=None))
        except Exception as e:
            if self.on_log:
                self.on_log(LogEvent(f"[RenderService] Full-frame error: {e}", level="error"))

    def _run_tiled(self) -> None:
        min_x, max_x, min_y, max_y = self._viewport
        vp = Viewport(min_x=min_x, max_x=max_x,
                      min_y=min_y, max_y=max_y,
                      width=self.width, height=self.height)
        engine = self.renderer.engine
        executor = self.renderer.executor
        settings = self.renderer.settings
        seq = self._render_seq

        if (self._base_canvas is not None and
            self._base_canvas.shape == (vp.height, vp.width) and
            self._base_canvas.dtype == settings.precision):
            canvas = self._base_canvas.copy()
        else:
            canvas = np.zeros((vp.height, vp.width), dtype=settings.precision)

        def cancel_cb() -> bool:
            return self._stop_flag.is_set() or (seq != self._render_seq)

        def on_tile(x0, y0, tile_iter) -> None:
            if cancel_cb():
                return
            h, w = tile_iter.shape
            with self._lock:
                canvas[y0:y0 + h, x0:x0 + w] = tile_iter
            if self.on_tile:
                evt = TileEvent(x0, y0, h, w,
                              tile_iter, seq,
                              int(vp.width), int(vp.height)
                              )
                self.on_tile(evt)

        engine.on_tile = on_tile

        try:
            engine.render(self.fractal, executor, settings, vp, cancel_cb=cancel_cb)
        except Exception as e:
            if self.on_log:
                evt = LogEvent(f"[RenderService] Tiled render error: {e}", level="error")
                self.on_log(evt)
            return

        if not cancel_cb():
            rgb_img = self.coloring.apply(
                canvas.astype(self.precision),
                self.coloring_mode,
                self.exter_palette, self.inter_palette,
                interior_color=self.inter_color
            )
            if self.on_frame:
                evt = FrameEvent(rgb_img, int(vp.width), int(vp.height), seq)
                self.on_frame(evt)
            if self.on_log and self._start_time is not None:
                elapsed = round(time.time() - self._start_time, 3)
                evt = LogEvent(f"Render time: {elapsed}s", level=None)
                self.on_log(evt)
