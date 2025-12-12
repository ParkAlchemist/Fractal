import threading

import numpy as np
import time
from PySide6.QtCore import Signal, QObject, QThread
from PySide6.QtGui import QImage

from utils.enums import Kernel, ColoringMode, EngineMode, Precisions
from coloring.palettes import palettes
from utils.utils import available_backends, ndarray_to_qimage
from backends.opencl_backend import OpenClBackend
from backends.cpu_backend import CpuBackend
from backends.cuda_backend import CudaBackend
from coloring.smooth_escape import SmoothEscapeColoring
from fractals.fractal_base import Viewport, RenderSettings
from fractals.mandelbrot import MandelbrotFractal
from rendering.renderer_core import Renderer
from rendering.render_engines import FullFrameEngine, TileEngine, AdaptiveTileEngine

# --- Workers ---------------------------------------------------------------

class FullImageRenderWorker(QThread):
    iter_calc_done = Signal(np.ndarray)

    def __init__(self, renderer: Renderer, viewport: Viewport):
        super().__init__()
        self.renderer = renderer
        self.viewport = viewport
        self._running = True

    def run(self):
        if not self._running:
            return
        data = self.renderer.render(self.viewport)
        if not data.flags["C_CONTIGUOUS"]:
            data = data.copy()
        self.iter_calc_done.emit(data)

    def stop(self):
        self._running = False


class ProgressiveTileRenderWorker(QThread):
    tile_done = Signal(int, int, int, int, np.ndarray, int)
    finished_frame = Signal(np.ndarray, int)

    def __init__(self, renderer: Renderer, viewport: Viewport, seq: int):
        super().__init__()
        self.renderer = renderer
        self.viewport = viewport
        self.seq = seq
        self._stop = False
        self.tile_lock = threading.Lock()

    def stop(self):
        self._stop = True

    def run(self):
        engine = self.renderer.engine
        backend = self.renderer.backend
        fractal = self.renderer.fractal
        settings = self.renderer.settings

        canvas = np.zeros((self.viewport.height, self.viewport.width), dtype=settings.precision)

        def cancel_cb():
            return self._stop

        def on_tile(x0, y0, tile):
            with self.tile_lock:
                h, w = tile.shape
                canvas[y0:y0 + h, x0:x0 + w] = tile
                self.tile_done.emit(x0, y0, w, h, tile, self.seq)

        engine.on_tile = on_tile
        engine.render(fractal, backend, settings, self.viewport, cancel_cb=cancel_cb)

        if not self._stop:
            self.finished_frame.emit(canvas, self.seq)


# --- Renderer facade -------------------------------------------------------

class FullImageRenderer(QObject):
    image_updated = Signal(QImage)                 # final full frame / full-frame mode
    tile_ready = Signal(int, int, int, QImage)   # (gen_id, x, y, tile QImage)
    log_text = Signal(str)

    def __init__(self, width, height, palette, kernel=Kernel.AUTO,
                 max_iter=200, samples=2, coloring_mode=ColoringMode.EXTERIOR,
                 precision=np.float32):
        super().__init__()
        self.width = width
        self.height = height

        self.kernel = kernel
        self.max_iter = max_iter
        self.samples = samples
        self.precision = precision

        self.exter_palette = np.array(palette, dtype=np.uint8)
        self.inter_palette = self.exter_palette
        self.inter_color = (100, 100, 100)

        self.coloring_mode = coloring_mode
        self.coloring = SmoothEscapeColoring()

        self._iter_buf = None
        self._viewport = None

        self.engine_mode = EngineMode.FULL_FRAME
        self.tile_w = 256
        self.tile_h = 256

        self.fractal = MandelbrotFractal()
        self.backend = self._select_backend(kernel)
        self.settings = RenderSettings(max_iter=max_iter, samples=samples,
                                       precision=self.precision)
        self.renderer = Renderer(self.fractal, self.backend, self.settings, engine=FullFrameEngine())

        self._worker = None
        self._iter_canvas = None
        self._render_seq = 0
        self._start_time = None
        self._stop_flag = False

    # -- Backend selection
    @staticmethod
    def _select_backend(kernel):
        if kernel == Kernel.OPENCL: return OpenClBackend()
        if kernel == Kernel.CUDA:   return CudaBackend()
        if kernel == Kernel.CPU:    return CpuBackend()
        backs = available_backends()
        if Kernel.OPENCL.name in backs: return OpenClBackend()
        if Kernel.CUDA.name   in backs: return CudaBackend()
        return CpuBackend()

    # -- Render entry point
    def render_frame(self, min_x, max_x, min_y, max_y):
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()

        # If viewport unchanged and we have a buffer, just re-color
        if self._viewport == (min_x, max_x, min_y, max_y) and self._iter_buf is not None:
            self.render_image()
            return

        self._start_time = time.time()
        self._viewport = (min_x, max_x, min_y, max_y)

        vp = Viewport(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
                      width=self.width, height=self.height)

        if self.engine_mode == EngineMode.FULL_FRAME:
            self.renderer.set_engine(FullFrameEngine())
            self._worker = FullImageRenderWorker(self.renderer, vp)
            self._worker.iter_calc_done.connect(self._set_iter_buf)
            self._worker.start()

        elif self.engine_mode == EngineMode.TILED:
            self._iter_canvas = np.zeros((self.height, self.width), dtype=self.precision)
            self._render_seq += 1
            self._worker = ProgressiveTileRenderWorker(self.renderer, vp, self._render_seq)
            self._worker.tile_done.connect(self._on_tile_done)
            self._worker.finished_frame.connect(self._on_frame_finished)
            self._worker.start()

    # -- Callbacks
    def _set_iter_buf(self, data: np.ndarray):
        self._iter_buf = data
        self.log_text.emit(f"Render time: {round(time.time() - self._start_time, 3)}s")
        self.render_image()

    def render_image(self):
        rgb_img = self._apply_palette()
        qimage = ndarray_to_qimage(rgb_img)
        # Deep copy before emitting across threads (prevents dangling buffers)
        self.image_updated.emit(qimage.copy())

    def _apply_palette(self):
        return self.coloring.apply(self._iter_buf.astype(np.float64), self.coloring_mode,
                                   self.exter_palette, self.inter_palette,
                                   interior_color=self.inter_color)

    # Progressive tile handlers
    def _on_tile_done(self, x0, y0, w, h, tile_iter, seq):
        if seq != self._render_seq:
            return
        self._iter_canvas[y0:y0 + h, x0:x0 + w] = tile_iter.astype(self.precision)

        # Colorize just the tile and emit it; deep copy for thread safety
        tile_rgb  = self.coloring.apply(tile_iter.astype(self.precision), self.coloring_mode,
                                        self.exter_palette, self.inter_palette,
                                        interior_color=self.inter_color)
        tile_qimg = ndarray_to_qimage(tile_rgb).copy()
        self.tile_ready.emit(self._render_seq, x0, y0, tile_qimg)

    def _on_frame_finished(self, full_iter, seq):
        if seq != self._render_seq:
            return
        self._iter_buf = full_iter.astype(self.precision)
        self.log_text.emit(f"Render time: {round(time.time() - self._start_time, 3)}s")
        self.render_image()

    # Stop/cancel
    def stop(self):
        self._stop_flag = True
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()

    # --- Setters (invalidate running renders) ----------------------------
    def set_kernel(self, new_kernel):
        self.kernel = new_kernel
        self._iter_buf = None
        self.backend = self._select_backend(new_kernel)
        self.renderer = Renderer(self.fractal, self.backend, self.settings, engine=self.renderer.engine)
        self.stop()

    def set_exter_palette(self, name):
        self.exter_palette = np.array(palettes[name], dtype=np.uint8)

    def set_inter_palette(self, name):
        self.inter_palette = np.array(palettes[name], dtype=np.uint8)

    def set_coloring_mode(self, mode):
        self.coloring_mode = mode

    def set_max_iter(self, new_max):
        self.max_iter = new_max
        self._iter_buf = None
        self.settings.max_iter = new_max
        self.renderer = Renderer(self.fractal, self.backend, self.settings, engine=self.renderer.engine)
        self.stop()

    def set_samples(self, new_samples):
        self.samples = new_samples
        self._iter_buf = None
        self.settings.samples = new_samples
        self.renderer = Renderer(self.fractal, self.backend, self.settings, engine=self.renderer.engine)
        self.stop()

    def set_image_size(self, width, height):
        self.width = width
        self.height = height
        self._iter_buf = None
        self._iter_canvas = None
        self.stop()

    def set_precision(self, new_precision_mode):
        if new_precision_mode == Precisions.Single:
            new_precision = np.float32
        else:
            new_precision = np.float64
        if new_precision == self.precision:
            return
        self.precision = new_precision
        self.settings.precision = new_precision
        self._iter_buf = None
        self.renderer = Renderer(self.fractal, self.backend, self.settings, engine=self.renderer.engine)
        self.stop()

    def set_engine_mode(self, mode, tile_w: int = None, tile_h: int = None, adaptive_opts: dict = None):
        self.engine_mode = mode
        if tile_w: self.tile_w = int(tile_w)
        if tile_h: self.tile_h = int(tile_h)
        self.stop()

        if self.engine_mode == EngineMode.FULL_FRAME:
            self.renderer.set_engine(FullFrameEngine())

        elif self.engine_mode == EngineMode.TILED:
            if adaptive_opts:
                self.renderer.set_engine(AdaptiveTileEngine(
                    min_tile=adaptive_opts.get("min_tile", 64),
                    max_tile=adaptive_opts.get("max_tile", self.tile_w),
                    target_ms=adaptive_opts.get("target_ms", 12.0),
                    max_depth=adaptive_opts.get("max_depth", 4),
                    sample_stride=adaptive_opts.get("sample_stride", 8),
                    parallel=adaptive_opts.get("parallel", True),
                    max_workers=adaptive_opts.get("max_workers", 0),
                    on_tile=None
                ))
            else:
                self.renderer.set_engine(TileEngine(tile_w=self.tile_w, tile_h=self.tile_h))

        self.renderer = Renderer(self.fractal, self.backend, self.settings, engine=self.renderer.engine)
        self.stop()
