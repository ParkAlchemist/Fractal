import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject, QThread
from PyQt5.QtGui import QImage

from fractal import Mandelbrot
from kernel import Kernel


class ColoringMode:
    EXTERIOR = 0
    INTERIOR = 1
    HYBRID = 2

class FullImageRenderWorker(QThread):
    iter_calc_done = pyqtSignal(np.ndarray)

    def __init__(self, mandelbrot: Mandelbrot, min_x, max_x, min_y, max_y, samples=2):
        super().__init__()
        self.mandelbrot = mandelbrot
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.samples = samples
        self._running = True

    def run(self):
        if not self._running:
            return

        # Render full image
        data = self.mandelbrot.render(
            min_x=self.min_x,
            max_x=self.max_x,
            min_y=self.min_y,
            max_y=self.max_y,
        )

        # Ensure contiguous for QImage
        if not data.flags["C_CONTIGUOUS"]:
            data = data.copy()

        self.iter_calc_done.emit(data)

    def stop(self):
        self._running = False


class FullImageRenderer(QObject):
    image_updated = pyqtSignal(QImage)

    def __init__(self, width, height, palette, kernel=Kernel.AUTO,
                 max_iter=200, samples=2, coloring_mode=ColoringMode.HYBRID,
                 interior_color=(100, 100, 100)):
        super().__init__()
        self.width = width
        self.height = height
        self.kernel = kernel
        self.max_iter = max_iter
        self.samples = samples
        self.palette = np.array(palette, dtype=np.uint8)
        self.coloring_mode = coloring_mode
        self.interior_color = np.array(interior_color, dtype=np.uint8)

        self._iter_buf = None
        self._viewport = None

        # Create Mandelbrot instance for full frame
        self.mandelbrot = Mandelbrot(kernel=kernel,
                                     img_width=width, img_height=height,
                                     max_iter=max_iter, enable_timing=True,
                                     samples=samples)

        # Worker thread
        self._worker = None

    def render_frame(self, min_x, max_x, min_y, max_y):
        # Stop any running worker
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()

        if self._viewport == (min_x, max_x, min_y, max_y) and self._iter_buf is not None:
            self.render_image()
            return

        if self._viewport != (min_x, max_x, min_y, max_y) or self._iter_buf is None:
            self._viewport = (min_x, max_x, min_y, max_y)

        # Start new worker
        self._worker = FullImageRenderWorker(
            mandelbrot=self.mandelbrot,
            min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
            samples=self.samples
        )
        self._worker.iter_calc_done.connect(self._set_iter_buf)
        self._worker.start()

    def _set_iter_buf(self, data: np.ndarray):
        self._iter_buf = data
        self.render_image()

    def render_image(self):
        # Coloring
        rgb_img = self._apply_palette()

        # Convert to QImage
        h, w, c = rgb_img.shape
        bytes_per_line = w * c
        qimage = QImage(rgb_img.data, w, h, bytes_per_line,
                        QImage.Format_RGB888).copy()
        self.image_updated.emit(qimage)

    def _apply_palette(self):
        h, w = self._iter_buf.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        palette_size = len(self.palette)

        # Mask for interior points
        interior_mask = self._iter_buf >= 0.999
        exterior_mask = ~interior_mask

        # Handle exterior points
        if self.coloring_mode in (ColoringMode.EXTERIOR, ColoringMode.HYBRID):
            vals = self._iter_buf[exterior_mask]
            idx_f = vals * (palette_size - 1)
            idx = np.clip(idx_f.astype(np.int32), 0, palette_size - 1)
            t = idx_f - idx
            idx_next = np.clip(idx + 1, 0, palette_size - 1)

            colors0 = self.palette[idx]
            colors1 = self.palette[idx_next]
            blended = ((1 - t)[:, None] * colors0 + t[:, None] * colors1).astype(np.uint8)
            rgb[exterior_mask] = blended

        # Handle interior points
        if self.coloring_mode in (ColoringMode.INTERIOR, ColoringMode.HYBRID):
            if np.any(interior_mask):
                y_coords, x_coords = np.where(interior_mask)
                cx, cy = w / 2.0, h / 2.0
                dist = np.sqrt((x_coords - cx)**2 + (y_coords - cy)**2)
                dist_norm = dist / dist.max() if dist.max() > 0 else dist
                gradient_colors = (self.interior_color * (1 - dist_norm[:, None])).astype(np.uint8)
                rgb[interior_mask] = gradient_colors

        return rgb

    def stop(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()

    # ------------- Parameter setters -----------------
    def set_kernel(self, new_kernel):
        self.kernel = new_kernel
        self._iter_buf = None
        self.mandelbrot.change_kernel(new_kernel)

    def set_palette(self, palette):
        self.palette = np.array(palette, dtype=np.uint8)

    def set_coloring_mode(self, mode):
        self.coloring_mode = mode

    def set_interior_color(self, color):
        self.interior_color = np.array(color, dtype=np.uint8)

    def set_max_iter(self, new_max):
        self.max_iter = new_max
        self._iter_buf = None
        self.mandelbrot.max_iter = new_max

    def set_samples(self, new_sample_amount):
        self.samples = new_sample_amount
        self._iter_buf = None
        self.mandelbrot.samples = new_sample_amount

    def set_image_size(self, width, height):
        self.width = width
        self.height = height
        self.mandelbrot.change_image_size(width, height)
        self._iter_buf = None
