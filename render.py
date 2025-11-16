import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject, QThread
from PyQt5.QtGui import QImage
from fractal import Mandelbrot


class FullImageRenderWorker(QThread):
    """Worker thread for full-frame rendering (single pass)."""
    image_rendered = pyqtSignal(QImage)  # QImage ready to display

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
            samples=self.samples
        )

        # Ensure contiguous for QImage
        if not data.flags["C_CONTIGUOUS"]:
            data = data.copy()

        h, w, c = data.shape
        bytes_per_line = w * c
        qimage = QImage(data.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()
        self.image_rendered.emit(qimage)

    def stop(self):
        self._running = False


class FullImageRenderer(QObject):
    """Simple full-frame Mandelbrot renderer (single pass only)."""
    image_updated = pyqtSignal(QImage)

    def __init__(self, width, height, palette, kernel="opencl", max_iter=1000, samples=2):
        super().__init__()
        self.width = width
        self.height = height
        self.palette = palette
        self.kernel = kernel
        self.max_iter = max_iter
        self.samples = samples

        # Create Mandelbrot instance for full frame
        self.mandelbrot = Mandelbrot(palette, kernel=kernel,
                                     img_width=width, img_height=height,
                                     max_iter=max_iter)

        # Worker thread
        self._worker = None

    def render_frame(self, min_x, max_x, min_y, max_y):
        """Render a single full-quality frame."""
        # Stop any running worker
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()

        # Start new worker
        self._worker = FullImageRenderWorker(
            mandelbrot=self.mandelbrot,
            min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
            samples=self.samples
        )
        self._worker.image_rendered.connect(self.image_updated.emit)
        self._worker.start()
        self._worker.wait()  # Wait for completion before returning

    def stop(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()

    def update_palette(self, palette):
        """Update the palette for Mandelbrot instance."""
        self.palette = palette
        self.mandelbrot.change_palette(palette)
