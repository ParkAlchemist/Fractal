from PyQt5.QtCore import pyqtSignal, QObject, QThread
from PyQt5.QtGui import QImage

from fractal import Mandelbrot


class FullImageRenderWorker(QThread):
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
    image_updated = pyqtSignal(QImage)

    def __init__(self, width, height, palette, kernel="auto", max_iter=1000, samples=2):
        super().__init__()
        self.width = width
        self.height = height
        self.kernel = kernel
        self.max_iter = max_iter
        self.samples = samples

        # Create Mandelbrot instance for full frame
        self.mandelbrot = Mandelbrot(palette, kernel=kernel,
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

        # Start new worker
        self._worker = FullImageRenderWorker(
            mandelbrot=self.mandelbrot,
            min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
            samples=self.samples
        )
        self._worker.image_rendered.connect(self.image_updated.emit)
        self._worker.start()

    def stop(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()

    def update_palette(self, palette):
        self.mandelbrot.change_palette(palette)

    def update_max_iter(self, new_max):
        self.max_iter = new_max
        self.mandelbrot.max_iter = new_max

    def update_samples(self, new_sample_amount):
        self.samples = new_sample_amount
        self.mandelbrot.samples = new_sample_amount

    def set_image_size(self, width, height):
        self.width = width
        self.height = height
        self.mandelbrot.change_image_size(width, height)
