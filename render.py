import numpy as np
from PyQt5.QtCore import pyqtSignal, QObject, QThread
from PyQt5.QtGui import QImage
from fractal import Mandelbrot


class FullImageRenderWorker(QThread):
    """Worker thread for full-frame rendering, allows optional progressive refinement."""
    image_rendered = pyqtSignal(QImage, int)  # QImage, pass_id (0=quick,1=refine)

    def __init__(self, mandelbrot: Mandelbrot, min_x, max_x, min_y, max_y,
                 samples=2, pass_id=0):
        super().__init__()
        self.mandelbrot = mandelbrot
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.samples = samples
        self.pass_id = pass_id
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
        self.image_rendered.emit(qimage, self.pass_id)

    def stop(self):
        self._running = False

class FullImageRenderer(QObject):
    """
    Simple full-frame Mandelbrot renderer.
    Supports optional two-pass progressive refinement (quick preview then high quality).
    Suitable for zoom animations where each frame is rendered completely.
    """
    image_updated = pyqtSignal(QImage)  # QImage ready to display

    def __init__(self, width, height, palette, kernel="opencl",
                 quick_max_iter=200, refine_max_iter=1000,
                 quick_samples=1, refine_samples=2):
        super().__init__()
        self.width = width
        self.height = height
        self.palette = palette
        self.kernel = kernel

        # Create Mandelbrot instance for full frame
        self.mandelbrot_quick = Mandelbrot(palette, kernel=kernel,
                                           img_width=width, img_height=height,
                                           max_iter=quick_max_iter)
        self.mandelbrot_refine = Mandelbrot(palette, kernel=kernel,
                                            img_width=width, img_height=height,
                                            max_iter=refine_max_iter)

        # Progressive refinement settings
        self.quick_max_iter = quick_max_iter
        self.refine_max_iter = refine_max_iter
        self.quick_samples = quick_samples
        self.refine_samples = refine_samples

        # Worker thread (single at a time)
        self._worker = None

    def render_frame(self, min_x, max_x, min_y, max_y, progressive=True):
        """
        Render a frame of the Mandelbrot set.
        If progressive=True, emit quick pass first, then refined pass.
        """
        # Stop any running worker
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()

        # Quick pass
        if progressive:
            self._worker = FullImageRenderWorker(
                mandelbrot=self.mandelbrot_quick,
                min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
                samples=self.quick_samples,
                pass_id=0
            )
            self._worker.image_rendered.connect(lambda img, pid: self.image_updated.emit(img))
            self._worker.start()
            self._worker.wait()  # optionally wait before starting refine, or let UI update asynchronously

        # Refine pass
        self._worker = FullImageRenderWorker(
            mandelbrot=self.mandelbrot_refine,
            min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y,
            samples=self.refine_samples,
            pass_id=1
        )
        self._worker.image_rendered.connect(lambda img, pid: self.image_updated.emit(img))
        self._worker.start()
        self._worker.wait()  # wait for frame completion

    def stop(self):
        if self._worker is not None:
            self._worker.stop()
            self._worker.wait()

    def update_palette(self, palette):
        """Update the palette for both quick and refine Mandelbrot instances."""
        self.palette = palette
        self.mandelbrot_quick.change_palette(palette)
        self.mandelbrot_refine.change_palette(palette)
