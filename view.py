import sys
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton,
    QComboBox, QFileDialog, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPoint, QTimer
from datetime import datetime

from numba.cuda import select_device

from palettes import palettes
from fractal import render_mandelbrot


class FractalRenderThread(QThread):
    image_ready = pyqtSignal(QImage)

    def __init__(self, width, height, max_iter, center_x, center_y, zoom,
                 palette):
        super().__init__()
        self.width = width
        self.height = height
        self.max_iter = max_iter
        self.center_x = center_x
        self.center_y = center_y
        self.zoom = zoom
        self.palette = palette

    def run(self):

        min_x = self.center_x - self.width / (2 * self.zoom)
        max_x = self.center_x + self.width / (2 * self.zoom)
        min_y = self.center_y - self.height / (2 * self.zoom)
        max_y = self.center_y + self.height / (2 * self.zoom)

        data = render_mandelbrot(self.width, self.height, self.max_iter, min_x, max_x, min_y, max_y)

        image = QImage(self.width, self.height, QImage.Format_RGB32)
        for x in range(self.width):
            for y in range(self.height):
                color = self.palette[data[y, x] % len(self.palette)]
                r, g, b = color
                image.setPixel(x, y, r << 16 | g << 8 | b)

        self.image_ready.emit(image)


class MandelbrotViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Segmented Mandelbrot Viewer")
        self.setGeometry(100, 100, 1280, 720)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        self.center_x = -0.5
        self.center_y = 0.0
        self.zoom = 200.0
        self.max_iter = 500

        self.zoom_timer = QTimer()
        self.zoom_timer.setSingleShot(True)
        self.zoom_timer.timeout.connect(self.render_fractal)

        self.palette_name = "Classic"
        self.palette = palettes[self.palette_name]

        self.cached_image = None
        self.drag_start = None
        self.drag_offset = QPoint(0, 0)

        self.init_ui()

    def init_ui(self):
        layout = QVBoxLayout()
        layout.addWidget(self.label)

        controls = QHBoxLayout()
        self.palette_combo = QComboBox()
        self.palette_combo.addItems(palettes.keys())
        self.palette_combo.setCurrentText(self.palette_name)
        self.palette_combo.currentTextChanged.connect(self.change_palette)
        controls.addWidget(self.palette_combo)

        save_button = QPushButton("Save Image")
        save_button.clicked.connect(self.save_image)
        controls.addWidget(save_button)

        layout.addLayout(controls)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

    def change_palette(self, name):
        self.palette_name = name
        self.palette = palettes[name]
        self.render_fractal()

    def save_image(self):
        if self.cached_image:
            path, _ = QFileDialog.getSaveFileName(self, "Save Image", f"mandelbrot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", "PNG Files (*.png)")
            if path:
                self.cached_image.save(path)

    def render_fractal(self):

        width = self.label.width()
        height = self.label.height()

        if not self.cached_image:
            self.cached_image = QImage(width, height, QImage.Format_RGB32)
            self.cached_image.fill(Qt.black)
        self.label.setPixmap(QPixmap.fromImage(self.cached_image))

        self.thread = FractalRenderThread(width, height, self.max_iter,
                                          self.center_x, self.center_y,
                                          self.zoom, self.palette)
        self.thread.image_ready.connect(self.update_image)
        self.thread.start()

    def update_image(self, image):
        self.cached_image = image
        self.label.setPixmap(QPixmap.fromImage(image))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            self.drag_start = event.pos()

    def mouseMoveEvent(self, event):
        if self.drag_start:
            self.drag_offset = event.pos() - self.drag_start
            self.update_drag_view()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drag_start:
            dx = self.drag_offset.x()
            dy = self.drag_offset.y()
            self.center_x -= dx / self.zoom
            self.center_y -= dy / self.zoom
            self.drag_start = None
            self.drag_offset = QPoint(0, 0)
            self.render_tiles()

    def update_drag_view(self):
        if self.cached_image:
            offset_x = self.drag_offset.x()
            offset_y = self.drag_offset.y()
            shifted = QImage(self.cached_image.size(), QImage.Format_RGB32)
            shifted.fill(Qt.black)

            for x in range(self.cached_image.width()):
                for y in range(self.cached_image.height()):
                    src_x = x - offset_x
                    src_y = y - offset_y
                    if 0 <= src_x < self.cached_image.width() and 0 <= src_y < self.cached_image.height():
                        shifted.setPixel(x, y, self.cached_image.pixel(src_x, src_y))

            self.label.setPixmap(QPixmap.fromImage(shifted))

    def wheelEvent(self, event):
        zoom_factor = 1.2 if event.angleDelta().y() > 0 else 0.8
        mouse_pos = event.pos()
        dx = mouse_pos.x() - self.label.width() / 2
        dy = mouse_pos.y() - self.label.height() / 2
        self.center_x += dx / self.zoom * (1 - 1 / zoom_factor)
        self.center_y += dy / self.zoom * (1 - 1 / zoom_factor)
        self.zoom *= zoom_factor

        # Smooth zoom using cached image
        if self.cached_image:
            scaled = self.cached_image.scaled(
                self.label.width(), self.label.height(),
                Qt.KeepAspectRatio, Qt.SmoothTransformation
            )
            self.label.setPixmap(QPixmap.fromImage(scaled))

        self.zoom_timer.start(150)  # Wait 150ms after last zoom before rendering


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = MandelbrotViewer()
    viewer.show()
    viewer.render_fractal()
    sys.exit(app.exec_())
