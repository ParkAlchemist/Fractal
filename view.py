import math
import sys

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton,
    QComboBox, QFileDialog, QHBoxLayout
)
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QPoint, QTimer
from datetime import datetime

from palettes import palettes
from tile import TileRenderer


class MandelbrotViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mandelbrot Viewer")
        self.setGeometry(100, 100, 2048, 1024)

        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)

        # Zoom params
        self.center_x = -0.5
        self.center_y = 0.0
        self.zoom = 400.0
        self.zoom_factor = 1.5
        self.prev_center_x = self.center_x
        self.prev_center_y = self.center_y
        self.prev_zoom = self.zoom
        self.zoom_timer = QTimer()
        self.zoom_timer.setSingleShot(True)
        self.zoom_timer.timeout.connect(self.render_fractal)

        # Zoom animation params
        self.zoom_queue = []
        self._zoom_animation_step = 0
        self._zoom_animation_steps = 20
        self._zoom_animation_step_interval = 16  # ~60FPS
        self.zoom_mouse_x = 0.0
        self.zoom_mouse_y = 0.0
        self.mouse_fx = 0.0
        self.mouse_fy = 0.0
        self._zoom_animation_timer = QTimer()
        self._zoom_animation_active = False
        self._zoom_animation_timer.timeout.connect(self._perform_zoom_step)

        self.max_iter = 1000

        self.palette_name = "Classic"
        self.palette = palettes[self.palette_name]

        self.cached_image = None
        self.drag_start = None
        self.drag_offset = QPoint(0, 0)

        self.tile_renderer = TileRenderer(self.palette)
        self.tile_renderer.image_updated.connect(self.update_image)

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
        self.tile_renderer.update_palette(self.palette)
        self.render_fractal()

    def save_image(self):
        if self.cached_image:
            path, _ = QFileDialog.getSaveFileName(self, "Save Image", f"mandelbrot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png", "PNG Files (*.png)")
            if path:
                self.cached_image.save(path)

    def render_fractal(self):
        width = self.label.width()
        height = self.label.height()

        self.tile_renderer.start_render(width, height, self.center_x,
                                        self.center_y, self.zoom,
                                        self.label.width(), self.label.height())

    def closeEvent(self, event):
        self.tile_renderer.stop()
        event.accept()

    def update_image(self, image):
        self.cached_image = image
        self.label.setPixmap(QPixmap.fromImage(self.cached_image))
    """
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.label.geometry().contains(event.pos()):
                self.drag_start = event.pos()
            else:
                self.drag_start = None

    def mouseMoveEvent(self, event):
        if self.drag_start is not None:
            self.drag_offset = event.pos() - self.drag_start
            self.update_drag_view()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.drag_start is not None:
            dx = self.drag_offset.x()
            dy = self.drag_offset.y()
            self.center_x -= (dx * self.image_size_mult) / self.zoom
            self.center_y -= (dy * self.image_size_mult) / self.zoom
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
    """
    def wheelEvent(self, event):
        zoom_in = event.angleDelta().y() > 0
        zoom_factor = self.zoom_factor if zoom_in else (1 / self.zoom_factor)
        mouse_pos = event.pos()

        # Add zoom task to queue
        self.zoom_queue.append((zoom_factor, mouse_pos))

        if not self._zoom_animation_active:
            self._start_next_zoom_animation()

    def _start_next_zoom_animation(self):
        if not self.zoom_queue:
            return

        zoom_factor, mouse_pos = self.zoom_queue.pop(0)

        self._zoom_animation_active = True

        # Save current state
        self.prev_center_x = self.center_x
        self.prev_center_y = self.center_y
        self.prev_zoom = self.zoom

        # Mouse pos widget coordinates
        self.zoom_mouse_x = mouse_pos.x()
        self.zoom_mouse_y = mouse_pos.y()

        # Calc fractal coords for mouse before zoom
        self.mouse_fx = self.center_x + (
                    self.zoom_mouse_x - self.label.width() / 2) / self.zoom
        self.mouse_fy = self.center_y + (
                    self.zoom_mouse_y - self.label.height() / 2) / self.zoom

        # Apply zoom
        self.zoom *= zoom_factor

        # Update center
        self.center_x = self.mouse_fx
        self.center_y = self.mouse_fy

        # Smooth zoom using cached image
        self._zoom_animation_step = 0
        self._zoom_animation_timer.start(self._zoom_animation_step_interval)

        # Wait for zoom animation to finish before rendering new image
        self.zoom_timer.start(
            self._zoom_animation_step_interval * self._zoom_animation_steps + 200)

    def _perform_zoom_step(self):
        if self._zoom_animation_step >= self._zoom_animation_steps:
            self._zoom_animation_timer.stop()
            self._zoom_animation_active = False
            self._start_next_zoom_animation()
            return

        t = self._zoom_animation_step / self._zoom_animation_steps
        eased_t = t * t * (3 - 2 * t)  # ease-in-out

        # Interpolate zoom (logarithmic)
        log_zoom_start = math.log(self.prev_zoom)
        log_zoom_end = math.log(self.zoom)
        current_zoom = math.exp(
            (1 - eased_t) * log_zoom_start + eased_t * log_zoom_end)

        # Interpolate center (linear)
        current_center_x = (1 - eased_t) * self.prev_center_x + eased_t * self.mouse_fx
        current_center_y = (1 - eased_t) * self.prev_center_y + eased_t * self.mouse_fy

        # Compute pixel offset from current center to top-left
        w = self.label.width()
        h = self.label.height()
        scale = current_zoom

        # Determine fractal-space coordinates of top-left corner
        fx0 = current_center_x - (w / 2) / scale
        fy0 = current_center_y - (h / 2) / scale

        # Determine pixel offset in cached image for cropping
        src_x = int((fx0 - (self.prev_center_x - (
                    w / 2) / self.prev_zoom)) * self.prev_zoom)
        src_y = int((fy0 - (self.prev_center_y - (
                    h / 2) / self.prev_zoom)) * self.prev_zoom)
        src_w = int(w * (self.prev_zoom / scale))
        src_h = int(h * (self.prev_zoom / scale))

        # Clamp to image bounds
        src_x = max(0, min(self.cached_image.width() - src_w, src_x))
        src_y = max(0, min(self.cached_image.height() - src_h, src_y))

        if self.cached_image and src_w > 0 and src_h > 0:
            cropped = self.cached_image.copy(src_x, src_y, src_w, src_h)
            scaled = cropped.scaled(w, h, Qt.IgnoreAspectRatio,
                                    Qt.SmoothTransformation)
            self.label.setPixmap(QPixmap.fromImage(scaled))

        self._zoom_animation_step += 1
        if self._zoom_animation_step >= self._zoom_animation_steps:
            self.tile_renderer.full_image = scaled
            self.cached_image = scaled

if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = MandelbrotViewer()
    viewer.show()
    viewer.render_fractal()
    sys.exit(app.exec_())
