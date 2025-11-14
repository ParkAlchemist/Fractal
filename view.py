import sys

from render import FullImageRenderer
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QComboBox, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt, QTimer
from datetime import datetime
import math
from palettes import palettes


class MandelbrotViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mandelbrot Viewer")
        self.setGeometry(100, 100, 1920, 1080)

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

        # Zoom animation
        self.zoom_queue = []
        self._zoom_animation_step = 0
        self._zoom_animation_steps = 20
        self._zoom_animation_step_interval = 16  # ~60 FPS
        self.zoom_mouse_x = 0.0
        self.zoom_mouse_y = 0.0
        self.mouse_fx = 0.0
        self.mouse_fy = 0.0
        self._zoom_animation_timer = QTimer()
        self._zoom_animation_timer.timeout.connect(self._perform_zoom_step)
        self._zoom_animation_active = False

        self.palette_name = "Classic"
        self.palette = palettes[self.palette_name]

        self.cached_image = None

        # Full image renderer
        self.renderer = None

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
        if self.renderer:
            self.renderer.update_palette(self.palette)
        self.render_fractal()

    def save_image(self):
        if self.cached_image:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Image",
                f"mandelbrot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                "PNG Files (*.png)"
            )
            if path:
                self.cached_image.save(path)

    def render_fractal(self, progressive=True):
        width = self.label.width()
        height = self.label.height()
        if width == 0 or height == 0:
            return

        # Initialize FullImageRenderer if not already
        if self.renderer is None:
            self.renderer = FullImageRenderer(
                width, height, self.palette, kernel="opencl"
            )
            self.renderer.image_updated.connect(self.update_image)

        # Compute bounds based on center and zoom
        scale = 1.0 / self.zoom
        min_x = self.center_x - (width / 2) * scale
        max_x = self.center_x + (width / 2) * scale
        min_y = self.center_y - (height / 2) * scale
        max_y = self.center_y + (height / 2) * scale

        self.renderer.render_frame(min_x, max_x, min_y, max_y, progressive=progressive)

    def update_image(self, image):
        self.cached_image = image
        self.label.setPixmap(QPixmap.fromImage(self.cached_image))

    # Zoom handling
    def wheelEvent(self, event):
        zoom_in = event.angleDelta().y() > 0
        zoom_factor = self.zoom_factor if zoom_in else (1 / self.zoom_factor)
        mouse_pos = event.pos()
        self.zoom_queue.append((zoom_factor, mouse_pos))
        if not self._zoom_animation_active:
            self._start_next_zoom_animation()

    def _start_next_zoom_animation(self):
        if not self.zoom_queue:
            return

        zoom_factor, mouse_pos = self.zoom_queue.pop(0)
        self._zoom_animation_active = True

        self.prev_center_x = self.center_x
        self.prev_center_y = self.center_y
        self.prev_zoom = self.zoom

        self.zoom_mouse_x = mouse_pos.x()
        self.zoom_mouse_y = mouse_pos.y()

        self.mouse_fx = self.center_x + (self.zoom_mouse_x - self.label.width() / 2) / self.zoom
        self.mouse_fy = self.center_y + (self.zoom_mouse_y - self.label.height() / 2) / self.zoom

        self.zoom *= zoom_factor
        self.center_x = self.mouse_fx
        self.center_y = self.mouse_fy

        self._zoom_animation_step = 0
        self._zoom_animation_timer.start(self._zoom_animation_step_interval)
        self.render_fractal(progressive=False)

    def _perform_zoom_step(self):
        t = self._zoom_animation_step / self._zoom_animation_steps
        eased_t = t * t * (3 - 2 * t)  # ease-in-out
        self._zoom_animation_step += 1

        if self._zoom_animation_step >= self._zoom_animation_steps:
            self._zoom_animation_timer.stop()
            self._zoom_animation_active = False
            if self.zoom_queue:
                self._start_next_zoom_animation()

    def closeEvent(self, event):
        if self.renderer:
            self.renderer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = MandelbrotViewer()
    viewer.show()
    viewer.render_fractal(progressive=False)
    sys.exit(app.exec_())
