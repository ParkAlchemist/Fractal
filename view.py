import sys
from render import FullImageRenderer
from PyQt5.QtWidgets import QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget, QPushButton, QComboBox, QFileDialog, QHBoxLayout
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtCore import Qt
from datetime import datetime
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

        self.palette_name = "Classic"
        self.palette = palettes[self.palette_name]

        self.cached_image = None
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

    def render_fractal(self):
        width = self.label.width()
        height = self.label.height()
        if width == 0 or height == 0:
            return

        # Initialize FullImageRenderer if not already
        if self.renderer is None:
            self.renderer = FullImageRenderer(
                width, height, self.palette, kernel="auto", max_iter=1000, samples=2
            )
            self.renderer.image_updated.connect(self.update_image)

        # Compute bounds based on center and zoom
        scale = 1.0 / self.zoom
        min_x = self.center_x - (width / 2) * scale
        max_x = self.center_x + (width / 2) * scale
        min_y = self.center_y - (height / 2) * scale
        max_y = self.center_y + (height / 2) * scale

        # Single-pass render
        self.renderer.render_frame(min_x, max_x, min_y, max_y)

    def update_image(self, image):
        self.cached_image = image
        self.label.setPixmap(QPixmap.fromImage(self.cached_image))

    # Instant zoom on mouse wheel
    def wheelEvent(self, event):
        zoom_in = event.angleDelta().y() > 0
        zoom_factor = self.zoom_factor if zoom_in else (1 / self.zoom_factor)
        mouse_pos = event.pos()

        # Adjust center based on mouse position
        mouse_fx = self.center_x + (mouse_pos.x() - self.label.width() / 2) / self.zoom
        mouse_fy = self.center_y + (mouse_pos.y() - self.label.height() / 2) / self.zoom

        self.zoom *= zoom_factor
        self.center_x = mouse_fx
        self.center_y = mouse_fy

        self.render_fractal()

    def closeEvent(self, event):
        if self.renderer:
            self.renderer.stop()
        event.accept()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = MandelbrotViewer()
    viewer.show()
    viewer.render_fractal()
    sys.exit(app.exec_())
