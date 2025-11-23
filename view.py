import sys
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
    QPushButton, QComboBox, QFileDialog, QHBoxLayout, QDockWidget, QTabWidget,
    QFormLayout, QLineEdit, QSizePolicy)
from PyQt5.QtGui import QPixmap, QPainter, QImage
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from torchgen.api.types import layoutT

from palettes import palettes
from render import FullImageRenderer
from kernel import Kernel


class MandelbrotViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mandelbrot Viewer")
        self.window_height = 720
        self.aspect_ratio = 16/9
        self.window_width = int(self.window_height * self.aspect_ratio)
        self.setGeometry(100, 100, self.window_width, self.window_height)

        # Main fractal display
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignCenter)
        self.label.setScaledContents(True)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding,
                                 QSizePolicy.Policy.Expanding)

        # View parameters
        self.center_x = -0.5
        self.center_y = 0.0
        self.zoom = 250.0
        self.zoom_factor = 1.5

        # Animation state
        self.animation_timer = None
        self.animation_steps = 40
        self.current_step = 0
        self.phase = 1
        self.start_x = self.start_y = self.start_zoom = 0
        self.target_x = self.target_y = self.target_zoom = 0
        self.final_render_pending = False
        self.mouse_focus_x = 0
        self.mouse_focus_y = 0

        # Drag state
        self.dragging = False
        self.last_mouse_pos = None
        self.start_center_x = None
        self.start_center_y = None

        # Palette
        self.palette_name = "Classic"
        self.palette = palettes[self.palette_name]

        # Image
        self.cached_image = None
        self.rendered_image = None
        self.renderer = None

        self.init_ui()

    def init_ui(self):
        # Central layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)

        # Bottom controls
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

        # --------------------------------------
        # ----------- Side Menu ----------------
        # --------------------------------------
        self.side_menu = QDockWidget("Controls", self)
        self.side_menu.setAllowedAreas(Qt.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.side_menu)
        self.side_menu.setFeatures(QDockWidget.NoDockWidgetFeatures)

        tabs = QTabWidget()
        self.side_menu.setWidget(tabs)

        # --------- Render tab -----------------
        render_tab = QWidget()
        render_layout = QFormLayout()

        # Max iter
        self.iter_input = QComboBox()
        self.iter_input.addItems(["100", "200", "500", "1000", "2000", "5000"])
        self.iter_input.setCurrentText("1000")
        render_layout.addRow("Max Iterations: ", self.iter_input)

        # Samples
        self.samples_input = QComboBox()
        self.samples_input.addItems(["1", "2", "4", "8"])
        self.samples_input.setCurrentText("2")
        render_layout.addRow("Samples: ", self.samples_input)

        # Apply button
        apply_render_btn = QPushButton("Apply")
        apply_render_btn.clicked.connect(self.apply_render_settings)
        render_layout.addRow(apply_render_btn)

        render_tab.setLayout(render_layout)

        # --------- Palette tab -----------------
        palette_tab = QWidget()
        palette_layout = QVBoxLayout()
        palette_layout.addWidget(self.palette_combo)
        palette_tab.setLayout(palette_layout)

        # --------- View tab -----------------
        view_tab = QWidget()
        view_layout = QFormLayout()

        # Center
        self.center_x_input = QLineEdit(str(self.center_x))
        self.center_y_input = QLineEdit(str(self.center_y))
        view_layout.addRow("Center X: ", self.center_x_input)
        view_layout.addRow("Center Y: ", self.center_y_input)

        # Zoom
        self.zoom_input = QLineEdit(str(self.zoom))
        view_layout.addRow("Zoom: ", self.zoom_input)

        # Zoom Factor
        self.zoom_factor_input = QLineEdit(str(self.zoom_factor))
        view_layout.addRow("Zoom Factor: ", self.zoom_factor_input)

        # Apply button
        apply_view_btn = QPushButton("Apply")
        apply_view_btn.clicked.connect(self.apply_view_settings)
        view_layout.addRow(apply_view_btn)

        view_tab.setLayout(view_layout)

        # ---------- Finalíze -------------
        tabs.addTab(render_tab, "Render")
        tabs.addTab(palette_tab, "Palette")
        tabs.addTab(view_tab, "View")


    # ---------- Tab Actions -----------------
    def apply_render_settings(self):
        if self.renderer:
            max_iter = int(self.iter_input.currentText())
            samples = int(self.samples_input.currentText())

            self.renderer.update_max_iter(max_iter)
            self.renderer.update_samples(samples)

            self.render_fractal()

    def apply_view_settings(self):
        try:
            self.center_x = float(self.center_x_input.text())
            self.center_y = float(self.center_y_input.text())
            self.zoom = float(self.zoom_input.text())
            self.zoom_factor = float(self.zoom_factor_input.text())
            self.render_fractal()
        except ValueError:
            pass

    def update_view_tab_fields(self):
        if hasattr(self, "center_x_input") and hasattr(self, "center_y_input") and hasattr(self, "zoom_input"):
            self.center_x_input.setText(str(self.center_x))
            self.center_y_input.setText(str(self.center_y))
            self.zoom_input.setText(str(self.zoom))

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

        if self.renderer is None:
            self.renderer = FullImageRenderer(width, height,
                                              self.palette, kernel=Kernel.CPU,
                                              max_iter=1000, samples=2)
            self.renderer.image_updated.connect(self.update_image)

        scale = 1.0 / self.zoom
        min_x = self.center_x - (width / 2) * scale
        max_x = self.center_x + (width / 2) * scale
        min_y = self.center_y - (height / 2) * scale
        max_y = self.center_y + (height / 2) * scale

        self.renderer.render_frame(min_x, max_x, min_y, max_y)

    def update_image(self, image):
        self.rendered_image = image
        if not self.cached_image:
            self.cached_image = image
        if self.animation_timer and self.animation_timer.isActive():
            self.final_render_pending = True
        else:
            self._fade_in_image(image)
            self.cached_image = self.rendered_image
            self.rendered_image = None

    def _fade_in_image(self, image):
        old_image = self.cached_image
        new_image = image

        width, height = self.label.width(), self.label.height()

        # Create animation
        anim = QPropertyAnimation(self, b"dummy")  # dummy property
        anim.setDuration(400)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Linear)

        # Update blend on each frame
        def update_blend(value):
            final_pixmap = QPixmap(self.label.size())
            final_pixmap.fill(Qt.black)
            painter = QPainter(final_pixmap)

            if old_image:
                old_pixmap = QPixmap.fromImage(old_image).scaled(width, height,
                                                                 Qt.AspectRatioMode.KeepAspectRatio,
                                                                 Qt.TransformationMode.SmoothTransformation)
                painter.setOpacity(1.0 - value)
                painter.drawPixmap(0, 0, old_pixmap)

            new_pixmap = QPixmap.fromImage(new_image).scaled(width, height,
                                                             Qt.AspectRatioMode.KeepAspectRatio,
                                                             Qt.TransformationMode.SmoothTransformation)
            painter.setOpacity(value)
            painter.drawPixmap(0, 0, new_pixmap)

            painter.end()
            self.label.setPixmap(final_pixmap)

        anim.valueChanged.connect(update_blend)

        # After animation finishes, show final rendered image
        def finalize():
            self.label.setPixmap(QPixmap.fromImage(new_image))
            self.cached_image = new_image

        anim.finished.connect(finalize)

        # Keep reference so GC doesn't kill animation
        self.fade_anim = anim
        anim.start()

    # ---------------- Zoom Animation ----------------
    def wheelEvent(self, event):
        if not self.label.geometry().contains(event.pos()):
            return

        zoom_in = event.angleDelta().y() > 0
        zoom_factor = self.zoom_factor if zoom_in else (1 / self.zoom_factor)
        mouse_pos = event.pos()

        self.mouse_focus_x = mouse_pos.x()
        self.mouse_focus_y = mouse_pos.y()

        self.target_zoom = self.zoom * zoom_factor
        self.target_x = self.center_x + (mouse_pos.x() - self.label.width() / 2) / self.zoom
        self.target_y = self.center_y + (mouse_pos.y() - self.label.height() / 2) / self.zoom

        # Start final render immediately
        self._start_final_render(self.target_x, self.target_y, self.target_zoom)

        # Prepare animation
        self.start_x, self.start_y, self.start_zoom = self.center_x, self.center_y, self.zoom
        self.update_view_tab_fields()
        self.phase = 0 if zoom_in else 1
        self.current_step = 0

        if self.animation_timer is None:
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self._perform_anim_step)

        self.animation_timer.start(16)  # ~60 FPS

    def _start_final_render(self, cx, cy, zoom):
        scale = 1.0 / zoom
        min_x = cx - (self.label.width() / 2) * scale
        max_x = cx + (self.label.width() / 2) * scale
        min_y = cy - (self.label.height() / 2) * scale
        max_y = cy + (self.label.height() / 2) * scale
        self.renderer.render_frame(min_x, max_x, min_y, max_y)

    def _perform_anim_step(self):
        self.current_step += 1

        if self.current_step <= self.animation_steps // 2:
            t = self.current_step / (self.animation_steps // 2)
        else:
            t = (self.current_step - (self.animation_steps // 2)) / (self.animation_steps // 2)

        eased_t = t * t * (3 - 2 * t)

        if self.cached_image:
            pixmap = QPixmap.fromImage(self.cached_image)
            if self.phase == 0:
                self._pan_step(eased_t, pixmap)
            elif self.phase == 1:
                self._zoom_step(eased_t, pixmap)

        # Change phase
        if self.current_step == self.animation_steps // 2:
            if self.phase == 0:
                self.center_x, self.center_y = self.target_x, self.target_y
                self.update_view_tab_fields()
                self.phase = 1
            elif self.phase == 1:
                self.zoom = self.target_zoom
                self.update_view_tab_fields()
                self.phase = 0
            self.cached_image = QImage(self.label.pixmap())

        # Animation complete
        if self.current_step >= self.animation_steps:
            self.animation_timer.stop()
            self.center_x, self.center_y, self.zoom = self.target_x, self.target_y, self.target_zoom
            self.update_view_tab_fields()
            self.cached_image = QImage(self.label.pixmap())
            if self.final_render_pending:
                self._fade_in_image(self.rendered_image)
                self.cached_image = self.rendered_image
                self.rendered_image = None
                self.final_render_pending = False


    def _pan_step(self, eased_t, pixmap):
        interp_x = self.start_x + (self.target_x - self.start_x) * eased_t
        interp_y = self.start_y + (self.target_y - self.start_y) * eased_t
        dx = int((interp_x - self.start_x) * self.zoom)
        dy = int((interp_y - self.start_y) * self.zoom)

        final_pixmap = QPixmap(self.label.size())
        final_pixmap.fill(Qt.black)
        painter = QPainter(final_pixmap)
        painter.drawPixmap(-dx, -dy, pixmap)
        painter.end()
        self.label.setPixmap(final_pixmap)

    def _zoom_step(self, eased_t, pixmap):
        interp_zoom = self.start_zoom + (
                    self.target_zoom - self.start_zoom) * eased_t
        scale_factor = interp_zoom / self.start_zoom
        scaled_width = int(self.label.width() * scale_factor)
        scaled_height = int(self.label.height() * scale_factor)
        scaled_pixmap = pixmap.scaled(scaled_width, scaled_height,
                                      Qt.KeepAspectRatio,
                                      Qt.SmoothTransformation)

        dx = int(self.label.width() / 2 - scaled_pixmap.width() / 2)
        dy = int(self.label.height() / 2 - scaled_pixmap.height() / 2)

        final_pixmap = QPixmap(self.label.size())
        final_pixmap.fill(Qt.black)
        painter = QPainter(final_pixmap)
        painter.drawPixmap(dx, dy, scaled_pixmap)
        painter.end()
        self.label.setPixmap(final_pixmap)

    # ---------------- Panning ----------------
    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.label.geometry().contains(event.pos()):
                self.dragging = True
                self.last_mouse_pos = event.pos()
                self.start_center_x = self.center_x
                self.start_center_y = self.center_y

    def mouseMoveEvent(self, event):
        if self.dragging and self.cached_image:
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()

            current_center_x = self.start_center_x - dx / self.zoom
            current_center_y = self.start_center_y - dy / self.zoom

            pixmap = QPixmap.fromImage(self.cached_image)
            scaled_pixmap = pixmap.scaled(self.label.width(),
                                          self.label.height(),
                                          Qt.AspectRatioMode.KeepAspectRatio,
                                          Qt.TransformationMode.SmoothTransformation)

            final_pixmap = QPixmap(self.label.size())
            final_pixmap.fill(Qt.black)
            painter = QPainter(final_pixmap)
            painter.drawPixmap(dx, dy, scaled_pixmap)
            painter.end()

            self.label.setPixmap(final_pixmap)

            self.center_x = current_center_x
            self.center_y = current_center_y
            self.update_view_tab_fields()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.LeftButton and self.dragging:
            self.dragging = False
            self.cached_image = QImage(self.label.pixmap())
            self.render_fractal()

    def resizeEvent(self, event):
        # Handle window resize: re-render fractal if viewport size changed
        if not hasattr(self, "resize_timer"):
            self.resize_timer = QTimer()
            self.resize_timer.setSingleShot(True)
            self.resize_timer.timeout.connect(self.render_fractal)

        if self.label.width() > 0 and self.label.height() > 0:
            if self.renderer is not None:
                self.renderer.set_image_size(self.label.width(),
                                             self.label.height())
                if self.cached_image:
                    scaled_pixmap = QPixmap.fromImage(self.cached_image).scaled(
                        self.label.size(),
                        Qt.AspectRatioMode.IgnoreAspectRatio,
                        Qt.TransformationMode.SmoothTransformation)
                    self.label.setPixmap(scaled_pixmap)
                self.resize_timer.start(250)
        super().resizeEvent(event)

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
