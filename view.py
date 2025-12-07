import sys
from datetime import datetime

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
    QPushButton, QComboBox, QFileDialog, QHBoxLayout, QDockWidget, QTabWidget,
    QFormLayout, QLineEdit, QSizePolicy, QRadioButton, QButtonGroup, QCheckBox)
from PyQt5.QtGui import QPixmap, QPainter, QImage
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve

from palettes import palettes
from render import FullImageRenderer, ColoringMode
from fractal import Precisions
from kernel import Kernel
from utils import available_backends


class Tools:
    Drag = 0
    Click_zoom = 1
    Wheel_zoom = 2
    Set_center = 3

class FractalViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fractal Viewer")
        self.window_height = 900
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
        self.anim_fps = 60
        self.anim_interval = int((1 / self.anim_fps) * 1000)

        # Drag state
        self.dragging = False
        self.last_mouse_pos = None
        self.start_center_x = None
        self.start_center_y = None

        # Palette
        self.default_palette_name = "Classic"

        # Image
        self.cached_image = None
        self.rendered_image = None
        self.renderer = None

        self.init_ui()

    def init_ui(self):
        # Central layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)

        # ------------ Bottom controls -----------------
        controls = QHBoxLayout()

        # Save Image Button
        save_button = QPushButton("Save Image")
        save_button.clicked.connect(self.save_image)
        controls.addWidget(save_button)

        # Tools
        tools = QHBoxLayout()

        # Drag
        self.drag_tool = QCheckBox("Drag tool")
        self.drag_tool.setChecked(False)
        tools.addWidget(self.drag_tool)
        self.drag_tool.clicked.connect(lambda checked: checked and self.set_tools(Tools.Drag))

        # Click zoom
        self.click_zoom_tool = QCheckBox("Click-Zoom tool")
        self.click_zoom_tool.setChecked(False)
        tools.addWidget(self.click_zoom_tool)
        self.click_zoom_tool.clicked.connect(lambda checked: checked and self.set_tools(Tools.Click_zoom))

        # Wheel zoom
        self.wheel_zoom_tool = QCheckBox("Wheel-Zoom tool")
        self.wheel_zoom_tool.setChecked(False)
        tools.addWidget(self.wheel_zoom_tool)
        self.wheel_zoom_tool.clicked.connect(lambda checked: checked and self.set_tools(Tools.Wheel_zoom))

        # Set center
        self.set_center_tool = QCheckBox("Set-Center tool")
        self.set_center_tool.setChecked(False)
        tools.addWidget(self.set_center_tool)
        self.set_center_tool.clicked.connect(lambda checked: checked and self.set_tools(Tools.Set_center))

        layout.addLayout(tools)

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

        # Kernel
        self.kernel_input = QComboBox()
        backends = available_backends()
        self.kernel_input.addItems(sorted(backends))
        self.kernel_input.setCurrentText(backends[0])
        render_layout.addRow("Kernel: ", self.kernel_input)

        # Max iter
        self.iter_input = QComboBox()
        self.iter_input.addItems(["100", "200", "500", "1000", "2000", "5000"])
        self.iter_input.setCurrentText("200")
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
        palette_layout = QFormLayout()

        # Coloring Mode
        self.coloring_mode_group = QButtonGroup(self)
        self.radio_exterior = QRadioButton("Exterior")
        self.radio_interior = QRadioButton("Interior")
        self.radio_hybrid = QRadioButton("Hybrid")
        self.radio_exterior.setChecked(True)  # Default

        self.coloring_mode_group.addButton(self.radio_exterior)
        self.coloring_mode_group.addButton(self.radio_interior)
        self.coloring_mode_group.addButton(self.radio_hybrid)

        palette_layout.addRow("Coloring Mode: ", self.radio_exterior)
        palette_layout.addRow("", self.radio_interior)
        palette_layout.addRow("", self.radio_hybrid)

        self.radio_exterior.toggled.connect(lambda checked: checked and self.change_coloring_mode(ColoringMode.EXTERIOR))
        self.radio_interior.toggled.connect(lambda checked: checked and self.change_coloring_mode(ColoringMode.INTERIOR))
        self.radio_hybrid.toggled.connect(lambda checked: checked and self.change_coloring_mode(ColoringMode.HYBRID))

        # Exterior Palette
        self.exter_palette_input = QComboBox()
        self.exter_palette_input.addItems(palettes.keys())
        self.exter_palette_input.setCurrentText(self.default_palette_name)
        self.exter_palette_input.currentTextChanged.connect(self.change_exter_palette)
        palette_layout.addRow("Exterior Palette: ", self.exter_palette_input)

        # Interior Palette
        self.inter_palette_input = QComboBox()
        self.inter_palette_input.addItems(palettes.keys())
        self.inter_palette_input.setCurrentText(self.default_palette_name)
        self.inter_palette_input.currentTextChanged.connect(self.change_inter_palette)
        palette_layout.addRow("Interior Palette: ", self.inter_palette_input)

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

    # ----------- Rendering ---------------
    def render_fractal(self):
        width = self.label.width()
        height = self.label.height()
        if width == 0 or height == 0:
            return

        if self.renderer is None:
            self.renderer = FullImageRenderer(width, height,
                                              palettes[
                                                  self.default_palette_name],
                                              kernel=Kernel.AUTO,
                                              max_iter=200, samples=2,
                                              coloring_mode=ColoringMode.EXTERIOR,
                                              precision=Precisions.single)
            self.renderer.image_updated.connect(self.update_image)

        scale = 1.0 / self.zoom
        min_x = self.center_x - (width / 2) * scale
        max_x = self.center_x + (width / 2) * scale
        min_y = self.center_y - (height / 2) * scale
        max_y = self.center_y + (height / 2) * scale

        # Update precision
        zoom_factor = self.zoom
        if zoom_factor > 1e14:
            precision_mode = Precisions.arbitrary
        elif zoom_factor > 1e6:
            precision_mode = Precisions.double
        else:
            precision_mode = Precisions.single

        self.renderer.set_precision(precision_mode)
        self.renderer.set_zoom_factor(zoom_factor)

        self.renderer.render_frame(min_x, max_x, min_y, max_y)



    # ---------- Tab Actions -----------------
    def apply_render_settings(self):
        if self.renderer:
            max_iter = int(self.iter_input.currentText())
            samples = int(self.samples_input.currentText())

            if self.renderer.max_iter != max_iter:
                self.renderer.set_max_iter(max_iter)
            if self.renderer.samples != samples:
                self.renderer.set_samples(samples)

            kernel_str = self.kernel_input.currentText()
            if kernel_str == "OPENCL":
                new_kernel = Kernel.OPENCL
            elif kernel_str == "CUDA":
                new_kernel = Kernel.CUDA
            elif kernel_str == "CPU":
                new_kernel = Kernel.CPU
            else:
                raise ValueError("Incorrect Kernel")

            if self.renderer.kernel != new_kernel:
                # Change current kernel
                self.renderer.set_kernel(new_kernel)

            self.render_fractal()

    def change_coloring_mode(self, mode):
        if self.renderer:
            self.renderer.set_coloring_mode(mode)
            self.render_fractal()

    def apply_view_settings(self):
        try:
            self.center_x = float(self.center_x_input.text())
            self.center_y = float(self.center_y_input.text())
            self.zoom = float(self.zoom_input.text())
            self.zoom_factor = float(self.zoom_factor_input.text())
            self.render_fractal()
        except ValueError as e:
            print(f"Error in view inputs: {e}")

    def update_view_tab_fields(self):
        if hasattr(self, "center_x_input") and hasattr(self, "center_y_input") and hasattr(self, "zoom_input"):
            self.center_x_input.setText(str(self.center_x))
            self.center_y_input.setText(str(self.center_y))
            self.zoom_input.setText(str(self.zoom))

    def set_tools(self, tool):
        if tool == Tools.Drag:
            self.drag_tool.setChecked(True)
            self.click_zoom_tool.setChecked(False)
            self.set_center_tool.setChecked(False)
        if tool == Tools.Click_zoom:
            self.click_zoom_tool.setChecked(True)
            self.drag_tool.setChecked(False)
            self.wheel_zoom_tool.setChecked(False)
            self.set_center_tool.setChecked(False)
        if tool == Tools.Wheel_zoom:
            self.wheel_zoom_tool.setChecked(True)
            self.click_zoom_tool.setChecked(False)
        if tool == Tools.Set_center:
            self.set_center_tool.setChecked(True)
            self.drag_tool.setChecked(False)
            self.click_zoom_tool.setChecked(False)

        print(f"Selected tool {tool}.")

    # --------------- Palette update -----------------
    def change_exter_palette(self, name):
        if self.renderer:
            self.renderer.set_exter_palette(name)
        self.render_fractal()

    def change_inter_palette(self, name):
        if self.renderer:
            self.renderer.set_inter_palette(name)
        self.render_fractal()

    # -------------- Save Image ------------------
    def save_image(self):
        if self.cached_image:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Image",
                f"mandelbrot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                "PNG Files (*.png)"
            )
            if path:
                self.cached_image.save(path)

    # ------------ View Update ---------------
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

    def update_view(self):
        if not getattr(self, "cached_image", None):
            return

        # Start final render immediately
        self._start_final_render(self.target_x, self.target_y, self.target_zoom)

        # Prepare animation
        self.start_x, self.start_y, self.start_zoom = self.center_x, self.center_y, self.zoom

        self.update_view_tab_fields()
        self.phase = 0 if self.target_zoom > self.zoom else 1
        self.current_step = 0

        if self.animation_timer is None:
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self._perform_anim_step)

        self.animation_timer.start(self.anim_interval)

    def _start_final_render(self, cx, cy, zoom):
        scale = 1.0 / zoom
        min_x = cx - (self.label.width() / 2) * scale
        max_x = cx + (self.label.width() / 2) * scale
        min_y = cy - (self.label.height() / 2) * scale
        max_y = cy + (self.label.height() / 2) * scale

        # Update precision
        if zoom > 1e14:
            precision_mode = Precisions.arbitrary
        elif zoom > 1e6:
            precision_mode = Precisions.double
        else:
            precision_mode = Precisions.single

        self.renderer.set_precision(precision_mode)
        self.renderer.set_zoom_factor(zoom)

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

        self.center_x = interp_x
        self.center_y = interp_y
        self.update_view_tab_fields()

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

        self.zoom = interp_zoom
        self.update_view_tab_fields()

    # ---------------- Wheel zoom ----------------
    def wheelEvent(self, event):
        if not self.label.geometry().contains(event.pos()):
            return

        if not self.wheel_zoom_tool.isChecked():
            return

        zoom_in = event.angleDelta().y() > 0
        zoom_factor = self.zoom_factor if zoom_in else (1 / self.zoom_factor)
        mouse_pos = event.pos()

        self.target_zoom = self.zoom * zoom_factor
        self.target_x = self.center_x + (mouse_pos.x() - self.label.width() / 2) / self.zoom
        self.target_y = self.center_y + (mouse_pos.y() - self.label.height() / 2) / self.zoom

        self.update_view()

    # ---------------- Panning ----------------
    def mousePressEvent(self, event):
        if self.label.geometry().contains(event.pos()):
            mouse_pos = event.pos()
            self.target_x = self.center_x + (mouse_pos.x() - self.label.width() / 2) / self.zoom
            self.target_y = self.center_y + (mouse_pos.y() - self.label.height() / 2) / self.zoom
            if event.button() == Qt.LeftButton:
                if self.drag_tool.isChecked():
                    self.dragging = True
                    self.last_mouse_pos = event.pos()
                    self.start_center_x = self.center_x
                    self.start_center_y = self.center_y
                if self.set_center_tool.isChecked():
                    self.update_view()
                if self.click_zoom_tool.isChecked():
                    self.target_zoom = self.zoom * self.zoom_factor
                    self.update_view()
            if event.button() == Qt.RightButton:
                if self.click_zoom_tool.isChecked():
                    self.target_zoom = self.zoom / self.zoom_factor
                    self.update_view()

    def mouseMoveEvent(self, event):
        if self.dragging and self.cached_image and self.drag_tool.isChecked():
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
        if event.button() == Qt.LeftButton and self.dragging and self.drag_tool.isChecked():
            self.dragging = False
            self.cached_image = QImage(self.label.pixmap())
            self.target_x, self.target_y = self.center_x, self.center_y
            self.render_fractal()

    # -------------- Window Actions -----------------
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
    viewer = FractalViewer()
    viewer.show()
    viewer.render_fractal()
    sys.exit(app.exec_())
