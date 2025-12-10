import sys
from datetime import datetime
import numpy as np
from collections import deque

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
    QPushButton, QComboBox, QFileDialog, QHBoxLayout, QDockWidget, QTabWidget,
    QFormLayout, QLineEdit, QSizePolicy, QRadioButton, QButtonGroup, QCheckBox,
    QPlainTextEdit
)
from PyQt5.QtGui import QPixmap, QPainter, QImage
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve

from palettes import palettes
from render import FullImageRenderer
from enums import Kernel, ColoringMode, EngineMode, Tools, Precisions
from utils import available_backends


class FractalViewer(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fractal Viewer")
        self.window_height = 900
        self.aspect_ratio = 16 / 9
        self.window_width = int(self.window_height * self.aspect_ratio)
        self.setGeometry(100, 100, self.window_width, self.window_height)

        # Main fractal display
        self.label = QLabel(self)
        self.label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.label.setScaledContents(True)
        self.label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

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
        self.renderer = None
        self.view_image = None
        self.cached_image = None
        self.anim_base = None

        # --- Progressive tile compositing state ---
        self.animation_active = False
        self.tile_queue = deque()         # items: (gen, x, y, QImage)
        self.tile_index = {}              # coalescing map: (gen,x,y,w,h)->index
        self.tile_flush_timer = QTimer(self)
        self.tile_flush_timer.setInterval(16)  # ~60 FPS
        self.tile_flush_timer.timeout.connect(self._flush_tile_queue)

        self.init_ui()

    def init_ui(self):
        # Central layout
        layout = QVBoxLayout()
        layout.addWidget(self.label)

        # ---------- Bottom controls ----------
        controls = QHBoxLayout()
        save_button = QPushButton("Save Image")
        save_button.clicked.connect(self.save_image)
        controls.addWidget(save_button)

        tools = QHBoxLayout()
        self.drag_tool = QCheckBox("Drag tool")
        self.drag_tool.setChecked(False)
        tools.addWidget(self.drag_tool)
        self.drag_tool.clicked.connect(lambda checked: checked and self.set_tools(Tools.Drag))

        self.click_zoom_tool = QCheckBox("Click-Zoom tool")
        self.click_zoom_tool.setChecked(False)
        tools.addWidget(self.click_zoom_tool)
        self.click_zoom_tool.clicked.connect(lambda checked: checked and self.set_tools(Tools.Click_zoom))

        self.wheel_zoom_tool = QCheckBox("Wheel-Zoom tool")
        self.wheel_zoom_tool.setChecked(False)
        tools.addWidget(self.wheel_zoom_tool)
        self.wheel_zoom_tool.clicked.connect(lambda checked: checked and self.set_tools(Tools.Wheel_zoom))

        self.set_center_tool = QCheckBox("Set-Center tool")
        self.set_center_tool.setChecked(False)
        tools.addWidget(self.set_center_tool)
        self.set_center_tool.clicked.connect(lambda checked: checked and self.set_tools(Tools.Set_center))

        layout.addLayout(tools)
        layout.addLayout(controls)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # ---------- Side Menu ----------
        self.side_menu = QDockWidget("Controls", self)
        self.side_menu.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.side_menu)
        self.side_menu.setFeatures(QDockWidget.NoDockWidgetFeatures)

        tabs = QTabWidget()
        self.side_menu.setWidget(tabs)

        # ---- Render tab ----
        render_tab = QWidget()
        render_layout = QFormLayout()

        self.kernel_input = QComboBox()
        backends = available_backends()
        self.kernel_input.addItems(sorted(backends))
        self.kernel_input.setCurrentText(backends[0])
        render_layout.addRow("Kernel: ", self.kernel_input)

        self.iter_input = QComboBox()
        self.iter_input.addItems(["100", "200", "500", "1000", "2000", "5000"])
        self.iter_input.setCurrentText("200")
        render_layout.addRow("Max Iterations: ", self.iter_input)

        self.samples_input = QComboBox()
        self.samples_input.addItems(["1", "2", "4", "8"])
        self.samples_input.setCurrentText("2")
        render_layout.addRow("Samples: ", self.samples_input)

        self.use_perturb_chk = QCheckBox("Use perturbation")
        self.use_perturb_chk.setChecked(False)
        render_layout.addRow(self.use_perturb_chk)

        self.perturb_order_input = QComboBox()
        self.perturb_order_input.addItems(["1", "2"])
        self.perturb_order_input.setCurrentText("2")
        render_layout.addRow("Perturbation order:", self.perturb_order_input)

        self.tile_render_check = QCheckBox("Tile rendering")
        self.tile_render_check.setChecked(False)
        render_layout.addRow(self.tile_render_check)

        self.tile_size_input = QLineEdit("256x256")
        render_layout.addRow("Tile size (WxH):", self.tile_size_input)

        apply_render_btn = QPushButton("Apply")
        apply_render_btn.clicked.connect(self.apply_render_settings)
        render_layout.addRow(apply_render_btn)

        render_tab.setLayout(render_layout)

        # ---- Palette tab ----
        palette_tab = QWidget()
        palette_layout = QFormLayout()

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

        # Mode toggles
        self.radio_exterior.toggled.connect(
            lambda checked: checked and self.change_coloring_mode(
                ColoringMode.EXTERIOR))
        self.radio_interior.toggled.connect(
            lambda checked: checked and self.change_coloring_mode(
                ColoringMode.INTERIOR))
        self.radio_hybrid.toggled.connect(
            lambda checked: checked and self.change_coloring_mode(
                ColoringMode.HYBRID))

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

        # ---- View tab ----
        view_tab = QWidget()
        view_layout = QFormLayout()

        self.center_x_input = QLineEdit(str(self.center_x))
        self.center_y_input = QLineEdit(str(self.center_y))
        view_layout.addRow("Center X: ", self.center_x_input)
        view_layout.addRow("Center Y: ", self.center_y_input)

        self.zoom_input = QLineEdit(str(self.zoom))
        view_layout.addRow("Zoom: ", self.zoom_input)

        self.zoom_factor_input = QLineEdit(str(self.zoom_factor))
        view_layout.addRow("Zoom Factor: ", self.zoom_factor_input)

        apply_view_btn = QPushButton("Apply")
        apply_view_btn.clicked.connect(self.apply_view_settings)
        view_layout.addRow(apply_view_btn)

        view_tab.setLayout(view_layout)

        tabs.addTab(render_tab, "Render")
        tabs.addTab(palette_tab, "Palette")
        tabs.addTab(view_tab, "View")

        # ---------- Log Dock ----------
        self.log_dock = QDockWidget("Log", self)
        self.log_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.log_dock.setFeatures(QDockWidget.NoDockWidgetFeatures)

        self.log_view = QPlainTextEdit(self)
        self.log_view.setReadOnly(True)
        self.log_view.setMaximumBlockCount(5000)
        self.log_view.setStyleSheet(
            "background: #0f0f10; color: #cfd2d6; font-family: Consolas, monospace; font-size: 11px;"
        )

        log_container = QWidget(self)
        log_v = QVBoxLayout(log_container)
        log_v.setContentsMargins(6, 6, 6, 6)

        log_controls = QHBoxLayout()
        log_controls.setSpacing(8)
        btn_clear = QPushButton("Clear")
        btn_copy = QPushButton("Copy")
        self.log_autoscroll_chk = QCheckBox("Auto-scroll")
        self.log_autoscroll_chk.setChecked(True)

        btn_clear.clicked.connect(lambda: self.log_view.clear())

        def copy_all():
            self.log_view.selectAll()
            self.log_view.copy()
            self.log_view.moveCursor(self.log_view.textCursor().End)

        btn_copy.clicked.connect(copy_all)

        log_controls.addWidget(btn_clear)
        log_controls.addWidget(btn_copy)
        log_controls.addStretch(1)
        log_controls.addWidget(self.log_autoscroll_chk)

        log_v.addLayout(log_controls)
        log_v.addWidget(self.log_view)

        self.log_dock.setWidget(log_container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.log_dock)
        self.splitDockWidget(self.side_menu, self.log_dock, Qt.Orientation.Vertical)

    # ----------------------- Rendering -----------------------
    def render_fractal(self):
        width = self.label.width()
        height = self.label.height()
        if width == 0 or height == 0:
            return

        if self.renderer is None:
            self.renderer = FullImageRenderer(
                width, height,
                palettes[self.default_palette_name],
                kernel=Kernel.AUTO,
                max_iter=200, samples=2,
                coloring_mode=ColoringMode.EXTERIOR,
                precision=np.float32
            )
            self.renderer.image_updated.connect(self.update_image)
            self.renderer.tile_ready.connect(self._on_tile_ready)
            self.renderer.log_text.connect(self.log)

        scale = 1.0 / self.zoom
        min_x = self.center_x - (width / 2) * scale
        max_x = self.center_x + (width / 2) * scale
        min_y = self.center_y - (height / 2) * scale
        max_y = self.center_y + (height / 2) * scale

        # Precision
        if self.zoom > 1e14:
            precision_mode = Precisions.Arbitrary
        elif self.zoom > 1e6:
            precision_mode = Precisions.Double
        else:
            precision_mode = Precisions.Single
        self.renderer.set_precision(precision_mode)

        self.log(
            f"Render frame: size={width}x{height}, "
            f"center=({self.center_x:.6g},{self.center_y:.6g}), "
            f"zoom={self.zoom:.6g}, precision={precision_mode.name}."
        )

        self._clear_tile_queue()
        self.renderer.render_frame(min_x, max_x, min_y, max_y)

    # ----------------------- Palette & Mode -----------------------
    def change_coloring_mode(self, mode):
        if self.renderer:
            self.renderer.set_coloring_mode(mode)
            self.log(f"Set coloring mode to {mode.name}.")
            self.render_fractal()

    def change_exter_palette(self, name):
        if self.renderer:
            self.renderer.set_exter_palette(name)
            self.log(f"Set Exterior palette to {name}.")
            self.render_fractal()

    def change_inter_palette(self, name):
        if self.renderer:
            self.renderer.set_inter_palette(name)
            self.log(f"Set Interior palette to {name}.")
            self.render_fractal()

    def apply_render_settings(self):
        """
        Apply engine/kernel/quality settings and trigger a new render.
        Keeps the tile-queue pipeline intact.
        """
        # Ensure renderer exists
        if self.renderer is None:
            # Lazily create the renderer with current viewport
            self.render_fractal()
            if self.renderer is None:
                self.log(
                    "Renderer was not created; aborting apply_render_settings.")
                return

        # --- Max iterations & samples ---
        try:
            max_iter = int(self.iter_input.currentText())
        except Exception:
            max_iter = self.renderer.max_iter
        try:
            samples = int(self.samples_input.currentText())
        except Exception:
            samples = self.renderer.samples

        if self.renderer.max_iter != max_iter:
            self.renderer.set_max_iter(max_iter)
            self.log(f"Set max_iter to {max_iter}.")

        if self.renderer.samples != samples:
            self.renderer.set_samples(samples)
            self.log(f"Set samples to {samples}.")

        # --- Perturbation settings ---
        use_perturb = self.use_perturb_chk.isChecked()
        try:
            perturb_order = int(self.perturb_order_input.currentText())
        except Exception:
            perturb_order = 2
        self.renderer.set_use_perturb(use_perturb, order=perturb_order,
                                      thresh=1e-6, hp_dps=160)
        self.log(
            f"Perturbation: {'ON' if use_perturb else 'OFF'}, order={perturb_order}, thresh=1e-6, hp_dps=160.")

        # --- Engine mode (FULL_FRAME vs TILED) ---
        if self.tile_render_check.isChecked():
            # Parse tile size safely
            try:
                tw, th = map(int,
                             self.tile_size_input.text().lower().replace(' ',
                                                                         '').split(
                                 'x'))
                if tw <= 0 or th <= 0:
                    raise ValueError("Tile size must be positive.")
            except Exception as e:
                self.log(
                    f"Error in tile size input: {e}. Falling back to 256x256.")
                tw, th = 256, 256
            self.renderer.set_engine_mode(EngineMode.TILED, tile_w=tw,
                                          tile_h=th)
            self.log(f"Engine: TILED ({tw}x{th}).")
        else:
            self.renderer.set_engine_mode(EngineMode.FULL_FRAME)
            self.log("Engine: FULL_FRAME.")

        # --- Kernel backend ---
        kernel_str = self.kernel_input.currentText().upper()
        if kernel_str == "OPENCL":
            new_kernel = Kernel.OPENCL
        elif kernel_str == "CUDA":
            new_kernel = Kernel.CUDA
        elif kernel_str == "CPU":
            new_kernel = Kernel.CPU
        else:
            self.log(f"Incorrect Kernel selected: {kernel_str}.")
            raise ValueError("Incorrect Kernel")

        if self.renderer.kernel != new_kernel:
            self.renderer.set_kernel(new_kernel)
            self.log(f"Kernel set to {kernel_str}.")

        # Trigger a new render with current viewport
        self.render_fractal()

    # ----------------------- View tab actions -----------------------
    def apply_view_settings(self):
        try:
            self.center_x = float(self.center_x_input.text())
            self.center_y = float(self.center_y_input.text())
            self.zoom = float(self.zoom_input.text())
            self.zoom_factor = float(self.zoom_factor_input.text())
            self.log(
                f"View updated: center=({self.center_x:.6g},{self.center_y:.6g}), "
                f"zoom={self.zoom:.6g}, zoom_factor={self.zoom_factor:.3g}."
            )
            self.render_fractal()
        except ValueError as e:
            self.log(f"Error in view inputs: {e}")

    def update_view_tab_fields(self):
        if hasattr(self, "center_x_input") and hasattr(self, "center_y_input") and hasattr(self, "zoom_input"):
            self.center_x_input.setText(str(self.center_x))
            self.center_y_input.setText(str(self.center_y))
            self.zoom_input.setText(str(self.zoom))

    def log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{timestamp}] {msg}"
        self.log_view.appendPlainText(line)
        if self.log_autoscroll_chk.isChecked():
            self.log_view.moveCursor(self.log_view.textCursor().End)

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
        self.log(f"Selected tool {tool.name}.")

    # ----------------------- Save Image / Updates -----------------------
    def save_image(self):
        if self.view_image:
            path, _ = QFileDialog.getSaveFileName(
                self, "Save Image",
                f"mandelbrot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png",
                "PNG Files (*.png)"
            )
            if path:
                self.view_image.save(path)
                self.log(f"Saved image to {path}.")

    def update_image(self, image: QImage):
        """
        Full-frame updates from renderer:
        - FULL_FRAME engine: normal final frame.
        - TILED engine: usually final frame when tiles are done.
        """
        self.log("Image updated from renderer.")

        # Guard: label not yet sized or image null
        if self.label.width() <= 0 or self.label.height() <= 0 or image is None or image.isNull():
            self.view_image = image
            self.label.setPixmap(QPixmap.fromImage(image))
            return

        # Set new image to cache
        self.cached_image = image

        # If we’re idle (no animation in progress), show it now
        if not (self.animation_timer and self.animation_timer.isActive()):
            self._fade_in_image(self.cached_image)
            self.view_image = self.cached_image
            self.cached_image = None
        else:
            self.final_render_pending = True

    def _fade_in_image(self, image: QImage):
        # If label is not sized yet, set directly
        if self.label.width() <= 0 or self.label.height() <= 0 or image is None or image.isNull():
            self.view_image = image
            self.label.setPixmap(QPixmap.fromImage(image))
            return

        old_image = self.view_image
        new_image = image
        width, height = self.label.width(), self.label.height()

        anim = QPropertyAnimation(self, b"dummy")  # dummy property
        anim.setDuration(400)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Linear)

        def update_blend(value):
            final_pixmap = QPixmap(self.label.size())
            if final_pixmap.isNull():
                self.label.setPixmap(QPixmap.fromImage(new_image))
                return
            final_pixmap.fill(Qt.GlobalColor.black)
            painter = QPainter(final_pixmap)

            if old_image and not old_image.isNull():
                old_pixmap = QPixmap.fromImage(old_image).scaled(
                    width, height,
                    Qt.AspectRatioMode.KeepAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                painter.setOpacity(1.0 - value)
                painter.drawPixmap(0, 0, old_pixmap)

            new_pixmap = QPixmap.fromImage(new_image).scaled(
                width, height,
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            painter.setOpacity(value)
            painter.drawPixmap(0, 0, new_pixmap)
            painter.end()

            self.label.setPixmap(final_pixmap)

        anim.valueChanged.connect(update_blend)

        def finalize():
            self.view_image = new_image
            self.label.setPixmap(QPixmap.fromImage(new_image))

        anim.finished.connect(finalize)
        self.fade_anim = anim
        self.fade_anim.start()

    # ----------------------- View change / Animation -----------------------
    def update_view(self):
        if not getattr(self, "view_image", None):
            return

        # Start the final render immediately (tiles start computing)
        self._start_final_render(self.target_x, self.target_y, self.target_zoom)

        # Fix an animation base for the whole animation
        self.anim_base = self.view_image.copy()

        # Setup animation
        self.start_x, self.start_y, self.start_zoom = self.center_x, self.center_y, self.zoom
        self.update_view_tab_fields()
        self.phase = 0 if self.target_zoom > self.zoom else 1
        self.current_step = 0

        self.animation_active = True
        if self.animation_timer is None:
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self._perform_anim_step)
        self.animation_timer.start(self.anim_interval)

    def _start_final_render(self, cx, cy, zoom):
        # Guard: avoid division by zero (e.g., Set-Center without zoom change)
        if zoom == 0:
            zoom = self.zoom

        scale = 1.0 / zoom
        min_x = cx - (self.label.width() / 2) * scale
        max_x = cx + (self.label.width() / 2) * scale
        min_y = cy - (self.label.height() / 2) * scale
        max_y = cy + (self.label.height() / 2) * scale

        # Precision
        if zoom > 1e14:
            precision_mode = Precisions.Arbitrary
        elif zoom > 1e6:
            precision_mode = Precisions.Double
        else:
            precision_mode = Precisions.Single
        self.renderer.set_precision(precision_mode)

        self.log(
            f"Final render scheduled: center=({cx:.6g},{cy:.6g}), "
            f"zoom={zoom:.6g}, precision={precision_mode.name}."
        )

        self._clear_tile_queue()
        self.renderer.render_frame(min_x, max_x, min_y, max_y)

    def _perform_anim_step(self):
        self.current_step += 1

        if self.current_step <= self.animation_steps // 2:
            t = self.current_step / (self.animation_steps // 2)
        else:
            t = (self.current_step - (self.animation_steps // 2)) / (
                        self.animation_steps // 2)
        eased_t = t * t * (3 - 2 * t)

        # Always render from the fixed base captured at animation start
        if self.anim_base is not None:
            base_pixmap = QPixmap.fromImage(self.anim_base)
            if self.phase == 0:
                self._pan_step(eased_t, base_pixmap)
            elif self.phase == 1:
                self._zoom_step(eased_t, base_pixmap)

        # Phase change at midpoint
        if self.current_step == self.animation_steps // 2:
            if self.phase == 0:
                self.center_x, self.center_y = self.target_x, self.target_y
                self.update_view_tab_fields()
                self.phase = 1
            elif self.phase == 1:
                self.zoom = self.target_zoom
                self.update_view_tab_fields()
                self.phase = 0
            self.anim_base = self.label.pixmap().toImage().copy()

        # Animation complete
        if self.current_step >= self.animation_steps:
            self.animation_timer.stop()
            self.center_x, self.center_y, self.zoom = self.target_x, self.target_y, self.target_zoom
            self.update_view_tab_fields()

            self.animation_active = False

            # Clear transient animation base
            self.view_image = self.label.pixmap().toImage().copy()
            self.anim_base = None
            if self.final_render_pending:
                self.final_render_pending = False
                self._fade_in_image(self.cached_image)
                self.view_image = self.cached_image
                self.cached_image = None

    def _pan_step(self, eased_t, pixmap):
        interp_x = self.start_x + (self.target_x - self.start_x) * eased_t
        interp_y = self.start_y + (self.target_y - self.start_y) * eased_t
        dx = int((interp_x - self.start_x) * self.zoom)
        dy = int((interp_y - self.start_y) * self.zoom)

        final_pixmap = QPixmap(self.label.size())
        final_pixmap.fill(Qt.GlobalColor.black)
        painter = QPainter(final_pixmap)
        painter.drawPixmap(-dx, -dy, pixmap)
        painter.end()
        self.label.setPixmap(final_pixmap)

        self.center_x = interp_x
        self.center_y = interp_y
        self.update_view_tab_fields()

    def _zoom_step(self, eased_t, pixmap: QPixmap):
        interp_zoom = self.start_zoom + (self.target_zoom - self.start_zoom) * eased_t
        scale_factor = interp_zoom / self.start_zoom

        scaled_width = int(self.label.width() * scale_factor)
        scaled_height = int(self.label.height() * scale_factor)
        scaled_pixmap = pixmap.scaled(
            scaled_width, scaled_height,
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )

        dx = int(self.label.width() / 2 - scaled_pixmap.width() / 2)
        dy = int(self.label.height() / 2 - scaled_pixmap.height() / 2)

        final_pixmap = QPixmap(self.label.size())
        final_pixmap.fill(Qt.GlobalColor.black)
        painter = QPainter(final_pixmap)
        painter.drawPixmap(dx, dy, scaled_pixmap)
        painter.end()
        self.label.setPixmap(final_pixmap)

        self.zoom = interp_zoom
        self.update_view_tab_fields()

    # ----------------------- Input handling -----------------------
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

    def mousePressEvent(self, event):
        if self.label.geometry().contains(event.pos()):
            mouse_pos = event.pos()
            self.target_x = self.center_x + (mouse_pos.x() - self.label.width() / 2) / self.zoom
            self.target_y = self.center_y + (mouse_pos.y() - self.label.height() / 2) / self.zoom

            if event.button() == Qt.MouseButton.LeftButton:
                if self.drag_tool.isChecked():
                    self.dragging = True
                    self.last_mouse_pos = event.pos()
                    self.start_center_x = self.center_x
                    self.start_center_y = self.center_y
                if self.set_center_tool.isChecked():
                    # Ensure zoom stays unchanged when only center changes
                    self.target_zoom = self.zoom
                    self.update_view()
                if self.click_zoom_tool.isChecked():
                    self.target_zoom = self.zoom * self.zoom_factor
                    self.update_view()

            if event.button() == Qt.MouseButton.RightButton:
                if self.click_zoom_tool.isChecked():
                    self.target_zoom = self.zoom / self.zoom_factor
                    self.update_view()

    def mouseMoveEvent(self, event):
        if self.dragging and self.view_image and self.drag_tool.isChecked():
            dx = event.x() - self.last_mouse_pos.x()
            dy = event.y() - self.last_mouse_pos.y()

            current_center_x = self.start_center_x - dx / self.zoom
            current_center_y = self.start_center_y - dy / self.zoom

            pixmap = QPixmap.fromImage(self.view_image)
            scaled_pixmap = pixmap.scaled(
                self.label.width(),
                self.label.height(),
                Qt.AspectRatioMode.KeepAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )

            final_pixmap = QPixmap(self.label.size())
            final_pixmap.fill(Qt.GlobalColor.black)
            painter = QPainter(final_pixmap)
            painter.drawPixmap(dx, dy, scaled_pixmap)
            painter.end()

            self.label.setPixmap(final_pixmap)

            self.center_x = current_center_x
            self.center_y = current_center_y
            self.update_view_tab_fields()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.dragging and self.drag_tool.isChecked():
            self.dragging = False
            self.target_x, self.target_y = self.center_x, self.center_y
            self.target_zoom = self.zoom
            self.render_fractal()

    # ----------------------- Window actions -----------------------
    def resizeEvent(self, event):
        if not hasattr(self, "resize_timer"):
            self.resize_timer = QTimer()
            self.resize_timer.setSingleShot(True)
            self.resize_timer.timeout.connect(self.render_fractal)

        if self.label.width() > 0 and self.label.height() > 0:
            if self.renderer is not None:
                self._clear_tile_queue()
                self.renderer.set_image_size(self.label.width(), self.label.height())
            self.log(f"Viewport resized: {self.label.width()}x{self.label.height()} (renderer size updated).")

            if self.view_image:
                scaled_pixmap = QPixmap.fromImage(self.view_image).scaled(
                    self.label.size(),
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.label.setPixmap(scaled_pixmap)

        self.resize_timer.start(250)
        super().resizeEvent(event)

    def closeEvent(self, event):
        if self.renderer:
            self.renderer.stop()
        event.accept()

    # ----------------------- Tile queue plumbing -----------------------
    def _on_tile_ready(self, gen: int, x: int, y: int, tile_qimg: QImage):
        key = (gen, x, y, tile_qimg.width(), tile_qimg.height())
        if key in self.tile_index:
            idx = self.tile_index[key]
            self.tile_queue[idx] = (gen, x, y, tile_qimg)
        else:
            self.tile_queue.append((gen, x, y, tile_qimg))
            self.tile_index[key] = len(self.tile_queue) - 1

        if not self.animation_active and not self.tile_flush_timer.isActive():
            self.tile_flush_timer.start()

    def _flush_tile_queue(self):
        if self.animation_active:
            return

        if self.cached_image is None:
            if self.view_image is not None:
                self.cached_image = self.view_image.copy()
            else:
                self.cached_image = QImage(self.label.width(), self.label.height(), QImage.Format_RGB32)
                self.cached_image.fill(Qt.GlobalColor.black)

        budget = min(16, len(self.tile_queue))
        painter = QPainter(self.cached_image)
        for _ in range(budget):
            gen, x, y, tile_qimg = self.tile_queue.popleft()
            key = (gen, x, y, tile_qimg.width(), tile_qimg.height())
            self.tile_index.pop(key, None)
            painter.drawImage(x, y, tile_qimg)
        painter.end()

        self.label.setPixmap(QPixmap.fromImage(self.cached_image))

        if not self.tile_queue:
            self.tile_flush_timer.stop()
            self.view_image = self.cached_image.copy()
            self.cached_image = None

    def _clear_tile_queue(self):
        self.tile_queue.clear()
        self.tile_index.clear()
        if self.tile_flush_timer.isActive():
            self.tile_flush_timer.stop()
        self.cached_image = None
        # composite_image stays; reinitialized on demand


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = FractalViewer()
    viewer.show()
    viewer.render_fractal()
    sys.exit(app.exec_())
