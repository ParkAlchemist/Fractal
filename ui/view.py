import sys
from datetime import datetime

from PySide6.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve
from PySide6.QtGui import QPixmap, QPainter, QImage, QTextCursor, QWheelEvent
from PySide6.QtWidgets import (
    QApplication, QMainWindow, QLabel, QVBoxLayout, QWidget,
    QPushButton, QComboBox, QFileDialog, QHBoxLayout, QDockWidget, QTabWidget,
    QFormLayout, QLineEdit, QSizePolicy, QRadioButton, QButtonGroup, QCheckBox,
    QPlainTextEdit
)

from api.render_api import RenderAPI
from adapters.qt_render_bridge import QtRenderBridge
from coloring.palettes import palettes
from rendering.service import RenderService
from utils.enums import BackendType, ColoringMode, EngineMode, Tools, PrecisionMode
from utils.backend_helpers import available_backends
from ui.view_components import (AspectRatioContainer, ViewState,
                                RenderSizePolicy, TileCompositor)


# =============================================================================
# Main Window
# =============================================================================
class FractalViewer(QMainWindow):
    # ---------- Construction & UI wiring ----------
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Fractal Viewer")
        self.aspect_ratio = 16 / 9
        self.window_height = 900
        self.window_width = int(self.window_height * self.aspect_ratio)
        self.setGeometry(100, 100, self.window_width, self.window_height)

        # World/view state
        self.state = ViewState(center_x=-0.5, center_y=0.0, zoom=250.0,
                               aspect_ratio=self.aspect_ratio, zoom_factor=2.0)

        # Render policy (target size / lod)
        self.rsize = RenderSizePolicy(pitch_multiple=32)
        self.rsize.target_quality = "1080p"

        # Images
        self.view_image: QImage | None = None
        self.cached_image: QImage | None = None

        # Animation / interaction state
        self.animation_timer: QTimer | None = None
        self.animation_steps = 40
        self.current_step = 0
        self.phase = 1
        self.start_x = self.start_y = self.start_zoom = 0.0
        self.target_x = self.target_y = 0.0
        self.target_zoom = self.state.zoom
        self.final_render_pending = False
        self.anim_fps = 60
        self.anim_interval = int((1 / self.anim_fps) * 1000)
        self.animation_active = False

        # Drag state
        self.dragging = False
        self.last_mouse_pos = None
        self.start_center_x = None
        self.start_center_y = None

        # UI defaults
        self.default_palette_name = "Classic"

        # Display label (scaled & policy) inside an aspect ratio container
        self.display = QLabel(self)
        self.display.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.display.setScaledContents(True)
        self.display.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Ignored)
        self.display_container = AspectRatioContainer(self.aspect_ratio, self.display, parent=self)
        self.display_container.set_padding(8, 0, 8, 0)

        # Tile compositor (handles queue & scaled painting)
        self.stats_label = QLabel("Tiles: 0 \nTiles/sec: 0.0")
        self.stats_label.setStyleSheet("color: #AAB; padding: 2px;")
        self.compositor = TileCompositor(self.display, self.stats_label, parent=self)
        self.compositor.tile_queue_empty.connect(self._final_render)

        # Initialize RenderAPI
        render_w, render_h = self.rsize.compute_target_size(self.aspect_ratio)
        service = RenderService(width=render_w, height=render_h, palette=palettes["Classic"])
        self.api = RenderAPI(service)

        # Bridge: Qt signals for the UI
        self.bridge = QtRenderBridge(self.api, parent=self)
        self.bridge.image_updated.connect(self.update_image)
        self.bridge.tile_ready.connect(self._on_tile_ready)
        self.bridge.log_text.connect(self.log)

        # UI sets
        self._build_ui()

        # Initial render
        self.apply_default_settings()
        self.api.set_view(self.state.center_x, self.state.center_y, self.state.zoom)
        self._start_render()

    def _build_ui(self):
        # ----- Central layout -----
        layout = QVBoxLayout()
        layout.setContentsMargins(16, 8, 12, 8)
        layout.addWidget(self.display_container)

        # Bottom controls
        controls = QHBoxLayout()
        save_button = QPushButton("Save Image")
        save_button.clicked.connect(self.save_image)
        controls.addWidget(save_button)

        tools = QHBoxLayout()
        self.drag_tool = QCheckBox("Drag tool")
        self.drag_tool.setChecked(False)
        self.click_zoom_tool = QCheckBox("Click-Zoom tool")
        self.click_zoom_tool.setChecked(False)
        self.wheel_zoom_tool = QCheckBox("Wheel-Zoom tool")
        self.wheel_zoom_tool.setChecked(False)
        self.set_center_tool = QCheckBox("Set-Center tool")
        self.set_center_tool.setChecked(False)

        self.drag_tool.clicked.connect(lambda checked: checked and self.set_tools(Tools.Drag))
        self.click_zoom_tool.clicked.connect(lambda checked: checked and self.set_tools(Tools.Click_zoom))
        self.wheel_zoom_tool.clicked.connect(lambda checked: checked and self.set_tools(Tools.Wheel_zoom))
        self.set_center_tool.clicked.connect(lambda checked: checked and self.set_tools(Tools.Set_center))

        for w in (self.drag_tool, self.click_zoom_tool, self.wheel_zoom_tool, self.set_center_tool):
            tools.addWidget(w)

        layout.addLayout(tools)
        layout.addLayout(controls)

        container = QWidget()
        container.setLayout(layout)
        self.setCentralWidget(container)

        # ----- Side dock: Controls -----
        self.side_menu = QDockWidget("Controls", self)
        self.side_menu.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.side_menu.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
        tabs = QTabWidget()
        self.side_menu.setWidget(tabs)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.side_menu)

        # Render tab
        render_tab = QWidget()
        render_layout = QFormLayout()
        self.kernel_input = QComboBox()
        backs = available_backends()
        self.kernel_input.addItems(sorted(backs))
        self.kernel_input.setCurrentText(backs[0])
        render_layout.addRow("Kernel: ", self.kernel_input)

        self.res_input = QComboBox()
        self.res_input.addItems(["2160p", "1440p", "1080p", "720p", "480p", "360p"])
        self.res_input.setCurrentText(self.rsize.target_quality)
        render_layout.addRow("Resolution: ", self.res_input)

        self.iter_input = QComboBox()
        self.iter_input.addItems(["100", "200", "500", "1000", "2000", "5000"])
        self.iter_input.setCurrentText("200")
        render_layout.addRow("Max Iterations: ", self.iter_input)

        self.samples_input = QComboBox()
        self.samples_input.addItems(["1", "2", "4", "8"])
        self.samples_input.setCurrentText("2")
        render_layout.addRow("Samples: ", self.samples_input)

        self.tile_render_check = QCheckBox("Tile rendering")
        self.tile_render_check.setChecked(False)
        render_layout.addRow(self.tile_render_check)

        self.tile_size_input = QLineEdit("512x512")
        render_layout.addRow("Tile size (WxH):", self.tile_size_input)

        # Adaptive tiling options
        self.adaptive_enabled = True
        self.adaptive_opts = {
            "min_tile": 32,
            "max_tile": 256,
            "target_ms": 12.0,
            "max_depth": 4,
            "sample_stride": 8,
            "parallel": False,
            "max_workers": 0,  # 0 -> auto
        }

        self.adaptive_enable_chk = QCheckBox("Enable Adaptive Tiling")
        self.adaptive_enable_chk.setChecked(True)
        self.adaptive_enable_chk.toggled.connect(lambda b: setattr(self, "adaptive_enabled", bool(b)))
        render_layout.addRow(self.adaptive_enable_chk)

        self.min_tile_input = QLineEdit(str(self.adaptive_opts["min_tile"]))
        self.max_tile_input = QLineEdit(str(self.adaptive_opts["max_tile"]))
        render_layout.addRow("Min tile (px):", self.min_tile_input)
        render_layout.addRow("Max tile (px):", self.max_tile_input)

        self.target_ms_input = QLineEdit(str(self.adaptive_opts["target_ms"]))
        self.max_depth_input = QLineEdit(str(self.adaptive_opts["max_depth"]))
        self.sample_stride_input = QLineEdit(str(self.adaptive_opts["sample_stride"]))
        render_layout.addRow("Target time per tile (ms):", self.target_ms_input)
        render_layout.addRow("Max quadtree depth:", self.max_depth_input)
        render_layout.addRow("Sample stride:", self.sample_stride_input)

        self.parallel_chk = QCheckBox("Parallel (GPU)")
        self.parallel_chk.setChecked(self.adaptive_opts["parallel"])
        self.max_workers_input = QLineEdit(str(self.adaptive_opts["max_workers"]))
        render_layout.addRow(self.parallel_chk)
        render_layout.addRow("Max workers (0=auto):", self.max_workers_input)

        self.overlay_chk = QCheckBox("Show tile overlay")
        self.overlay_chk.setChecked(True)
        self.overlay_chk.toggled.connect(lambda b: self.compositor.set_overlay_enabled(bool(b)))
        render_layout.addRow(self.overlay_chk)

        apply_render_btn = QPushButton("Apply")
        apply_render_btn.clicked.connect(self.apply_render_settings)
        render_layout.addRow(apply_render_btn)

        render_tab.setLayout(render_layout)

        # Palette tab
        palette_tab = QWidget()
        palette_layout = QFormLayout()
        self.coloring_mode_group = QButtonGroup(self)
        self.radio_exterior = QRadioButton("Exterior")
        self.radio_interior = QRadioButton("Interior")
        self.radio_hybrid = QRadioButton("Hybrid")
        self.radio_exterior.setChecked(True)
        self.coloring_mode_group.addButton(self.radio_exterior)
        self.coloring_mode_group.addButton(self.radio_interior)
        self.coloring_mode_group.addButton(self.radio_hybrid)
        palette_layout.addRow("Coloring Mode: ", self.radio_exterior)
        palette_layout.addRow("", self.radio_interior)
        palette_layout.addRow("", self.radio_hybrid)

        self.radio_exterior.toggled.connect(lambda checked: checked and self.change_coloring_mode(ColoringMode.EXTERIOR))
        self.radio_interior.toggled.connect(lambda checked: checked and self.change_coloring_mode(ColoringMode.INTERIOR))
        self.radio_hybrid.toggled.connect(lambda checked: checked and self.change_coloring_mode(ColoringMode.HYBRID))

        self.exter_palette_input = QComboBox()
        self.exter_palette_input.addItems(palettes.keys())
        self.exter_palette_input.setCurrentText(self.default_palette_name)
        self.exter_palette_input.currentTextChanged.connect(self.apply_palette)
        palette_layout.addRow("Exterior Palette: ", self.exter_palette_input)

        self.inter_palette_input = QComboBox()
        self.inter_palette_input.addItems(palettes.keys())
        self.inter_palette_input.setCurrentText(self.default_palette_name)
        self.inter_palette_input.currentTextChanged.connect(self.apply_palette)
        palette_layout.addRow("Interior Palette: ", self.inter_palette_input)

        palette_tab.setLayout(palette_layout)

        # View tab
        view_tab = QWidget()
        view_layout = QFormLayout()
        self.aspect_ratio_input = QComboBox()
        self.aspect_ratio_input.addItems(["16:9", "4:3", "1:1"])
        self.aspect_ratio_input.setCurrentText("16:9")
        view_layout.addRow("Aspect Ratio: ", self.aspect_ratio_input)

        self.center_x_input = QLineEdit(str(self.state.center_x))
        self.center_y_input = QLineEdit(str(self.state.center_y))
        self.zoom_input = QLineEdit(str(self.state.zoom))
        self.zoom_factor_input = QLineEdit(str(self.state.zoom_factor))
        view_layout.addRow("Center X: ", self.center_x_input)
        view_layout.addRow("Center Y: ", self.center_y_input)
        view_layout.addRow("Zoom: ", self.zoom_input)
        view_layout.addRow("Zoom Factor: ", self.zoom_factor_input)
        apply_view_btn = QPushButton("Apply")
        apply_view_btn.clicked.connect(self.apply_view_settings)
        view_layout.addRow(apply_view_btn)
        view_tab.setLayout(view_layout)

        tabs.addTab(render_tab, "Render")
        tabs.addTab(palette_tab, "Palette")
        tabs.addTab(view_tab, "View")

        # Log dock
        self.log_dock = QDockWidget("Log", self)
        self.log_dock.setAllowedAreas(Qt.DockWidgetArea.RightDockWidgetArea)
        self.log_dock.setFeatures(QDockWidget.DockWidgetFeature.NoDockWidgetFeatures)
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
            self.log_view.moveCursor(self.log_view.textCursor().MoveOperation.End)

        btn_copy.clicked.connect(copy_all)
        log_controls.addWidget(btn_clear)
        log_controls.addWidget(btn_copy)
        log_controls.addStretch(1)
        log_controls.addWidget(self.log_autoscroll_chk)
        log_v.addLayout(log_controls)
        log_v.addWidget(self.log_view)
        log_v.addWidget(self.stats_label)
        self.log_dock.setWidget(log_container)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.log_dock)
        self.splitDockWidget(self.side_menu, self.log_dock, Qt.Orientation.Vertical)

    # ---------- Palette/View/Render setting handlers ----------
    def change_coloring_mode(self, mode):
        pass

    def apply_palette(self):
        exter = self._get_combo(self.exter_palette_input, 'Classic')
        inter = self._get_combo(self.inter_palette_input, 'Classic')
        self.api.set_palettes(exter, inter)
        self.log(f"Applied palettes: {exter} exterior, {inter} interior.")
        self._start_render()

    def apply_default_settings(self):
        (self.api.configure()
         .resolution(self.rsize.target_quality)
         .max_iter(int(self._get_combo(self.iter_input, 200)))
         .samples(int(self._get_combo(self.samples_input, 2)))
         .engine_mode(EngineMode.FULL_FRAME)
         .backend(BackendType.AUTO)
         .apply())

    def apply_view_settings(self):
        try:
            ar = str(self.aspect_ratio_input.currentText())
            nom, denom = map(int, ar.split(':'))
            new_ar = nom / denom
            prev_ar = self.state.aspect_ratio
            prev_w, prev_h = self.rsize.compute_target_size(prev_ar)
            self.aspect_ratio = new_ar
            self.state.aspect_ratio = new_ar

            # Update container immediately
            if self.display_container:
                self.display_container.set_aspect_ratio(self.aspect_ratio)

            # Compute how the target render width changes under the same preset height
            target_w_old, target_h_old = self.rsize.compute_target_size(new_ar)

            # Preserve FOV across aspect ratio change by adjusting zoom to width delta
            if (prev_w, prev_h) != (target_w_old, target_h_old):
                self._adjust_zoom_for_resolution_change(prev_w, prev_h,
                                                        target_w_old,
                                                        target_h_old)
                self.log(f"Adjusted zoom to {self.state.zoom:.6g} for aspect {prev_ar:.4g} → {new_ar:.4g}.")

            # Fetch new view inputs
            center_x = self._get_float(self.center_x_input, self.state.center_x)
            center_y = self._get_float(self.center_y_input, self.state.center_y)
            zoom = self._get_float(self.zoom_input, self.state.zoom)
            zoom_factor = self._get_float(self.zoom_factor_input, self.state.zoom_factor)

            # Update view state
            self.state.center_x = center_x
            self.state.center_y = center_y
            self.state.zoom = zoom
            self.state.zoom_factor = zoom_factor

            # Apply view changes to renderer
            self.api.set_view(center_x, center_y, zoom)

            self.log(
                f"View updated: center=({self.state.center_x:.6g},{self.state.center_y:.6g}), "
                f"zoom={self.state.zoom:.6g}, zoom_factor={self.state.zoom_factor:.3g}."
            )
            self._start_render()
        except ValueError as e:
            self.log(f"Error in view inputs: {e}")

    def apply_render_settings(self):

        # Resolution preset
        prev_w, prev_h = self.rsize.compute_target_size(self.aspect_ratio)
        self.rsize.target_quality = self.res_input.currentText()
        new_w, new_h = self.rsize.compute_target_size(self.aspect_ratio)

        if (new_w, new_h) != (prev_w, prev_h):
            self._adjust_zoom_for_resolution_change(prev_w, prev_h, new_w, new_h)
            self.log(f"Resolution preset changed to {self.rsize.target_quality}.")
            self.state.set_frame_size(new_w, new_h)

        resolution = self._get_combo(self.res_input, '1080p')
        max_iter = int(self._get_combo(self.iter_input, 200))
        samples = int(self._get_combo(self.samples_input, 2))

        if self.tile_render_check.isChecked() and self.adaptive_enable_chk.isChecked():
            engine_mode = EngineMode.ADAPTIVE
            self._apply_adaptive_settings()
        elif self.tile_render_check.isChecked():
            engine_mode = EngineMode.TILED
        else:
            engine_mode = EngineMode.FULL_FRAME

        try:
            tw, th = map(int, self.tile_size_input.text().lower().replace(" ", "").split('x'))
        except ValueError:
            tw, th = 256, 256

        kernel_str = self._get_combo(self.kernel_input, 'CPU').upper()
        if kernel_str == "OPENCL":
            backend = BackendType.OPENCL
        elif kernel_str == "CUDA":
            backend = BackendType.CUDA
        else:
            backend = BackendType.CPU

        (self.api.configure()
         .resolution(resolution)
         .max_iter(max_iter)
         .samples(samples)
         .engine_mode(engine_mode, tile_w=tw, tile_h=th)
         .adaptive_opts(self.adaptive_opts)
         .backend(backend)
         .apply())

        self._start_render()

    def _apply_adaptive_settings(self):
        self.adaptive_opts["min_tile"] = self._get_int(self.min_tile_input, self.adaptive_opts["min_tile"])
        self.adaptive_opts["max_tile"] = self._get_int(self.max_tile_input, self.adaptive_opts["max_tile"])
        self.adaptive_opts["target_ms"] = self._get_float(self.target_ms_input, self.adaptive_opts["target_ms"])
        self.adaptive_opts["max_depth"] = self._get_int(self.max_depth_input, self.adaptive_opts["max_depth"])
        self.adaptive_opts["sample_stride"] = self._get_int(self.sample_stride_input, self.adaptive_opts["sample_stride"])
        self.adaptive_opts["parallel"] = bool(self.parallel_chk.isChecked())
        mw = self._get_int(self.max_workers_input, self.adaptive_opts["max_workers"])
        self.adaptive_opts["max_workers"] = mw if mw >= 0 else 0
        self.log(f"Adaptive settings applied: {self.adaptive_opts}")

    # ---------- Rendering control ----------
    def _start_render(self):
        # Compute target render size based on preset & aspect
        render_w, render_h = self.rsize.compute_target_size(self.aspect_ratio)
        self.state.set_frame_size(render_w, render_h)

        self.log(
            f"Render frame: size={render_w}x{render_h} "
            f"(display={self.display.width()}x{self.display.height()}), "
            f"center=({self.state.center_x:.6g},{self.state.center_y:.6g}), "
            f"zoom={self.state.zoom:.6g}."
        )

        self.compositor.clear()
        self.api.set_view(self.state.center_x, self.state.center_y, self.state.zoom)

        if self.state.zoom > 1e6:
            precision = PrecisionMode.Double
        else:
            precision = PrecisionMode.Single

        self.api.set_precision(precision)

        self.api.start_async_render()

    # ---------- Renderer callbacks ----------
    def update_image(self, image: QImage, render_w: int, render_h: int):
        """
        Full-frame updates (both FULL_FRAME and TILED final frame).
        """
        self.state.set_frame_size(render_w, render_h)
        self.log(f"Image updated from renderer (frame={render_w}x{render_h}).")

        # Guard against premature sizing
        if self.display.width() <= 0 or self.display.height() <= 0 or image is None or image.isNull():
            self.view_image = image
            scaled = QPixmap.fromImage(image).scaled(
                max(1, self.display.width()),
                max(1, self.display.height()),
                Qt.AspectRatioMode.IgnoreAspectRatio,  # label already has the right aspect
                Qt.TransformationMode.SmoothTransformation
            )
            self.display.setPixmap(scaled)
            return

        # Cache & present (fade)
        self.cached_image = image
        if not (self.animation_timer and self.animation_timer.isActive()):
            if not self.tile_render_check.isChecked():
                self._fade_in_image(self.cached_image)
                self.view_image = self.cached_image
                self.cached_image = None
                # Keep compositor's committed copy in sync
                self.compositor.view_image = self.view_image
            else:
                self.final_render_pending = True
        else:
            self.final_render_pending = True

    def _on_tile_ready(self, gen: int, x: int, y: int, tile_img: QImage, frame_w: int, frame_h: int):
        # Route to compositor
        self.compositor.on_tile_ready(gen, x, y, tile_img, frame_w, frame_h)

    def _final_render(self):
        if self.final_render_pending:
            self.final_render_pending = False
            self.view_image = self.compositor.view_image
            self._fade_in_image(self.cached_image)
            self.view_image = self.cached_image
            self.cached_image = None
            self.compositor.view_image = self.view_image


    # ---------- Input & animation ----------
    def wheelEvent(self, event: QWheelEvent):
        local = self.display.mapFromGlobal(event.globalPosition().toPoint())
        if local.x() < 0 or local.y() < 0 or local.x() >= self.display.width() or local.y() >= self.display.height():
            return
        if not self.wheel_zoom_tool.isChecked():
            return

        lw, lh = self.display.width(), self.display.height()

        zoom_in = event.angleDelta().y() > 0
        factor = self.state.zoom_factor if zoom_in else (1 / self.state.zoom_factor)
        self.target_zoom = self.state.zoom * factor

        dx_world, dy_world = self.state.world_delta_from_label_px(local.x() - lw / 2, local.y() - lh / 2, lw, lh)
        self.target_x = self.state.center_x + dx_world
        self.target_y = self.state.center_y + dy_world

        self.tile_render_check.setChecked(True)
        self.update_view()

    def mousePressEvent(self, event):
        if not self.display.geometry().contains(event.pos()):
            return
        local = self.display.mapFrom(self, event.pos())
        lw, lh = self.display.width(), self.display.height()
        dx_world, dy_world = self.state.world_delta_from_label_px(local.x() - lw / 2, local.y() - lh / 2, lw, lh)
        self.target_x = self.state.center_x + dx_world
        self.target_y = self.state.center_y + dy_world

        if event.button() == Qt.MouseButton.LeftButton:
            if self.drag_tool.isChecked():
                self.dragging = True
                self.last_mouse_pos = local
                self.start_center_x = self.state.center_x
                self.start_center_y = self.state.center_y

            if self.set_center_tool.isChecked():
                self.target_zoom = self.state.zoom
                self.update_view()

            if self.click_zoom_tool.isChecked():
                self.target_zoom = self.state.zoom * self.state.zoom_factor
                self.update_view()

        if event.button() == Qt.MouseButton.RightButton and self.click_zoom_tool.isChecked():
            self.target_zoom = self.state.zoom / self.state.zoom_factor
            self.update_view()

    def mouseMoveEvent(self, event):
        if self.dragging and self.view_image and self.drag_tool.isChecked():
            local = self.display.mapFrom(self, event.pos())
            lw, lh = self.display.width(), self.display.height()

            dx_px = local.x() - self.last_mouse_pos.x()
            dy_px = local.y() - self.last_mouse_pos.y()
            dx_world, dy_world = self.state.world_delta_from_label_px(dx_px, dy_px, lw, lh)

            current_center_x = self.start_center_x - dx_world
            current_center_y = self.start_center_y - dy_world

            pixmap = QPixmap.fromImage(self.view_image).scaled(
                lw, lh,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            final_pixmap = QPixmap(self.display.size())
            final_pixmap.fill(Qt.GlobalColor.black)
            painter = QPainter(final_pixmap)
            painter.drawPixmap(dx_px, dy_px, pixmap)
            painter.end()
            self.display.setPixmap(final_pixmap)

            self.state.center_x = current_center_x
            self.state.center_y = current_center_y
            self._update_view_tab_fields()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton and self.dragging and self.drag_tool.isChecked():
            self.dragging = False
            self.target_x, self.target_y = self.state.center_x, self.state.center_y
            self.target_zoom = self.state.zoom
            # Commit the preview as the current view image
            self.view_image = self.display.pixmap().toImage().copy()
            self.compositor.view_image = self.view_image
            self._start_render()

    def update_view(self):
        if not getattr(self, "view_image", None):
            return

        # Trigger the final render at the target view immediately
        self.compositor.set_paused(True)

        # Fix an animation base for the whole animation
        self.anim_base = self.view_image.copy()

        # Setup animation
        self.start_x, self.start_y, self.start_zoom = self.state.center_x, self.state.center_y, self.state.zoom
        self._update_view_tab_fields()
        self.phase = 0 if self.target_zoom > self.state.zoom else 1
        self.current_step = 0
        self.animation_active = True
        if self.animation_timer is None:
            self.animation_timer = QTimer()
            self.animation_timer.timeout.connect(self._perform_anim_step)
        self.animation_timer.start(self.anim_interval)

    def _perform_anim_step(self):
        self.current_step += 1
        half = self.animation_steps // 2
        t = (self.current_step / half) if self.current_step <= half else ((self.current_step - half) / half)
        eased_t = t * t * (3 - 2 * t)

        if getattr(self, "anim_base", None) is not None:
            base_pixmap = QPixmap.fromImage(self.anim_base)

            if self.phase == 0:
                self._pan_step(eased_t, base_pixmap)
            elif self.phase == 1:
                self._zoom_step(eased_t, base_pixmap)

        # Change animation phase
        if self.current_step == half:
            if self.phase == 0:
                self.state.center_x, self.state.center_y = self.target_x, self.target_y
                self._update_view_tab_fields()
                self.phase = 1
            elif self.phase == 1:
                self.state.zoom = self.target_zoom
                self._update_view_tab_fields()
                self.phase = 0
            self.anim_base = self.display.pixmap().toImage().copy()

        # Animation done
        if self.current_step >= self.animation_steps:
            self.animation_timer.stop()
            self.state.center_x, self.state.center_y, self.state.zoom = self.target_x, self.target_y, self.target_zoom
            self._update_view_tab_fields()
            self.animation_active = False
            self.view_image = self.display.pixmap().toImage().copy()
            self.compositor.view_image = self.view_image

            self.compositor.set_paused(False)
            self.compositor.flush()
            self._start_render()

            if self.final_render_pending:
                self.final_render_pending = False
                self._fade_in_image(self.cached_image)
                self.view_image = self.cached_image
                self.compositor.view_image = self.view_image
                self.cached_image = None

    def _pan_step(self, eased_t, pixmap: QPixmap):
        lw, lh = self.display.width(), self.display.height()
        scaled_pixmap = pixmap.scaled(
            lw, lh,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        pxu_x, pxu_y = self.state.pixels_per_unit_xy(lw, lh)

        interp_x = self.start_x + (self.target_x - self.start_x) * eased_t
        interp_y = self.start_y + (self.target_y - self.start_y) * eased_t

        dx_px = int(round((interp_x - self.start_x) * pxu_x))
        dy_px = int(round((interp_y - self.start_y) * pxu_y))

        final_pixmap = QPixmap(self.display.size())
        final_pixmap.fill(Qt.GlobalColor.black)
        painter = QPainter(final_pixmap)
        painter.drawPixmap(-dx_px, -dy_px, scaled_pixmap)
        painter.end()
        self.display.setPixmap(final_pixmap)

        self.state.center_x = interp_x
        self.state.center_y = interp_y
        self._update_view_tab_fields()

    def _zoom_step(self, eased_t, pixmap: QPixmap):
        lw, lh = self.display.width(), self.display.height()
        interp_zoom = self.start_zoom + (self.target_zoom - self.start_zoom) * eased_t
        scale_factor = interp_zoom / self.start_zoom

        scaled_w = int(round(lw * scale_factor))
        scaled_h = int(round(lh * scale_factor))
        scaled_pixmap = pixmap.scaled(
            scaled_w, scaled_h,
            Qt.AspectRatioMode.IgnoreAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        dx = (lw - scaled_pixmap.width()) // 2
        dy = (lh - scaled_pixmap.height()) // 2

        final_pixmap = QPixmap(self.display.size())
        final_pixmap.fill(Qt.GlobalColor.black)
        painter = QPainter(final_pixmap)
        painter.drawPixmap(dx, dy, scaled_pixmap)
        painter.end()
        self.display.setPixmap(final_pixmap)

        self.state.zoom = interp_zoom
        self._update_view_tab_fields()

    # ---------- Helpers ------------
    @staticmethod
    def _get_int(widget, default):
        try:
            if hasattr(widget, "text"):
                return int(widget.text().strip())
            elif hasattr(widget, "currentText"):
                return int(widget.currentText().strip())
        except ValueError:
            return default

    @staticmethod
    def _get_float(widget, default):
        try:
            if hasattr(widget, "text"):
                return float(widget.text().strip())
            elif hasattr(widget, "currentText"):
                return float(widget.currentText().strip())
        except ValueError:
            return default

    @staticmethod
    def _get_combo(widget, default):
        value = widget.currentText().strip()
        return value if value else default

    # ---------- Utilities ----------
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

    def _fade_in_image(self, image: QImage):
        if self.display.width() <= 0 or self.display.height() <= 0 or image is None or image.isNull():
            self.view_image = image
            scaled = QPixmap.fromImage(image).scaled(
                max(1, self.display.width()),
                max(1, self.display.height()),
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.display.setPixmap(scaled)
            return

        old_image = self.view_image
        new_image = image
        width, height = self.display.width(), self.display.height()

        anim = QPropertyAnimation(self, b"dummy")
        anim.setDuration(400)
        anim.setStartValue(0.0)
        anim.setEndValue(1.0)
        anim.setEasingCurve(QEasingCurve.Type.Linear)

        def update_blend(value):
            final_pixmap = QPixmap(self.display.size())
            if final_pixmap.isNull():
                scaled = QPixmap.fromImage(new_image).scaled(
                    width, height,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                self.display.setPixmap(scaled)
                return
            final_pixmap.fill(Qt.GlobalColor.black)
            painter = QPainter(final_pixmap)
            if old_image and not old_image.isNull():
                old_pixmap = QPixmap.fromImage(old_image).scaled(
                    width, height,
                    Qt.AspectRatioMode.IgnoreAspectRatio,
                    Qt.TransformationMode.SmoothTransformation
                )
                painter.setOpacity(1.0 - value)
                painter.drawPixmap(0, 0, old_pixmap)
            new_pixmap = QPixmap.fromImage(new_image).scaled(
                width, height,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            painter.setOpacity(value)
            painter.drawPixmap(0, 0, new_pixmap)
            painter.end()
            self.display.setPixmap(final_pixmap)

        anim.valueChanged.connect(update_blend)

        def finalize():
            self.view_image = new_image
            scaled = QPixmap.fromImage(new_image).scaled(
                width, height,
                Qt.AspectRatioMode.IgnoreAspectRatio,
                Qt.TransformationMode.SmoothTransformation
            )
            self.display.setPixmap(scaled)
            self.compositor.view_image = self.view_image

        anim.finished.connect(finalize)
        self.fade_anim = anim
        self.fade_anim.start()

    def _update_view_tab_fields(self):
        if hasattr(self, "center_x_input") and hasattr(self, "center_y_input") and hasattr(self, "zoom_input"):
            self.center_x_input.setText(str(self.state.center_x))
            self.center_y_input.setText(str(self.state.center_y))
            self.zoom_input.setText(str(self.state.zoom))

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

    def log(self, msg: str):
        timestamp = datetime.now().strftime("%H:%M:%S.%f")[:-3]
        line = f"[{timestamp}] {msg}"
        self.log_view.appendPlainText(line)
        if self.log_autoscroll_chk.isChecked():
            self.log_view.moveCursor(QTextCursor.MoveOperation.End)

    def _current_render_size(self) -> tuple[int, int]:
        """
        Returns the most reliable current render size:
        - Prefer the latest frame size stored in state (from renderer callbacks).
        - Else compute from policy as a last resort.
        """
        if self.state.frame_w and self.state.frame_h:
            return int(self.state.frame_w), int(self.state.frame_h)
        return self.rsize.compute_target_size(self.aspect_ratio)

    def _adjust_zoom_for_resolution_change(self, old_w: int, old_h: int,
                                           new_w: int, new_h: int) -> None:
        """
        Adjust zoom so the world field of view stays the same when resolution changes.
        We preserve horizontal FOV: zoom' = zoom * (new_w / old_w).
        """
        old_w = max(1, int(old_w))
        new_w = max(1, int(new_w))
        factor = float(new_w) / float(old_w)
        self.state.zoom *= factor
        # reflect in UI
        if hasattr(self, "zoom_input"):
            self.zoom_input.setText(str(self.state.zoom))

    # ---------- Qt events ----------
    def resizeEvent(self, event):
        # No longer force renderer to label size; aspect container takes care of label geometry.
        super().resizeEvent(event)
        # Optionally: trigger a re-render after resize debounce
        if not hasattr(self, "resize_timer"):
            self.resize_timer = QTimer()
            self.resize_timer.setSingleShot(True)
            self.resize_timer.timeout.connect(self._start_render)
        self.resize_timer.start(250)

    def closeEvent(self, event):
        if self.api:
            self.api.stop_render()
        event.accept()


# =============================================================================
# Entrypoint
# =============================================================================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = FractalViewer()
    viewer.show()
