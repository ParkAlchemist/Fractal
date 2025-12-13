import time as _t
from dataclasses import dataclass
from collections import deque
from typing import Deque, Dict, Tuple, List, Optional

from PySide6.QtCore import Qt, QTimer
from PySide6.QtGui import QImage, QPainter, QPixmap, QResizeEvent
from PySide6.QtWidgets import QWidget, QLabel, QSizePolicy


# ---------- Aspect-ratio container ----------
class AspectRatioContainer(QWidget):
    """
    A container that keeps a single child (the display QLabel) sized to a fixed aspect ratio,
    centered within the available space. This removes in-image letterboxing: the label area
    itself matches the chosen aspect ratio.
    """
    def __init__(self, aspect_ratio: float, child: QLabel, parent=None):
        super().__init__(parent)
        self.aspect_ratio = float(aspect_ratio)
        self.child = child
        self._padding: Tuple[int, int, int, int] = (0, 0, 0, 0) # L, T, R, B
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

    def set_aspect_ratio(self, ar: float) -> None:
        self.aspect_ratio = float(ar)
        self.update()
        # Trigger geometry recompute
        self.resizeEvent(QResizeEvent(self.size(), self.size()))

    def set_padding(self, l: int, t: int, r: int, b: int) -> None:
        self._padding = (int(l), int(t), int(r), int(b))
        self.update()
        self.resizeEvent(QResizeEvent(self.size(), self.size()))

    def resizeEvent(self, event):
        W = max(1, self.width())
        H = max(1, self.height())
        L, T, R, B = self._padding
        Wp = max(1, W - (L + R))
        Hp = max(1, H - (T + B))

        ar = self.aspect_ratio

        if Wp / Hp >= ar:
            # Width too large: height saturates
            target_h = Hp
            target_w = int(round(Hp * ar))
        else:
            # Height too large: width saturates
            target_w = Wp
            target_h = int(round(Wp / ar))

        x = L + (Wp - target_w) // 2
        y = T + (Hp - target_h) // 2
        self.child.setGeometry(x, y, target_w, target_h)


# ---------- View state & math ----------
@dataclass
class ViewState:
    center_x: float = -0.5
    center_y: float = 0.0
    zoom: float = 250.0        # pixels per world-unit on the *render frame*
    aspect_ratio: float = 16/9
    zoom_factor: float = 2.0

    # Latest frame (render grid) size
    frame_w: Optional[int] = None
    frame_h: Optional[int] = None

    def set_frame_size(self, w: int, h: int) -> None:
        self.frame_w = int(max(1, w))
        self.frame_h = int(max(1, h))

    def pixels_per_unit_xy(self, label_w: int, label_h: int) -> Tuple[float, float]:
        """
        Returns (px_per_unit_x, px_per_unit_y) in label pixels.
        px/unit on the render grid is self.zoom by definition.
        We convert to label pixels by scaling with label_size / frame_size.
        """
        rw = float(max(1, self.frame_w or label_w))
        rh = float(max(1, self.frame_h or label_h))
        sx = float(label_w) / rw
        sy = float(label_h) / rh
        return self.zoom * sx, self.zoom * sy

    def world_delta_from_label_px(self, dx_px: float, dy_px: float, label_w: int, label_h: int) -> Tuple[float, float]:
        pxu_x, pxu_y = self.pixels_per_unit_xy(label_w, label_h)
        return dx_px / pxu_x, dy_px / pxu_y


# ---------- Render size policy (target resolution + optional LOD) ----------
class RenderSizePolicy:
    def __init__(self, pitch_multiple: int = 32):
        self.pitch_multiple = max(1, int(pitch_multiple))
        self.target_quality = "1080p"  # default
        self.lod_steps: List[float] = [1.0]  # progressive disabled by default

    @staticmethod
    def _preset_to_height(label: str) -> int:
        mapping = {
            "2160p": 2160,
            "1440p": 1440,
            "1080p": 1080,
            "720p":   720,
            "480p":   480,
            "360p":   360,
        }
        return mapping.get(label, 1080)

    def compute_target_size(self, aspect_ratio: float) -> Tuple[int, int]:
        h = self._preset_to_height(self.target_quality)
        w = int(round(h * float(aspect_ratio)))
        snap = self.pitch_multiple

        def snap_down(v: int) -> int:
            return max(snap, (int(v) // snap) * snap)

        return snap_down(w), snap_down(h)

    def compute_lods(self, base_w: int, base_h: int) -> List[Tuple[int, int]]:
        snap = self.pitch_multiple

        def sd(v: int) -> int: return max(snap, (int(v) // snap) * snap)

        sizes = []
        for s in self.lod_steps:
            w = sd(int(round(base_w * s)))
            h = sd(int(round(base_h * s)))
            if not sizes or sizes[-1] != (w, h):
                sizes.append((w, h))
        return sizes


# ---------- Tile compositor (receives tiles, composites to label size) ----------
class TileCompositor(QWidget):
    """
    Owns the progressive tile queue and compositing.
    Scales each tile from its frame grid (frame_w x frame_h) to the current label size.
    """
    def __init__(self, display_label: QLabel, stats_label: QLabel, parent=None):
        super().__init__(parent)
        self.display = display_label
        self.stats_label = stats_label

        self.tile_queue: Deque[Tuple[int, int, int, QImage, int, int]] = deque()  # (gen,x,y,qimg,frame_w,frame_h)
        self.tile_index: Dict[Tuple[int, int, int, int, int, int, int], int] = {}
        self.tile_overlay_rects: Deque[Tuple[int, int, int, int, int, int]] = deque(maxlen=5000)
        self.show_overlay: bool = True

        self.tiles_start_time: Optional[float] = None
        self.tiles_processed: int = 0

        self.cached_image: Optional[QImage] = None   # display-sized working image
        self.view_image: Optional[QImage] = None     # last committed display image

        self.paused: bool = False

        self.flush_timer = QTimer(self)
        self.flush_timer.setInterval(16)  # ~60 FPS
        self.flush_timer.timeout.connect(self.flush)

    def set_paused(self, paused: bool) -> None:
        self.paused = bool(paused)
        if self.paused and self.flush_timer.isActive():
            self.flush_timer.stop()

    def clear(self):
        self.tile_queue.clear()
        self.tile_index.clear()
        if self.flush_timer.isActive():
            self.flush_timer.stop()
        self.cached_image = None
        self.tiles_start_time = None
        self.tiles_processed = 0
        self.tile_overlay_rects.clear()

    def set_overlay_enabled(self, enabled: bool):
        self.show_overlay = bool(enabled)

    # ---- Incoming tiles ----
    def on_tile_ready(self, gen: int, x: int, y: int, tile_qimg: QImage, frame_w: int, frame_h: int):
        key = (gen, x, y, tile_qimg.width(), tile_qimg.height(), frame_w, frame_h)
        entry = (gen, x, y, tile_qimg, frame_w, frame_h)
        if key in self.tile_index:
            idx = self.tile_index[key]
            self.tile_queue[idx] = entry
        else:
            self.tile_queue.append(entry)
            self.tile_index[key] = len(self.tile_queue) - 1

        if self.tiles_start_time is None:
            self.tiles_start_time = _t.time()
        self.tiles_processed += 1

        if self.show_overlay:
            self.tile_overlay_rects.append((x, y, tile_qimg.width(), tile_qimg.height(), frame_w, frame_h))

        if not self.paused and not self.flush_timer.isActive():
            self.flush_timer.start()

    # ---- Compositing pass ----
    def flush(self):
        if self.paused:
            return

        disp_w, disp_h = self.display.width(), self.display.height()
        if disp_w <= 0 or disp_h <= 0:
            return

        # Stats
        if self.tiles_start_time:
            dt = max(1e-3, _t.time() - self.tiles_start_time)
            tps = self.tiles_processed / dt
            if self.stats_label:
                self.stats_label.setText(f"Tiles: {self.tiles_processed}\nTiles/sec: {tps:.1f}")

        # Prepare backing image
        if self.cached_image is None:
            if self.view_image is not None:
                self.cached_image = self.view_image.copy()
            else:
                self.cached_image = QImage(disp_w, disp_h, QImage.Format_RGB32)
                self.cached_image.fill(Qt.GlobalColor.black)

        budget = min(16, len(self.tile_queue))
        painter = QPainter(self.cached_image)
        for _ in range(budget):
            gen, x, y, tile_qimg, frame_w, frame_h = self.tile_queue.popleft()
            key = (gen, x, y, tile_qimg.width(), tile_qimg.height(), frame_w, frame_h)
            self.tile_index.pop(key, None)

            sx = disp_w / max(1.0, float(frame_w))
            sy = disp_h / max(1.0, float(frame_h))

            dst_x = int(round(x * sx))
            dst_y = int(round(y * sy))
            dst_w = int(round(tile_qimg.width() * sx))
            dst_h = int(round(tile_qimg.height() * sy))

            if dst_w > 0 and dst_h > 0:
                scaled_tile = tile_qimg.scaled(dst_w, dst_h,
                                               Qt.AspectRatioMode.IgnoreAspectRatio,
                                               Qt.TransformationMode.SmoothTransformation)
                painter.drawImage(dst_x, dst_y, scaled_tile)
        painter.end()

        # Overlay (optional)
        if self.show_overlay and self.tile_overlay_rects:
            painter = QPainter(self.cached_image)
            painter.setPen(Qt.GlobalColor.red)
            overlay_budget = min(64, len(self.tile_overlay_rects))
            for _ in range(overlay_budget):
                x, y, w, h, fw, fh = self.tile_overlay_rects.popleft()
                sx = disp_w / max(1.0, float(fw))
                sy = disp_h / max(1.0, float(fh))
                painter.drawRect(int(round(x * sx)),
                                 int(round(y * sy)),
                                 int(round(w * sx)),
                                 int(round(h * sy)))
            painter.end()

        # Present
        self.display.setPixmap(QPixmap.fromImage(self.cached_image))

        if not self.tile_queue:
            self.flush_timer.stop()
            self.view_image = self.cached_image.copy()
