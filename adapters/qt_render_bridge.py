from PySide6.QtCore import QObject, Signal
from PySide6.QtGui import QImage

from api.render_api import RenderAPI
from utils.image_helpers import ndarray_to_qimage
from rendering.events import FrameEvent, TileEvent, LogEvent


class QtRenderBridge(QObject):
    """
    Thin adapter that converts service events to Qt signals for the UI.
    """
    image_updated = Signal(QImage, int, int)
    tile_ready = Signal(int, int, int, QImage, int, int)
    log_text = Signal(str)

    def __init__(self, api: RenderAPI, parent=None):
        super().__init__(parent)
        self.api = api

        # Subscribe to API events with conversions
        self.api.on_frame(self._on_frame)
        self.api.on_tile(self._on_tile)
        self.api.on_log(self._on_log)

    # --------- Conversions ---------------------
    def _on_frame(self, evt: FrameEvent) -> None:
        qimg = ndarray_to_qimage(evt.data).copy()
        self.image_updated.emit(qimg, evt.width, evt.height)

    def _on_tile(self, evt: TileEvent) -> None:
        qimg = ndarray_to_qimage(evt.data).copy()
        self.tile_ready.emit(int(evt.seq), int(evt.x), int(evt.y), qimg, int(evt.frame_w), int(evt.frame_h))

    def _on_log(self, evt: LogEvent) -> None:
        self.log_text.emit(evt.message)
