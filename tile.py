import heapq
import itertools
import math
import typing
from dataclasses import dataclass, field

from PyQt5.QtCore import pyqtSignal, QObject, QThread, QMutexLocker, QMutex
from PyQt5.QtGui import QPainter, QImage

from fractal import render_mandelbrot


@dataclass(order=True)
class Tile:
    priority: float
    tile_id: int = field(compare=False)
    x: int = field(compare=False)
    y: int = field(compare=False)
    width: int = field(compare=False)
    height: int = field(compare=False)
    min_x: float = field(compare=False)
    max_x: float = field(compare=False)
    min_y: float = field(compare=False)
    max_y: float = field(compare=False)
    resolution: float = field(compare=False)
    session_id: int = field(compare=False)

class TileScheduler:
    def __init__(self):
        self._queue = []
        self._tile_counter = itertools.count()
        self._lock = QMutex()
        self._active_session = 0

    def new_session(self) -> int:
        with QMutexLocker(self._lock):
            self._queue.clear()
            self._active_session += 1
            return self._active_session

    def add_tile(self, tile: Tile):
        with QMutexLocker(self._lock):
            heapq.heappush(self._queue, tile)

    def get_next_tile(self) -> typing.Optional[Tile]:
        with QMutexLocker(self._lock):
            if self._queue:
                return heapq.heappop(self._queue)
            return None

    def has_tiles(self) -> bool:
        with QMutexLocker(self._lock):
            return bool(self._queue)

    def current_session(self) -> int:
        with QMutexLocker(self._lock):
            return self._active_session

class TileRenderWorker(QThread):
    tile_rendered = pyqtSignal(int, int, QImage, int)   # x, y, image, session_id

    def __init__(self, scheduler:  TileScheduler, palette):
        super().__init__()
        self.scheduler = scheduler
        self.palette = palette
        self._running = True

    def stop(self):
        self._running = False

    def run(self):
        while self._running:
            tile = self.scheduler.get_next_tile()
            if tile is None:
                self.msleep(10)
                continue

            # Skip outdated tiles
            if tile.session_id != self.scheduler.current_session():
                continue

            # Render with CUDA
            data = render_mandelbrot(width=tile.width,
                                     height=tile.height,
                                     max_iter=1000,
                                     min_x=tile.min_x,
                                     max_x=tile.max_x,
                                     min_y=tile.min_y,
                                     max_y=tile.max_y,
                                     palette=self.palette
                                     )

            # Convert to QImage
            h, w, c, = data.shape
            bytes_per_line = c * w
            image = QImage(data.data, w, h, bytes_per_line, QImage.Format_RGB888).copy()

            # Emit signal
            self.tile_rendered.emit(tile.x, tile.y, image, tile.session_id)

class TileRenderer(QObject):
    image_updated = pyqtSignal(QImage)

    def __init__(self, palette, tile_size=2048):
        super().__init__()
        self.scheduler = TileScheduler()
        self.worker = TileRenderWorker(self.scheduler, palette)
        self.worker.tile_rendered.connect(self.handle_tile_rendered)
        self.worker.start()

        self.tile_size = tile_size
        self.full_image = None
        self.session_id = 0

    def start_render(self, width, height, center_x, center_y, zoom, view_width, view_height):
        self.session_id = self.scheduler.new_session()

        if self.full_image is None:
            self.full_image = QImage(width, height, QImage.Format_RGB32)
            self.full_image.fill(0)

        res = 1.0

        tile_w = int(self.tile_size * res)
        tile_h = int(self.tile_size * res)
        cols = math.ceil(width / tile_w)
        rows = math.ceil(height / tile_h)

        for row in range(rows):
            for col in range(cols):
                x = col * tile_w
                y = row * tile_h
                w = min(tile_w, width - x)
                h = min(tile_h, height - y)

                # Convert view coordinates to fractal coordinates
                scale = 1.0 / zoom
                fx0 = center_x + (x - width / 2) * scale
                fx1 = center_x + (x + w - width / 2) * scale
                fy0 = center_y + (y - height / 2) * scale
                fy1 = center_y + (y + h - height / 2) * scale

                # Priority: higher for tiles inside view, close to center
                cx = (x - tile_w / 2) + w / 2
                cy = (y - tile_h / 2) + h / 2
                dist = math.hypot(cx - width / 2, cy - height / 2)
                if (width + view_width) / 2 < cx < (width - view_width) / 2 and (
                        height + view_height) / 2 < cy < (
                        height - view_height) / 2:
                    dist += 10**4

                tile = Tile(priority=dist + (1.0 - res),
                            tile_id=next(self.scheduler._tile_counter),
                            x=x, y=y, width=w, height=h,
                            min_x=fx0, max_x=fx1, min_y=fy0, max_y=fy1,
                            resolution=res,
                            session_id=self.session_id
                            )
                self.scheduler.add_tile(tile)

    def handle_tile_rendered(self, x, y, image, session_id):
        if session_id != self.session_id:
            return  # Outdated tile

        painter = QPainter(self.full_image)
        painter.drawImage(x, y, image)
        painter.end()

        self.image_updated.emit(self.full_image)

    def stop(self):
        self.worker.stop()
        self.worker.wait()

    def update_palette(self, palette):
        self.worker.palette = palette
