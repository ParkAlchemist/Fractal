from typing import Optional, Callable, Generator, Tuple

import numpy as np

from fractals.base import Viewport, Fractal, RenderSettings
from rendering.engines.base import BaseRenderEngine
from backend.manager import BackendManager


class TileEngine(BaseRenderEngine):
    """
    Simple tile-based rendering engine. Divides viewport into given sized tiles
    and renders them sequentially.
    """
    def __init__(self, tile_w: int = 256, tile_h: int = 256,
                 on_tile: Optional[Callable[[int, int, np.ndarray], None]] = None):
        super().__init__(on_tile)
        self.tile_w = int(tile_w)
        self.tile_h = int(tile_h)
        self.center_out = True

    def _tile_viewports(self, vp: Viewport) -> Generator[Tuple[int, int, Viewport]]:
        W, H = vp.width, vp.height
        cx, cy = W / 2.0, H / 2.0
        tiles = []
        for y0 in range(0, H, self.tile_h):
            for x0 in range(0, W, self.tile_w):
                tw = min(self.tile_w, W - x0)
                th = min(self.tile_h, H - y0)
                txc, tyc = x0 + tw/2.0, y0 + th/2.0
                dist2 = (txc - cx)**2 + (tyc - cy)**2
                sx, ex = x0 / W, (x0 + tw) / W
                sy, ey = y0 / H, (y0 + th) / H
                sub_min_x = vp.min_x + (vp.max_x - vp.min_x) * sx
                sub_max_x = vp.min_x + (vp.max_x - vp.min_x) * ex
                sub_min_y = vp.min_y + (vp.max_y - vp.min_y) * sy
                sub_max_y = vp.min_y + (vp.max_y - vp.min_y) * ey
                sub_vp = Viewport(sub_min_x, sub_max_x, sub_min_y, sub_max_y, tw, th)
                tiles.append((x0, y0, sub_vp, dist2))
        if self.center_out:
            tiles.sort(key=lambda t: t[3])
        for x0, y0, sub_vp, _ in tiles:
            yield x0, y0, sub_vp

    def render(self, fractal: Fractal, manager: BackendManager, settings: RenderSettings,
               vp: Viewport,
               cancel_cb: Optional[Callable[[], bool]] = None,
               backend: Optional[str] = None,
               device: Optional[int] = None) -> np.ndarray:
        canvas = np.zeros((vp.height, vp.width), dtype=settings.precision)
        for x0, y0, sub_vp in self._tile_viewports(vp):
            if cancel_cb is not None and cancel_cb(): break
            tile = manager.render_async(fractal, sub_vp, settings, backend=backend, device=device).result
            canvas[y0:y0+sub_vp.height, x0:x0+sub_vp.width] = tile
            if callable(self.on_tile):
                self.on_tile(x0, y0, tile)
        return canvas
