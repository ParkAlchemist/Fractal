import math
from dataclasses import dataclass
from typing import Dict, Tuple

@dataclass
class Weights:
    vis: float
    center: float
    area: float
    motion: float
    age: float
    var: float
    bnd: float

class TileScorer:
    """
    Escape-time-friendly, mode-aware scorer.
    Produces a composite score in [0, +inf) where higher means higher priority.
    """

    def __init__(self,
                 weights_motion: Dict[str, float],
                 weights_idle: Dict[str, float],
                 age_target_s: float = 2.0,
                 seam_cap: int = 4,
                 variance_gain: float = 1.0):
        self._wm = Weights(**weights_motion)
        self._wi = Weights(**weights_idle)
        self.age_target = age_target_s
        self.seam_cap = seam_cap
        self.variance_gain = variance_gain

    @staticmethod
    def _vis(x0: int, y0: int, w: int, h: int,
             frame_width: int, frame_height: int) -> float:
        return 1.0 if (x0 < frame_width and y0 < frame_height and w > 0 and h > 0) else 0.0

    @staticmethod
    def _center_proximity(x0: int, y0: int, w: int, h: int, frame_width: int,
                          frame_height: int) -> float:
        center_x, center_y = frame_width * 0.5, frame_height * 0.5
        tile_center_x, tile_center_y = x0 + w * 0.5, y0 + h * 0.5
        dist = math.hypot(tile_center_x - center_x, tile_center_y - center_y)
        dmax = math.hypot(center_x, center_y)
        c_raw = 1.0 / (1.0 + dist)
        c_min = 1.0 / (1.0 + dmax)
        denom = (1.0 - c_min) if (1.0 - c_min) > 1e-12 else 1.0
        return max(0.0, min(1.0, (c_raw - c_min) / denom))

    @staticmethod
    def _area_on_screen(w: int, h: int,
                        frame_width: int, frame_height: int) -> float:
        return (w * h) / max(1.0, (frame_width * frame_height))

    @staticmethod
    def _motion_ahead(x0: int, y0: int, w: int, h: int,
                      frame_width: int, frame_height: int,
                      motion_x_px: float, motion_y_px: float) -> float:
        center_x, center_y = frame_width * 0.5, frame_height * 0.5
        tile_center_x, tile_center_y = x0 + w * 0.5 - center_x, y0 + h * 0.5 - center_y
        tile_vec_len = math.hypot(tile_center_x, tile_center_y)
        motion_len = math.hypot(motion_x_px, motion_y_px)
        if tile_vec_len == 0.0 or motion_len == 0.0:
            return 0.0
        return max(
            0.0,
            (tile_center_x / tile_vec_len) * (motion_x_px / motion_len) +
            (tile_center_y / tile_vec_len) * (motion_y_px / motion_len)
        )

    def _age(self, enqueue_time: float, now: float) -> float:
        if now <= enqueue_time:
            return 0.0
        return max(0.0, min(1.0, (now - enqueue_time) / self.age_target))

    def _var_norm(self, iteration_variance: float) -> float:
        v = max(0.0, iteration_variance)
        return max(0.0, min(1.0, self.variance_gain * v))

    @staticmethod
    def _bnd_norm(boundary_likelihood: float) -> float:
        return max(0.0, min(1.0, boundary_likelihood))

    def _seam_norm(self, neighbors_rendered: int) -> float:
        cap = max(1, self.seam_cap)
        return max(0.0, min(1.0, neighbors_rendered / cap))

    def score(self,
              x0: int, y0: int, w: int, h: int,
              frame_width: int, frame_height: int,
              enqueue_time: float, now: float,
              iteration_variance: float, boundary_likelihood: float,
              neighbors_rendered: int,
              motion_vector_px: Tuple[float, float],
              is_moving: bool) -> float:

        motion_x_px, motion_y_px = motion_vector_px
        weights = self._wm if is_moving else self._wi

        return (
            weights.vis * self._vis(x0, y0, w, h, frame_width, frame_height) +
            weights.center * self._center_proximity(x0, y0, w, h,
                                                    frame_width, frame_height) +
            weights.area * self._area_on_screen(w, h, frame_width, frame_height) +
            weights.motion * self._motion_ahead(x0, y0, w, h,
                                                frame_width, frame_height,
                                                motion_x_px, motion_y_px) +
            weights.age * self._age(enqueue_time, now) +
            weights.var * self._var_norm(iteration_variance) +
            weights.bnd * self._bnd_norm(boundary_likelihood) +
            self._seam_norm(neighbors_rendered)
        )
