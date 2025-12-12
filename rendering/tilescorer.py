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
    Escape-time friendly, mode-aware scorer.
    Produces a composite score in [0, +inf) where higher means higher priority.
    """

    def __init__(self, weights_motion: Dict[str, float], weights_idle: Dict[str, float],
                 age_target_s: float = 2.0,
                 seam_cap: int = 4,
                 variance_gain: float = 1.0):
        self._wm = Weights(**weights_motion)
        self._wi = Weights(**weights_idle)
        self.age_target = age_target_s
        self.seam_cap = seam_cap
        self.variance_gain = variance_gain

    @staticmethod
    def _vis(x0: int, y0: int, w: int, h: int, W: int, H: int) -> float:
        """
        Checks wether given point is inside viewport i.e. is visible.
        return 1.0 if visible, 0.0 if not.
        """
        return 1.0 if (x0 < W and y0 < H and w > 0 and h > 0) else 0.0

    @staticmethod
    def _center_proximity(x0: int, y0: int, w: int, h: int, W: int,
                          H: int) -> float:
        """
        Calculates distance from center of viewport and normalizes value
        """
        cx, cy = W * 0.5, H * 0.5
        tx, ty = x0 + w * 0.5, y0 + h * 0.5
        dist = math.hypot(tx - cx, ty - cy)
        dmax = math.hypot(cx, cy)
        c_raw = 1.0 / (1.0 + dist)
        c_min = 1.0 / (1.0 + dmax)
        denom = (1.0 - c_min) if (1.0 - c_min)  > 1e-12 else 1.0
        return max(0.0, min(1.0, (c_raw - c_min) / denom))

    @staticmethod
    def _area_on_screen(w: int, h: int, W: int, H: int) -> float:
        """
        Calculates relative area of given tile compared to viewport.
        :return: tile_area / view_port_area
        """
        return (w * h) / max(1.0, (W * H))

    @staticmethod
    def _motion_ahead(x0: int, y0: int, w: int, h: int, W: int, H: int,
                      vx: float, vy: float) -> float:
        """
        Calculates motion of given tile compared to previous viewport.
        """
        cx, cy = W * 0.5, H * 0.5
        tx, ty = x0 + w * 0.5 - cx, y0 + h * 0.5 - cy
        tv = math.hypot(tx, ty)
        vv = math.hypot(vx, vy)
        if tv == 0.0 or vv == 0.0:
            return 0.0
        # radial dot product toward the motion vector; only forward boosts
        return max(0.0, (tx / tv) * (vx / vv) + (ty / tv) * (vy / vv))

    def _age(self, enqueue_time: float, now: float) -> float:
        """
        Calculates a normalized age for how long tile has been in queue
        :param enqueue_time: when tile was added to queue
        :param now: now
        """
        # simple aging: ~1.0 after ~2 seconds in the queue
        if now <= enqueue_time:
            return 0.0
        return max(0.0, min(1.0, (now - enqueue_time) / self.age_target))

    def _var_norm(self, iteration_variance: float) -> float:
        """
        Normalizes iteration variance
        :param iteration_variance:
        :return: normalized iteration variance
        """
        v = max(0.0, iteration_variance)
        return max(0.0, min(1.0, self.variance_gain * v))

    @staticmethod
    def _bnd_norm(boundary_likelihood: float) -> float:
        """
        Normalizes boundary likelihood
        :param boundary_likelihood:
        :return: normalized boundary likelihood
        """
        return max(0.0, min(1.0, boundary_likelihood))

    def _seam_norm(self, neighbors_rendered: int) -> float:
        """
        Normalizes neighbors rendered
        :param neighbors_rendered:
        :return: normalized seam val
        """
        cap = max(1, self.seam_cap)
        return max(0.0, min(1.0, neighbors_rendered / cap))

    def score(self,
              x0: int, y0: int, w: int, h: int, W: int, H: int,
              enqueue_time: float, now: float,
              iteration_variance: float, boundary_likelihood: float,
              neighbors_rendered: int,
              motion_vec: Tuple[float, float],
              moving: bool) -> float:
        """
        Calculates priority score for given tile based on multiple metrics
        :return: priority score
        """

        vx, vy = motion_vec
        Wt = self._wm if moving else self._wi

        return (
                Wt.vis * self._vis(x0, y0, w, h, W, H) +
                Wt.center * self._center_proximity(x0, y0, w, h, W, H) +
                Wt.area * self._area_on_screen(w, h, W, H) +
                Wt.motion * self._motion_ahead(x0, y0, w, h, W, H, vx, vy) +
                Wt.age * self._age(enqueue_time, now) +
                Wt.var * self._var_norm(iteration_variance) +
                Wt.bnd * self._bnd_norm(boundary_likelihood) +
                self._seam_norm(neighbors_rendered)
        )
