import heapq
import time
from dataclasses import dataclass
from typing import Dict, Optional, Tuple, List

from rendering.tilescorer import TileScorer
from fractals.fractal_base import Viewport


@dataclass
class TileInfo:
    x0: int
    y0: int
    w: int
    h: int
    depth: int
    enqueue_time: float
    resolution: str = None
    # dynamic
    stale: bool = False
    iteration_variance: float = 0.0
    boundary_likelihood: float = 0.0
    neighbors_rendered: int = 0


class TileScheduler:
    """
    Maintains phase-specific priority queues and provides pop/enqueue operations.
    Phases: seam -> refine -> prefetch -> bg (strict order).
    """

    def __init__(self, scorer: TileScorer):
        self.scorer = scorer
        self.queues: Dict[str, List[Tuple[float, int, TileInfo]]] = {
            'seam': [], 'refine': [], 'prefetch': [], 'bg': []
        }
        self.motion_vec: Tuple[float, float] = (0.0, 0.0)
        self._moving: bool = False
        self._W = 0
        self._H = 0
        self._now = time.perf_counter()

    def clear(self):
        """
        Clears all priority queues.
        :return: None
        """
        for q in self.queues.values():
            q.clear()

    def update_view(self, W: int, H: int, current_viewport: Viewport,
                    last_viewport: Viewport) -> None:
        """
        Updates parameters based on new viewport.
        :return: None
        """
        self._W = W
        self._H = H
        if last_viewport is not None:
            vx = (current_viewport.min_x - last_viewport.min_x) * W
            vy = (current_viewport.min_y - last_viewport.min_y) * H
        else:
            vx = 0.0
            vy = 0.0

        self.motion_vec = (vx, vy)
        self._moving = (abs(vx) + abs(vy)) > 1e-6
        self._now = time.perf_counter()

    @staticmethod
    def _select_phase(t: TileInfo, visible: bool) -> str:
        """
        Selects correct priority queue for given tile
        :return: name of priority queue
        """
        if t.stale:
            return 'bg'
        if visible and (
                t.iteration_variance > 0.5 or t.boundary_likelihood > 0.5):
            return 'refine'
        if visible:
            return 'seam'
        return 'bg'

    def enqueue(self, t: TileInfo, visible: bool) -> None:
        """
        Adds given tile into priority queue.
        :return:
        """
        phase = self._select_phase(t, visible)
        s = self.scorer.score(
            t.x0, t.y0, t.w, t.h, self._W, self._H,
            t.enqueue_time, self._now, t.iteration_variance,
            t.boundary_likelihood, t.neighbors_rendered,
            self.motion_vec, self._moving
        )
        heapq.heappush(self.queues[phase], (-s, id(t), t))

    def pop_next(self) -> Optional[Tuple[str, TileInfo]]:
        """
        Yields next tile from priority queue.
        :return: Next tile or None if queue is empty
        """
        for phase in ('seam', 'refine', 'prefetch', 'bg'):
            if self.queues[phase]:
                _, _, t = heapq.heappop(self.queues[phase])
                if t.stale:
                    continue
                return phase, t
        return None
