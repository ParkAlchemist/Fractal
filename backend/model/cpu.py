import numpy as np
from typing import Dict, Any, Optional, List

from fractals.base import Fractal, Viewport, RenderSettings
from backend.model.base import Backend


class CpuBackend(Backend):
    """
    Backend for CPU-based fractal rendering.
    """
    name = "CPU"

    def __init__(self):
        self.kernel_func = None
        self.precision = None

        # Warm up params
        self._wu_min_x = -2.0
        self._wu_max_x = 1.0
        self._wu_min_y = -1.5
        self._wu_max_y = 1.5
        self._wu_width = 64
        self._wu_height = 64
        self._wu_max_iter = 64
        self._wu_samples = 1
        self._warmed_up = False

    @staticmethod
    def enumerate_devices() -> List[dict]:
        return [{
            "device_id": None,
            "name": "CPU",
            "vendor": None,
            "driver": None,
            "compute_capability": None,
            "memory_total_mb": None,
            "memory_free_mb": None,
            "is_available": True
        }]

    def compile(self, fractal: Fractal, settings: RenderSettings) -> None:

        spec = fractal.get_backend_spec(settings, self.name)
        self.kernel_func = spec["kernel_source"]
        self.precision = spec["precision"]
        self._warmup()

    def _warmup(self) -> None:
        """
        Used to warm up the backend with a small sample of the fractal to ensure the kernel is compiled by numba.
        """
        if self._warmed_up: return
        real = np.linspace(self._wu_min_x, self._wu_max_x, self._wu_width, dtype=self.precision)
        imag = np.linspace(self._wu_min_y, self._wu_max_y, self._wu_height, dtype=self.precision)
        real_grid, imag_grid = np.meshgrid(real, imag)
        self.kernel_func(real_grid, imag_grid, self._wu_width, self._wu_height, self._wu_max_iter, self._wu_samples)
        self._warmed_up = True

    def render(self, fractal: Fractal, vp: Viewport, settings: RenderSettings,
               reference: Optional[Dict[str, Any]] = None) -> np.ndarray:

        if self.kernel_func is None:
            raise RuntimeError("Backend has not been compiled yet")

        params = fractal.get_backend_params(vp, settings)

        real = np.linspace(params["min_x"], params["max_x"], params["width"],
                           dtype=self.precision)
        imag = np.linspace(params["min_y"], params["max_y"], params["height"],
                           dtype=self.precision)
        real_grid, imag_grid = np.meshgrid(real, imag)

        return self.kernel_func(real_grid, imag_grid,
                                params["width"], params["height"],
                                params["max_iter"], params["samples"])

    def close(self) -> None:
        self.kernel_func = None
        self.precision = None
