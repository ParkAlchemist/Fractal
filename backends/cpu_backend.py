import numpy as np
from typing import Dict, Any, Optional

from fractals.fractal_base import Fractal, Viewport, RenderSettings
from backends.backend_base import Backend


class CpuBackend(Backend):
    name = "CPU"

    def __init__(self):
        self.kernel_func = None
        self.precision = None

    def compile(self, fractal: Fractal, settings: RenderSettings) -> None:

        spec = fractal.get_backend_spec(settings, self.name)
        self.kernel_func = spec["kernel_source"]
        self.precision = spec["precision"]

    def render(self, fractal: Fractal, vp: Viewport, settings: RenderSettings,
               reference: Optional[Dict[str, Any]] = None) -> np.ndarray:

        params = fractal.get_backend_params(vp, settings)

        real = np.linspace(params["min_x"], params["max_x"], params["width"],
                           dtype=self.precision)
        imag = np.linspace(params["min_y"], params["max_y"], params["height"],
                           dtype=self.precision)
        real_grid, imag_grid = np.meshgrid(real, imag)

        return self.kernel_func(real_grid, imag_grid,
                                params["width"], params["height"],
                                params["max_iter"], params["samples"])
