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

        params = fractal.get_kernel_source(settings, self.name)
        self.kernel_func = params["kernel_source"]
        self.precision = params["precision"]

    def render(self, fractal: Fractal, vp: Viewport, settings: RenderSettings,
               reference: Optional[Dict[str, Any]] = None) -> np.ndarray:

        args = fractal.get_kernel_args(vp, settings)

        real = np.linspace(args["min_x"], args["max_x"], args["width"],
                           dtype=self.precision)
        imag = np.linspace(args["min_y"], args["max_y"], args["height"],
                           dtype=self.precision)
        real_grid, imag_grid = np.meshgrid(real, imag)

        return self.kernel_func(real_grid, imag_grid,
                                args["width"], args["height"],
                                args["max_iter"], args["samples"])
