import numpy as np
from typing import Dict, Any, Optional

from fractals.fractal_base import Fractal, Viewport, RenderSettings
from backends.backend_base import Backend
from kernel_sources.cpu.mandelbrot import (
    mandelbrot_kernel_cpu_f32, mandelbrot_kernel_cpu_f64
)

class CpuBackend(Backend):
    name = "CPU"

    def compile(self, fractal: Fractal, settings: RenderSettings) -> None:
        pass  # Numba JIT on first call

    def render(self, fractal: Fractal, vp: Viewport, settings: RenderSettings,
               reference: Optional[Dict[str, Any]] = None) -> np.ndarray:
        real = np.linspace(vp.min_x, vp.max_x, vp.width,
                           dtype=settings.precision)
        imag = np.linspace(vp.min_y, vp.max_y, vp.height,
                           dtype=settings.precision)
        real_grid, imag_grid = np.meshgrid(real, imag)

        if settings.precision == np.float32:
            return mandelbrot_kernel_cpu_f32(real_grid, imag_grid, vp.width,
                                             vp.height,
                                             settings.max_iter,
                                             settings.samples)
        else:
            return mandelbrot_kernel_cpu_f64(real_grid, imag_grid, vp.width,
                                             vp.height,
                                             settings.max_iter,
                                             settings.samples)
