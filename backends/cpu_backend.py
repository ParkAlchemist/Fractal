
# --- Reference backends/cpu_backend.py v1.0 --------------------------------
import numpy as np
from typing import Dict, Any, Optional
from fractals.fractal_base import Fractal, Viewport, RenderSettings
from backends.backend_base import Backend
from kernel_sources.cpu.mandelbrot import (
    mandelbrot_kernel_cpu_f32, mandelbrot_kernel_cpu_f64,
    mandelbrot_kernel_cpu_perturb,
)

class CpuBackend(Backend):
    name = "CPU"

    def compile(self, fractal: Fractal, settings: RenderSettings) -> None:
        pass  # Numba JIT on first call

    def render(self, fractal: Fractal, vp: Viewport, settings: RenderSettings,
               reference: Optional[Dict[str, Any]]) -> np.ndarray:
        if not settings.use_perturb:
            real = np.linspace(vp.min_x, vp.max_x, vp.width, dtype=settings.precision)
            imag = np.linspace(vp.min_y, vp.max_y, vp.height, dtype=settings.precision)
            real_grid, imag_grid = np.meshgrid(real, imag)
            if settings.precision == np.float32:
                return mandelbrot_kernel_cpu_f32(real_grid, imag_grid, vp.width, vp.height,
                                                 settings.max_iter, settings.samples)
            else:
                return mandelbrot_kernel_cpu_f64(real_grid, imag_grid, vp.width, vp.height,
                                                 settings.max_iter, settings.samples)

        args = fractal.kernel_args(vp, settings, reference)
        total_samples = settings.samples * settings.samples
        accum = np.zeros((vp.height, vp.width), dtype=np.float64)
        step_x, step_y, c0 = args["step_x"], args["step_y"], args["c0"]

        for sx in range(settings.samples):
            for sy in range(settings.samples):
                off_x = (sx + 0.5) / settings.samples
                off_y = (sy + 0.5) / settings.samples
                c0_off = c0 + off_x*step_x + off_y*step_y
                counts = mandelbrot_kernel_cpu_perturb(
                    zref=args["zref"], c_ref=args["c_ref"],
                    width=vp.width, height=vp.height,
                    c0=c0_off, step_x=step_x, step_y=step_y,
                    max_iter=settings.max_iter,
                    order=settings.perturb_order, w_fallback_thresh=settings.perturb_thresh
                )
                accum += counts.astype(np.float64) / float(settings.max_iter)
        return (accum / float(total_samples)).astype(settings.precision)
