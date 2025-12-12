import numpy as np
from typing import Dict, Any, Optional
from numba import cuda

from fractals.fractal_base import Fractal, Viewport, RenderSettings
from backends.backend_base import Backend
from kernel_sources.cuda.mandelbrot import (
    mandelbrot_kernel_cuda_f32, mandelbrot_kernel_cuda_f64
)

class CudaBackend(Backend):
    name = "CUDA"

    def __init__(self):
        if not cuda.is_available():
            raise RuntimeError("CUDA not available")
        self.threads_per_block = (16, 16)
        self.blocks_per_grid = None

    def compile(self, fractal: Fractal, settings: RenderSettings) -> None:
        pass

    def render(self, fractal: Fractal, vp: Viewport, settings: RenderSettings,
               reference: Optional[Dict[str, Any]] = None) -> np.ndarray:
        H, W = vp.height, vp.width
        if self.blocks_per_grid is None:
            self.blocks_per_grid = (
                (W + self.threads_per_block[0] - 1) // self.threads_per_block[0],
                (H + self.threads_per_block[1] - 1) // self.threads_per_block[1],
            )

        d_out = cuda.device_array((H, W), dtype=settings.precision)
        if settings.precision == np.float32:
            mandelbrot_kernel_cuda_f32[self.blocks_per_grid, self.threads_per_block](
                np.float32(vp.min_x), np.float32(vp.max_x),
                np.float32(vp.min_y), np.float32(vp.max_y),
                d_out, np.int32(settings.max_iter), np.int32(settings.samples)
            )
        else:
            mandelbrot_kernel_cuda_f64[self.blocks_per_grid, self.threads_per_block](
                np.float64(vp.min_x), np.float64(vp.max_x),
                np.float64(vp.min_y), np.float64(vp.max_y),
                d_out, np.int32(settings.max_iter), np.int32(settings.samples)
            )
        cuda.synchronize()
        return d_out.copy_to_host()
