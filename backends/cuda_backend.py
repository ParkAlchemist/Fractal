
# --- Reference backends/cuda_backend.py v1.0 --------------------------------
import numpy as np
from typing import Dict, Any, Optional
from numba import cuda
from fractals.fractal_base import Fractal, Viewport, RenderSettings
from backends.backend_base import Backend
from kernel_sources.cuda.mandelbrot import (
    mandelbrot_kernel_cuda_f32, mandelbrot_kernel_cuda_f64,
    mandelbrot_kernel_cuda_perturb_f64,
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
               reference: Optional[Dict[str, Any]]) -> np.ndarray:
        H, W = vp.height, vp.width
        if self.blocks_per_grid is None:
            self.blocks_per_grid = (
                (W + self.threads_per_block[0] - 1) // self.threads_per_block[0],
                (H + self.threads_per_block[1] - 1) // self.threads_per_block[1],
            )

        if settings.use_perturb:
            args = fractal.kernel_args(vp, settings, reference)
            zref = args["zref"]
            d_zref = cuda.to_device(zref)
            d_out = cuda.device_array((H, W), dtype=np.int32)

            mandelbrot_kernel_cuda_perturb_f64[self.blocks_per_grid, self.threads_per_block](
                np.float64(args["c_ref"].real), np.float64(args["c_ref"].imag),
                np.float64(args["c0"].real), np.float64(args["c0"].imag),
                np.float64(args["step_x"].real), np.float64(args["step_x"].imag),
                np.float64(args["step_y"].real), np.float64(args["step_y"].imag),
                np.int32(W), np.int32(H), np.int32(settings.max_iter),
                np.int32(settings.perturb_order), np.float64(settings.perturb_thresh),
                d_out
            )
            cuda.synchronize()
            counts = d_out.copy_to_host()

            if settings.samples > 1:
                total = settings.samples * settings.samples
                accum = counts.astype(np.float64) / float(settings.max_iter)
                for sx in range(settings.samples):
                    for sy in range(settings.samples):
                        if sx == 0 and sy == 0: continue
                        off_x = (sx + 0.5) / settings.samples
                        off_y = (sy + 0.5) / settings.samples
                        c0_off = args["c0"] + off_x*args["step_x"] + off_y*args["step_y"]
                        mandelbrot_kernel_cuda_perturb_f64[self.blocks_per_grid, self.threads_per_block](
                            np.float64(args["c_ref"].real), np.float64(args["c_ref"].imag),
                            np.float64(c0_off.real), np.float64(c0_off.imag),
                            np.float64(args["step_x"].real), np.float64(args["step_x"].imag),
                            np.float64(args["step_y"].real), np.float64(args["step_y"].imag),
                            np.int32(W), np.int32(H), np.int32(settings.max_iter),
                            np.int32(settings.perturb_order), np.float64(settings.perturb_thresh),
                            d_out
                        )
                        cuda.synchronize()
                        counts = d_out.copy_to_host()
                        accum += counts.astype(np.float64) / float(settings.max_iter)
                return (accum / float(total)).astype(settings.precision)
            else:
                return (counts.astype(np.float64) / float(settings.max_iter)).astype(settings.precision)

        # Non-perturb
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
