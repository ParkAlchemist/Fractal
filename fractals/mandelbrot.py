from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

from fractals.fractal_base import Fractal, Viewport, RenderSettings
from kernel_sources.cpu.mandelbrot import (mandelbrot_kernel_cpu_f32,
                                           mandelbrot_kernel_cpu_f64)
from kernel_sources.cuda.mandelbrot import (mandelbrot_kernel_cuda_f32,
                                            mandelbrot_kernel_cuda_f64)
from kernel_sources.opencl.mandelbrot import (mandelbrot_kernel_cl_f32,
                                              mandelbrot_kernel_cl_f64)


@dataclass
class MandelbrotFractal(Fractal):
    name: str = "mandelbrot"

    def get_kernel_args(self, vp: Viewport, st: RenderSettings) -> Dict[str, Any]:
        min_x = vp.min_x
        max_x = vp.max_x
        min_y = vp.min_y
        max_y = vp.max_y
        width = vp.width
        height = vp.height
        max_iter = st.max_iter
        samples = st.samples
        return {
            "min_x": min_x,
            "max_x": max_x,
            "min_y": min_y,
            "max_y": max_y,
            "width": width,
            "height": height,
            "max_iter": max_iter,
            "samples": samples,
        }

    def get_kernel_source(self, settings: RenderSettings,
                          backend_name: str) -> Optional[Dict[str, Any]]:
        src = None
        name = "mandelbrot_kernel"
        cast = settings.precision
        if backend_name.lower() == "opencl":
            if cast == np.float32:
                src = mandelbrot_kernel_cl_f32
            elif cast == np.float64:
                src = mandelbrot_kernel_cl_f64
        elif backend_name.lower() == "cuda":
            if cast == np.float32:
                src = mandelbrot_kernel_cuda_f32
            elif cast == np.float64:
                src = mandelbrot_kernel_cuda_f64
        elif backend_name.lower() == "cpu":
            if cast == np.float32:
                src = mandelbrot_kernel_cpu_f32
            if cast == np.float64:
                src = mandelbrot_kernel_cpu_f64
        else:
            raise NotImplementedError
        return {
            "kernel_source": src,
            "kernel_name": name,
            "precision": cast
        }

    def output_semantics(self) -> str:
        pass
