from dataclasses import dataclass
from typing import Dict, Any
import numpy as np

from fractals.base import Fractal, Viewport, RenderSettings, ProgramSpec, \
    ArgSpec, KernelStep
from kernel_sources.cpu.mandelbrot import (mandelbrot_kernel_cpu_f32,
                                           mandelbrot_kernel_cpu_f64)
from kernel_sources.cuda.mandelbrot import (mandelbrot_kernel_cuda_f32,
                                            mandelbrot_kernel_cuda_f64)
from kernel_sources.opencl.mandelbrot import (mandelbrot_kernel_cl_f32,
                                              mandelbrot_kernel_cl_f64)


@dataclass
class MandelbrotFractal(Fractal):
    """
    Mandelbrot fractal type.
    Provides a program spec per backend and the scalar argument values
    computed from a given viewport and settings.
    """
    name: str = "mandelbrot"

    # ------------ Scalar values computed from viewport and settings ------------
    def build_arg_values(self, vp: Viewport, st: RenderSettings) -> Dict[str, Any]:
        """Compute scalar argument values"""
        return {
            "min_x": vp.min_x,
            "max_x": vp.max_x,
            "min_y": vp.min_y,
            "max_y": vp.max_y,
            "width": vp.width,
            "height": vp.height,
            "max_iter": st.max_iter,
            "samples": st.samples
        }

    # ---------- Per-backend program description ------------------
    def get_program_spec(self, st: RenderSettings, backend_name: str) -> ProgramSpec:
        """Return a backend agnostic program description"""

        cast = st.precision
        backend = backend_name.upper()

        # Choose proper backend kernel function
        if backend == "OPENCL":
            func = mandelbrot_kernel_cl_f32 if cast == np.float32 else mandelbrot_kernel_cl_f64
        elif backend == "CUDA":
            func = mandelbrot_kernel_cuda_f32 if cast == np.float32 else mandelbrot_kernel_cuda_f64
        elif backend == "CPU":
            func = mandelbrot_kernel_cpu_f32 if cast == np.float32 else mandelbrot_kernel_cpu_f64
        else:
            raise NotImplementedError(f"Unsupported backend: {backend}")

        # Declare all arguments and their semantics
        # Buffer shapes reference "H, W" (height, widht) computed from the viewport
        args: Dict[str, ArgSpec] = {
            "min_x": ArgSpec("min_x", role="scalar",
                             dtype=cast, source="viewport"),
            "max_x": ArgSpec("max_x", role="scalar",
                             dtype=cast, source="viewport"),
            "min_y": ArgSpec("min_y", role="scalar",
                             dtype=cast, source="viewport"),
            "max_y": ArgSpec("max_y", role="scalar",
                             dtype=cast, source="viewport"),
            "out": ArgSpec("out", role="buffer_out", dtype=cast,
                           shape_expr="H,W", source="runtime"),
            "max_iter": ArgSpec("max_iter", role="scalar",
                                dtype=np.int32, source="settings"),
            "samples": ArgSpec("samples", role="scalar",
                               dtype=np.int32, source="settings"),
            "width": ArgSpec("width", role="scalar",
                             dtype=np.int32, source="viewport"),
            "height": ArgSpec("height", role="scalar",
                              dtype=np.int32, source="viewport")

        }

        # The ordered argument list for the kernel function
        step = KernelStep(
            name="mandelbrot_main",
            func=func,
            args=["min_x", "max_x", "min_y", "max_y", "out", "max_iter", "samples"],
            meta=None
        )

        return ProgramSpec(
            backend=backend,
            precision=cast,
            args=args,
            steps=[step],
            output_arg="out"
        )

    def output_semantics(self) -> str:
        """Describe output semantics for the coloring pipeline"""
        return "iterations"
