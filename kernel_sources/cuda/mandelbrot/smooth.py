import numpy as np
from numba import cuda

from kernel_sources.registry import register_kernel

ARG_SCALARS = ["max_iter"]
ARG_BUFFERS_IN = ["iter_raw", "z_mag"]
ARG_BUFFERS_OUT = ["iter_smooth"]

ARG_ORDER = ARG_SCALARS + ARG_BUFFERS_IN + ARG_BUFFERS_OUT

@cuda.jit
def _mandelbrot_smooth(max_iter, iter_raw, z_mag, iter_smooth):
    x, y = cuda.grid(2)
    H, W = iter_raw.shape
    if x >= W or y >= H:
        return
    n = iter_raw[y, x]
    if n >= max_iter:
        iter_smooth[y, x] = n  # interior
        return
    r = z_mag[y, x]
    eps = iter_raw.dtype.type(1e-20)
    iter_smooth[y, x] = n + 1.0 - np.log(np.log(r + eps)) / np.log(2.0)

register_kernel(
    fractal="mandelbrot",
    op_name="smooth",
    backend="CUDA",
    precision="f32",
    func=_mandelbrot_smooth,
    arg_order=ARG_ORDER,
    scalars=ARG_SCALARS,
    produces=ARG_BUFFERS_OUT,
    consumes=ARG_BUFFERS_IN,
    buffers=ARG_BUFFERS_IN + ARG_BUFFERS_OUT,
    block=(16, 16)
)

register_kernel(
    fractal="mandelbrot",
    op_name="smooth",
    backend="CUDA",
    precision="f64",
    func=_mandelbrot_smooth,
    arg_order=ARG_ORDER,
    scalars=ARG_SCALARS,
    produces=ARG_BUFFERS_OUT,
    consumes=ARG_BUFFERS_IN,
    buffers=ARG_BUFFERS_IN + ARG_BUFFERS_OUT,
    block=(16, 16)
)
