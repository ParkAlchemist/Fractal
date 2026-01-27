import numpy as np
from numba import njit, prange

from kernel_sources.registry import register_kernel

ARG_SCALARS = ["max_iter"]
ARG_BUFFERS_IN = ["iter_raw", "z_mag"]
ARG_BUFFERS_OUT = ["iter_smooth"]

ARG_ORDER = ARG_SCALARS + ARG_BUFFERS_IN + ARG_BUFFERS_OUT

@njit(cache=True, fastmath=True, parallel=True)
def _mandelbrot_smooth(max_iter, iter_raw, z_mag, iter_smooth):
    H, W = iter_raw.shape

    log2 = iter_raw.dtype.type(np.log(2.0))
    eps = iter_raw.dtype.type(1e-20)

    for y in prange(H):
        for x in range(W):
            n = iter_raw[y, x]
            if n >= max_iter:
                iter_smooth[y, x] = n
                continue
            r = z_mag[y, x]
            iter_smooth[y, x] = n + 1.0 - np.log(np.log(r + eps)) / log2

register_kernel(
    fractal="mandelbrot",
    op_name="smooth",
    backend="CPU",
    precision="f32",
    func=_mandelbrot_smooth,
    arg_order=ARG_ORDER,
    scalars=ARG_SCALARS,
    produces=ARG_BUFFERS_OUT,
    consumes=ARG_BUFFERS_IN,
    buffers=ARG_BUFFERS_IN + ARG_BUFFERS_OUT,
    block=None
)

register_kernel(
    fractal="mandelbrot",
    op_name="smooth",
    backend="CPU",
    precision="f64",
    func=_mandelbrot_smooth,
    arg_order=ARG_ORDER,
    scalars=ARG_SCALARS,
    produces=ARG_BUFFERS_OUT,
    consumes=ARG_BUFFERS_IN,
    buffers=ARG_BUFFERS_IN + ARG_BUFFERS_OUT,
    block=None
)
