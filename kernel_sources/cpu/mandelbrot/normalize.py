from numba import njit, prange

from kernel_sources.registry import register_kernel

ARG_SCALARS = ["min_val", "max_val"]
ARG_BUFFERS_IN = ["iter_smooth"]
ARG_BUFFERS_OUT = ["iter_norm"]

ARG_ORDER = ARG_SCALARS + ARG_BUFFERS_IN + ARG_BUFFERS_OUT

@njit(cache=True, fastmath=True, parallel=True)
def _minmax_normalize(min_val, max_val, iter_smooth, iter_norm):
    H, W = iter_smooth.shape
    if max_val <= min_val:
        for y in prange(H):
            for x in range(W):
                iter_norm[y, x] = iter_smooth.dtype.type(0.0)
        return
    denom = (max_val - min_val)
    for y in prange(H):
        for x in range(W):
            t = (iter_smooth[y, x] - min_val) / denom
            if t < 0: t = 0
            if t > 1: t = 1
            iter_norm[y, x] = iter_smooth.dtype.type(t)

register_kernel(
    fractal="mandelbrot",
    op_name="normalize",
    backend="CPU",
    precision="f32",
    func=_minmax_normalize,
    arg_order=ARG_ORDER,
    scalars=ARG_SCALARS,
    produces=ARG_BUFFERS_OUT,
    consumes=ARG_BUFFERS_IN,
    buffers=ARG_BUFFERS_IN + ARG_BUFFERS_OUT,
    block=None
)

register_kernel(
    fractal="mandelbrot",
    op_name="normalize",
    backend="CPU",
    precision="f64",
    func=_minmax_normalize,
    arg_order=ARG_ORDER,
    scalars=ARG_SCALARS,
    produces=ARG_BUFFERS_OUT,
    consumes=ARG_BUFFERS_IN,
    buffers=ARG_BUFFERS_IN + ARG_BUFFERS_OUT,
    block=None
)
