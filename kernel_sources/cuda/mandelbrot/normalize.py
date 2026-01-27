from numba import cuda

from kernel_sources.registry import register_kernel

ARG_SCALARS = ["min_val", "max_val"]
ARG_BUFFERS_IN = ["iter_smooth"]
ARG_BUFFERS_OUT = ["iter_norm"]

ARG_ORDER = ARG_SCALARS + ARG_BUFFERS_IN + ARG_BUFFERS_OUT

@cuda.jit
def _minmax_normalize(min_val, max_val, iter_smooth, iter_norm):
    x, y = cuda.grid(2)
    H, W = iter_smooth.shape
    if x >= W or y >= H:
        return

    mn = min_val
    mx = max_val
    val = iter_smooth[y, x]
    if mx > mn:
        t = (val - mn) / (mx - mn)
        # Clamp to [0,1]
        if t < 0: t = 0
        if t > 1: t = 1
        iter_norm[y, x] = iter_smooth.dtype.type(t)
    else:
        iter_norm[y, x] = iter_smooth.dtype.type(0.0)

register_kernel(
    fractal="mandelbrot",
    op_name="normalize",
    backend="CUDA",
    precision="f32",
    func=_minmax_normalize,
    arg_order=ARG_ORDER,
    scalars=ARG_SCALARS,
    produces=ARG_BUFFERS_OUT,
    consumes=ARG_BUFFERS_IN,
    buffers=ARG_BUFFERS_IN + ARG_BUFFERS_OUT,
    block=(16, 16)
)

register_kernel(
    fractal="mandelbrot",
    op_name="normalize",
    backend="CUDA",
    precision="f64",
    func=_minmax_normalize,
    arg_order=ARG_ORDER,
    scalars=ARG_SCALARS,
    produces=ARG_BUFFERS_OUT,
    consumes=ARG_BUFFERS_IN,
    buffers=ARG_BUFFERS_IN + ARG_BUFFERS_OUT,
    block=(16, 16)
)
