from numba import njit, prange

from kernel_sources.registry import register_kernel


ARG_SCALARS = [
    "min_x", "max_x", "min_y", "max_y",
    "max_iter", "bailout",
]
ARG_BUFFERS_IN = []
ARG_BUFFERS_OUT = ["iter_raw", "z_mag"]

ARG_ORDER = ARG_SCALARS + ARG_BUFFERS_IN + ARG_BUFFERS_OUT


@njit(cache=True, fastmath=True, parallel=True)
def _mandelbrot_iter(min_x, max_x, min_y, max_y,
                     max_iter, bailout,
                     iter_raw, z_mag,):
    H, W = iter_raw.shape
    for y in prange(H):
        v = y / (H - 1.0)
        ci = min_y + (max_y - min_y) * v
        for x in range(W):
            u = x / (W - 1.0)
            cr = min_x + (max_x - min_x) * u

            zr = iter_raw.dtype.type(0.0)
            zi = iter_raw.dtype.type(0.0)
            n  = 0

            while n < int(max_iter) and zr*zr + zi*zi <= bailout:
                zr2 = zr*zr - zi*zi + cr
                zi2 = (iter_raw.dtype.type(2.0) * zr * zi) + ci
                zr, zi = zr2, zi2
                n += 1

            iter_raw[y, x] = n
            z_mag[y, x]    = (zr*zr + zi*zi)**0.5

register_kernel(
    fractal="mandelbrot",
    op_name="iter",
    backend="CPU",
    precision="f32",
    func=_mandelbrot_iter,
    arg_order=ARG_ORDER,
    scalars=ARG_SCALARS,
    produces=ARG_BUFFERS_OUT,
    consumes=ARG_BUFFERS_IN,
    buffers=ARG_BUFFERS_IN + ARG_BUFFERS_OUT,
    block=None
)

register_kernel(
    fractal="mandelbrot",
    op_name="iter",
    backend="CPU",
    precision="f64",
    func=_mandelbrot_iter,
    arg_order=ARG_ORDER,
    scalars=ARG_SCALARS,
    produces=ARG_BUFFERS_OUT,
    consumes=ARG_BUFFERS_IN,
    buffers=ARG_BUFFERS_IN + ARG_BUFFERS_OUT,
    block=None
)
