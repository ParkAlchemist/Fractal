from numba import njit, prange

from kernel_sources.registry import register_kernel


ARG_SCALARS = ["samples"]
ARG_BUFFERS_IN = ["src_hi"]
ARG_BUFFERS_OUT = ["dst_lo"]

ARG_ORDER = ARG_SCALARS + ARG_BUFFERS_IN + ARG_BUFFERS_OUT

@njit(cache=True, fastmath=True, parallel=True)
def _ssaa_resolve(samples, src_hi, dst_lo):
    H_lo, W_lo = dst_lo.shape
    S = int(samples)
    for y in prange(H_lo):
        for x in range(W_lo):
            hi_y0 = y * S
            hi_x0 = x * S
            acc = src_hi.dtype.type(0.0)
            for j in range(S):
                for i in range(S):
                    acc += src_hi[hi_y0 + j, hi_x0 + i]
            dst_lo[y, x] = acc / (S * S)


register_kernel(
    fractal="common",
    op_name="supersample",
    backend="CPU",
    precision="f32",
    func=_ssaa_resolve,
    arg_order=ARG_ORDER,
    scalars=ARG_SCALARS,
    produces=ARG_BUFFERS_OUT,
    consumes=ARG_BUFFERS_IN,
    buffers=ARG_BUFFERS_IN + ARG_BUFFERS_OUT,
    block=None
)

register_kernel(
    fractal="common",
    op_name="supersample",
    backend="CPU",
    precision="f64",
    func=_ssaa_resolve,
    arg_order=ARG_ORDER,
    scalars=ARG_SCALARS,
    produces=ARG_BUFFERS_OUT,
    consumes=ARG_BUFFERS_IN,
    buffers=ARG_BUFFERS_IN + ARG_BUFFERS_OUT,
    block=None
)
