from kernel_sources.registry import register_kernel

SRC = r"""
#ifdef USE_DOUBLE
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  typedef double real_t;
#else
  typedef float  real_t;
#endif

__kernel void ssaa_resolve(
    const int samples,
    const int W_lo,
    const int H_lo,
    __global const real_t* src_hi,
    __global real_t* dst_lo
    )
{
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= W_lo || y >= H_lo) return;

    const int S = samples;
    const int W_hi = W_lo * S;
    const int hi_y0 = y * S;
    const int hi_x0 = x * S;

    real_t acc = (real_t)0;
    for (int j = 0; j < S; ++j) {
        const int row = (hi_y0 + j) * W_hi;
        for (int i = 0; i < S; ++i) {
            acc += src_hi[row + (hi_x0 + i)];
        }
    }
    dst_lo[y * W_lo + x] = acc / (real_t)(S * S);
}
"""

KERNEL_NAME = "ssaa_resolve"

ARG_SCALARS = ["samples", "width", "height"]    # width= W_lo, height= H_lo
ARG_BUFFERS_IN = ["src_hi"]
ARG_BUFFERS_OUT = ["dst_lo"]

ARG_ORDER = ARG_SCALARS + ARG_BUFFERS_IN + ARG_BUFFERS_OUT

opts_f32 = ["-cl-fast-relaxed-math"]
opts_f64 = opts_f32 + ["-D", "USE_DOUBLE=1"]

register_kernel(
    fractal="common",
    op_name="supersample",
    backend="opencl",
    precision="f32",
    func={"src": SRC, "kernel_name": KERNEL_NAME, "build_options": opts_f32},
    arg_order=ARG_ORDER,
    produces=ARG_BUFFERS_OUT,
    consumes=ARG_BUFFERS_IN,
    scalars=ARG_SCALARS,
    buffers=ARG_BUFFERS_IN + ARG_BUFFERS_OUT,
    block=(8, 8)
)

register_kernel(
    fractal="common",
    op_name="supersample",
    backend="opencl",
    precision="f64",
    func={"src": SRC, "kernel_name": KERNEL_NAME, "build_options": opts_f64},
    arg_order=ARG_ORDER,
    produces=ARG_BUFFERS_OUT,
    consumes=ARG_BUFFERS_IN,
    scalars=ARG_SCALARS,
    buffers=ARG_BUFFERS_IN + ARG_BUFFERS_OUT,
    block=(8, 8)
)
