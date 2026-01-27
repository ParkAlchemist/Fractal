from kernel_sources.registry import register_kernel

SRC = r"""
#ifdef USE_DOUBLE
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  typedef double real_t;
#else
  typedef float  real_t;
#endif

__kernel void minmax_normalize(
    real_t min_val,
    real_t max_val,
    __global const real_t* iter_smooth,
    __global real_t* iter_norm)
{
    const int W = get_global_size(0);
    const int H = get_global_size(1);
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= W || y >= H) return;

    const real_t mn = min_val;
    const real_t mx = max_val;
    real_t t = (real_t)0;
    if (mx > mn) {
        t = (iter_smooth[y * W + x] - mn) / (mx - mn);
        if (t < (real_t)0) t = (real_t)0;
        if (t > (real_t)1) t = (real_t)1;
    }
    iter_norm[y * W + x] = t;
}
"""

KERNEL_NAME = "minmax_normalize"

ARG_SCALARS = ["min_val", "max_val"]
ARG_BUFFERS_IN = ["iter_smooth"]
ARG_BUFFERS_OUT = ["iter_norm"]

ARG_ORDER = ARG_SCALARS + ARG_BUFFERS_IN + ARG_BUFFERS_OUT

opts_f32 = ["-cl-fast-relaxed-math"]
opts_f64 = opts_f32 + ["-D", "USE_DOUBLE=1"]

register_kernel(
    fractal="mandelbrot",
    op_name="normalize",
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
    fractal="mandelbrot",
    op_name="normalize",
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
