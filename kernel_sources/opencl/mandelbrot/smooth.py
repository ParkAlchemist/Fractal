from kernel_sources.registry import register_kernel

SRC = r"""
#ifdef USE_DOUBLE
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  typedef double real_t;
#else
  typedef float  real_t;
#endif

__kernel void mandelbrot_smooth(
    real_t max_iter,
    __global const real_t* iter_raw,
    __global const real_t* z_mag,
    __global real_t* iter_smooth)
{
    const int W = get_global_size(0);
    const int H = get_global_size(1);
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= W || y >= H) return;

    const real_t n = iter_raw[y * W + x];
    if (n >= max_iter) {
        iter_smooth[y * W + x] = n; // interior
        return;
    }
    const real_t r = z_mag[y * W + x];
    const real_t eps = (real_t)1e-20;
    iter_smooth[y * W + x] = n + (real_t)1 - log(log(r + eps)) / log((real_t)2);
}
"""

KERNEL_NAME = "mandelbrot_smooth"

ARG_SCALARS = ["max_iter"]
ARG_BUFFERS_IN = ["iter_raw","z_mag"]
ARG_BUFFERS_OUT = ["iter_smooth"]

ARG_ORDER = ARG_SCALARS + ARG_BUFFERS_IN + ARG_BUFFERS_OUT

@register_kernel(backend="OpenCL", fractal="mandelbrot", operation="smooth")
def get_kernel(precision: str):
    opts = ["-cl-fast-relaxed-math"]
    if precision == "f64":
        opts += ["-D", "USE_DOUBLE=1"]
    func = {"src": SRC, "kernel_name": KERNEL_NAME, "build_options": opts}
    return {
        "func": func,
        "name": KERNEL_NAME,
        "arg_order": ARG_ORDER,
        "produces": ARG_BUFFERS_OUT,
        "consumes": ARG_BUFFERS_IN,
        "scalars": ARG_SCALARS,
        "buffers": ARG_BUFFERS_IN + ARG_BUFFERS_OUT,
        "block": (8, 8),
    }

opts_f32 = ["-cl-fast-relaxed-math"]
opts_f64 = opts_f32 + ["-D", "USE_DOUBLE=1"]

register_kernel(
    fractal="mandelbrot",
    op_name="smooth",
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
    op_name="smooth",
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
