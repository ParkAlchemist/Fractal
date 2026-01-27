from kernel_sources.registry import register_kernel

SRC = r"""
#ifdef USE_DOUBLE
  #pragma OPENCL EXTENSION cl_khr_fp64 : enable
  typedef double real_t;
#else
  typedef float  real_t;
#endif

__kernel void mandelbrot_iter(
    real_t min_x, real_t max_x, real_t min_y, real_t max_y,
    real_t max_iter, real_t bailout,
    __global real_t* iter_raw, __global real_t* z_mag)
{
    const int W = get_global_size(0);
    const int H = get_global_size(1);
    const int x = get_global_id(0);
    const int y = get_global_id(1);
    if (x >= W || y >= H) return;

    real_t u = (real_t)x / (real_t)(W - 1);
    real_t v = (real_t)y / (real_t)(H - 1);
    real_t cr = min_x + (max_x - min_x) * u;
    real_t ci = min_y + (max_y - min_y) * v;

    real_t zr = 0, zi = 0;
    int n = 0;
    for (; n < (int)max_iter; ++n) {
        real_t zr2 = zr*zr - zi*zi + cr;
        real_t zi2 = (real_t)2 * zr*zi + ci;
        zr = zr2; zi = zi2;
        if (zr*zr + zi*zi > bailout) break;
    }
    iter_raw[y * W + x] = (real_t)n;
    z_mag[y * W + x]    = sqrt(zr*zr + zi*zi);
}
"""

KERNEL_NAME = "mandelbrot_iter"

ARG_SCALARS = [
    "min_x", "max_x", "min_y", "max_y",
    "max_iter", "bailout"
]
ARG_BUFFERS_IN = []
ARG_BUFFERS_OUT = ["iter_raw", "z_mag"]

ARG_ORDER = ARG_SCALARS + ARG_BUFFERS_IN + ARG_BUFFERS_OUT

opts_f32 = ["-cl-fast-relaxed-math"]
opts_f64 = opts_f32 + ["-D", "USE_DOUBLE=1"]

register_kernel(
    fractal="mandelbrot",
    op_name="iter",
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
    op_name="iter",
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
