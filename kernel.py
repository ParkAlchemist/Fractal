import math
from enum import Enum
import numpy as np
from numba import cuda, njit, prange

class Kernel(Enum):
    AUTO = 0
    OPENCL = 1
    CUDA = 2
    CPU = 3

# --------------------------------------------------------
# ---------------------- OpenCL KERNELS ------------------
# --------------------------------------------------------

# -------------- Single precision -------------------
mandelbrot_kernel_cl_f32 = """
__kernel void mandelbrot_kernel(
    const float min_x, const float max_x,
    const float min_y, const float max_y,
    __global float *iter_buf,
    const int width, const int height,
    const int max_iter,
    const int samples)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float pixel_size_x = (max_x - min_x) / (float)width;
    float pixel_size_y = (max_y - min_y) / (float)height;

    float accum = 0.0f;
    float total_samples = (float)(samples * samples);

    for (int sx = 0; sx < samples; sx++)
    for (int sy = 0; sy < samples; sy++)
    {
        float offset_x = ((float)sx + 0.5f) / (float)samples;
        float offset_y = ((float)sy + 0.5f) / (float)samples;
        float real = min_x + (x + offset_x) * pixel_size_x;
        float imag = min_y + (y + offset_y) * pixel_size_y;

        float zr = real;
        float zi = imag;
        int i = 0;
        for (i = 0; i < max_iter; i++)
        {
            float zr2 = zr * zr;
            float zi2 = zi * zi;
            if (zr2 + zi2 >= 4.0f) break;
            float temp = zr2 - zi2 + real;
            zi = 2.0f * zr * zi + imag;
            zr = temp;
        }

        float mag2 = zr*zr + zi*zi;
        if (mag2 < 1e-20f) mag2 = 1e-20f;

        float log_zn = 0.5f * log(mag2);
        float nu = log(log_zn / 0.69314718f) / 0.69314718f;
        float smooth_iter = i < max_iter ? (float)(i + 1) - nu : (float)i;

        accum += smooth_iter / (float)max_iter;
    }

    int idx = y * width + x;
    iter_buf[idx] = accum / total_samples;
}
"""

# ------------------ Double precision ----------------
mandelbrot_kernel_cl_f64 = """
__kernel void mandelbrot_kernel(
    const double min_x, const double max_x,
    const double min_y, const double max_y,
    __global double *iter_buf,
    const int width, const int height,
    const int max_iter,
    const int samples)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    double pixel_size_x = (max_x - min_x) / (double)width;
    double pixel_size_y = (max_y - min_y) / (double)height;

    double accum = 0.0f;
    double total_samples = (double)(samples * samples);

    for (int sx = 0; sx < samples; sx++)
    for (int sy = 0; sy < samples; sy++)
    {
        double offset_x = ((double)sx + 0.5f) / (double)samples;
        double offset_y = ((double)sy + 0.5f) / (double)samples;
        double real = min_x + (x + offset_x) * pixel_size_x;
        double imag = min_y + (y + offset_y) * pixel_size_y;

        double zr = real;
        double zi = imag;
        int i = 0;
        for (i = 0; i < max_iter; i++)
        {
            double zr2 = zr * zr;
            double zi2 = zi * zi;
            if (zr2 + zi2 >= 4.0f) break;
            double temp = zr2 - zi2 + real;
            zi = 2.0f * zr * zi + imag;
            zr = temp;
        }

        double mag2 = zr*zr + zi*zi;
        if (mag2 < 1e-20f) mag2 = 1e-20f;

        double log_zn = 0.5f * log(mag2);
        double nu = log(log_zn / 0.69314718f) / 0.69314718f;
        double smooth_iter = i < max_iter ? (double)(i + 1) - nu : (double)i;

        accum += smooth_iter / (double)max_iter;
    }

    int idx = y * width + x;
    iter_buf[idx] = accum / total_samples;
}
"""

#------------------ Perturbation -------------------------
mandelbrot_kernel_cl_perturb = """
__kernel void mandelbrot_perturb_kernel(
    const double cRef_r, const double cRef_i,
    __global const double* zref_r,
    __global const double* zref_i,
    const int refLen,
    const double min_x, const double max_x,
    const double min_y, const double max_y,
    __global double* iter_buf,
    const int width, const int height,
    const int max_iter,
    const int samples,
    const int order,
    const double wFallbackThresh
){
    
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    double pix_x = (max_x - min_x) / (double)width;
    double pix_y = (max_y - min_y) / (double)height;

    double accum = 0.0;
    double total_samples = (double)(samples * samples);
    
    for (int sx = 0; sx < samples; ++sx)
    for (int sy = 0; sy < samples; ++sy) {
        double offx = ((double)sx + 0.5) / (double)samples;
        double offy = ((double)sy + 0.5) / (double)samples;

        double c_r = min_x + (x + offx) * pix_x;
        double c_i = min_y + (y + offy) * pix_y;
        double dc_r = c_r - cRef_r;
        double dc_i = c_i - cRef_i;

        double wr = 0.0, wi = 0.0;
        int n = 0;
        int usedFallback = 0;
        double zFinal_r = 0.0, zFinal_i = 0.0;
        
        while (n < max_iter) {
            double zr_n = zref_r[n];
            double zi_n = zref_i[n];
            double zrx = zr_n + wr;
            double zry = zi_n + wi;

            if (zrx*zrx + zry*zry > 4.0) {
                zFinal_r = zrx; zFinal_i = zry;
                break;
            }

            double t_r = 2.0 * zr_n * wr - 2.0 * zi_n * wi;
            double t_i = 2.0 * zr_n * wi + 2.0 * zi_n * wr;

            if (order >= 2) {
                double w2_r = wr*wr - wi*wi;
                double w2_i = 2.0 * wr * wi;
                t_r += w2_r; t_i += w2_i;
            }
            
            wr = t_r + dc_r;
            wi = t_i + dc_i;

            if (fabs(wr) > wFallbackThresh || fabs(wi) > wFallbackThresh) {
                double zf_r = zrx, zf_i = zry;
                n += 1;
                while (n < max_iter) {
                    double zr2 = zf_r*zf_r - zf_i*zf_i + c_r;
                    double zi2 = 2.0*zf_r*zf_i + c_i;
                    zf_r = zr2; zf_i = zi2;
                    if (zf_r*zf_r + zf_i*zf_i > 4.0) break;
                    n += 1;
                }
                usedFallback = 1;
                zFinal_r = zf_r; zFinal_i = zf_i;
                break;
            }

            n += 1;
        }
        
        double eps = 1e-20;
        double mag2 = zFinal_r*zFinal_r + zFinal_i*zFinal_i;
        if (mag2 < eps) mag2 = eps;
        double log_zn = 0.5 * log(mag2);
        double nu = log(log_zn / 0.6931471805599453) / 0.6931471805599453; // log2
        double smooth_iter = (n < max_iter) ? ((double)(n + 1) - nu) : (double)n;

        accum += smooth_iter / (double)max_iter;
    }

    int idx = y * width + x;
    iter_buf[idx] = accum / total_samples;
}
"""

# --------------------------------------------------------
# ---------------------- CUDA KERNELS --------------------
# --------------------------------------------------------

# ----------------- Single precision --------------------
@cuda.jit(fastmath=True)
def mandelbrot_kernel_cuda_f32(min_x, max_x, min_y, max_y, iter_buf, max_iter, samples):
    height, width = iter_buf.shape
    pixel_size_x = (max_x - min_x) / float(width)
    pixel_size_y = (max_y - min_y) / float(height)

    startX, startY = cuda.grid(2)
    gridX, gridY = cuda.gridsize(2)

    eps = 1e-20

    for x in range(startX, width, gridX):
        for y in range(startY, height, gridY):
            accum = 0.0
            for sx in range(samples):
                for sy in range(samples):
                    offset_x = (sx + 0.5) / samples
                    offset_y = (sy + 0.5) / samples

                    real = min_x + (x + offset_x) * pixel_size_x
                    imag = min_y + (y + offset_y) * pixel_size_y
                    zr = real
                    zi = imag

                    i = 0
                    while zr * zr + zi * zi <= 4.0 and i < max_iter:
                        temp = zr * zr - zi * zi + real
                        zi = 2.0 * zr * zi + imag
                        zr = temp
                        i += 1

                    mag2 = zr * zr + zi * zi
                    if mag2 < eps:
                        mag2 = eps

                    # Smooth coloring
                    log_zn = 0.5 * math.log(mag2)
                    nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
                    smooth_iter = (i + 1) - nu if i < max_iter else i

                    accum += smooth_iter / max_iter

            iter_buf[y, x] = accum / (samples * samples)

# ------------------ Double Precision ------------------
@cuda.jit(fastmath=True)
def mandelbrot_kernel_cuda_f64(min_x, max_x, min_y, max_y, iter_buf, max_iter, samples):
    height, width = iter_buf.shape
    pixel_size_x = (max_x - min_x) / float(width)
    pixel_size_y = (max_y - min_y) / float(height)

    startX, startY = cuda.grid(2)
    gridX, gridY = cuda.gridsize(2)

    eps = 1e-20

    for x in range(startX, width, gridX):
        for y in range(startY, height, gridY):
            accum = 0.0
            for sx in range(samples):
                for sy in range(samples):
                    offset_x = (sx + 0.5) / samples
                    offset_y = (sy + 0.5) / samples

                    real = min_x + (x + offset_x) * pixel_size_x
                    imag = min_y + (y + offset_y) * pixel_size_y
                    zr = real
                    zi = imag

                    i = 0
                    while zr * zr + zi * zi <= 4.0 and i < max_iter:
                        temp = zr * zr - zi * zi + real
                        zi = 2.0 * zr * zi + imag
                        zr = temp
                        i += 1

                    mag2 = zr * zr + zi * zi
                    if mag2 < eps:
                        mag2 = eps

                    # Smooth coloring
                    log_zn = 0.5 * math.log(mag2)
                    nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
                    smooth_iter = (i + 1) - nu if i < max_iter else i

                    accum += smooth_iter / max_iter

            iter_buf[y, x] = accum / (samples * samples)


@cuda.jit(device=True, inline=True)
def _cnorm2(x, y):
    return x*x + y*y

# ------------- Perturbation ------------------
@cuda.jit(fastmath=True)
def mandelbrot_kernel_cuda_perturb(
    zref,       # (ref_len, 2) float64
    ref_len,
    c_ref_r, c_ref_i,
    c0_r, c0_i,
    step_x_r, step_x_i,
    step_y_r, step_y_i,
    width, height, max_iter,
    order, w_fallback_thresh,
    iter_out   # (height, width) int32
):
    ix = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    iy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if ix >= width or iy >= height:
        return

    # Map pixel -> c and Δc
    c_r = c0_r + step_x_r * ix + step_y_r * iy
    c_i = c0_i + step_x_i * ix + step_y_i * iy
    dc_r = c_r - c_ref_r
    dc_i = c_i - c_ref_i

    wr = 0.0
    wi = 0.0
    n = 0

    while n < max_iter:
        zr_n = zref[n, 0]
        zi_n = zref[n, 1]
        zrx = zr_n + wr
        zry = zi_n + wi
        if _cnorm2(zrx, zry) > 4.0:
            break

        # w_{n+1} = 2 z*_n w_n [+ w_n^2] + Δc
        t_r = 2.0 * zr_n * wr - 2.0 * zi_n * wi
        t_i = 2.0 * zr_n * wi + 2.0 * zi_n * wr

        if order >= 2:
            w2_r = wr * wr - wi * wi
            w2_i = 2.0 * wr * wi
            t_r += w2_r
            t_i += w2_i

        wr = t_r + dc_r
        wi = t_i + dc_i

        # Fallback to raw iteration for correctness
        if (wr if wr >= 0.0 else -wr) > w_fallback_thresh or (
        wi if wi >= 0.0 else -wi) > w_fallback_thresh:
            zf_r = zrx
            zf_i = zry
            n += 1
            while n < max_iter:
                zr2 = zf_r * zf_r - zf_i * zf_i + c_r
                zi2 = 2.0 * zf_r * zf_i + c_i
                zf_r = zr2
                zf_i = zi2
                if _cnorm2(zf_r, zf_i) > 4.0:
                    break
                n += 1
            break

        n += 1

    iter_out[iy, ix] = n

# --------------------------------------------------------
# ---------------------- CPU KERNELS ---------------------
# --------------------------------------------------------

# ---------------- Single precision ------------
@njit(parallel=True, fastmath=True)
def mandelbrot_kernel_cpu_f32(
        real_grid: np.ndarray, imag_grid: np.ndarray,
        width: int, height: int, max_iter: int, samples: int):
    iter_buf = np.zeros((height, width), dtype=np.float32)
    eps = 1e-20

    for y in prange(height):
        for x in range(width):
            accum = 0.0
            for sx in range(samples):
                for sy in range(samples):
                    offset_x = (sx + 0.5) / samples
                    offset_y = (sy + 0.5) / samples
                    zr = real_grid[y, x] + offset_x * (
                                real_grid[0, 1] - real_grid[0, 0])
                    zi = imag_grid[y, x] + offset_y * (
                                imag_grid[1, 0] - imag_grid[0, 0])

                    i = 0
                    while zr * zr + zi * zi <= 4.0 and i < max_iter:
                        temp = zr * zr - zi * zi + real_grid[y, x]
                        zi = 2.0 * zr * zi + imag_grid[y, x]
                        zr = temp
                        i += 1

                    mag2 = zr * zr + zi * zi
                    if mag2 < eps:
                        mag2 = eps

                    log_zn = 0.5 * math.log(mag2)
                    nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
                    smooth_iter = (i + 1) - nu if i < max_iter else i

                    accum += smooth_iter / max_iter

            iter_buf[y, x] = accum / (samples * samples)

    return iter_buf

# ---------------- Double precision ------------
@njit(parallel=True, fastmath=True)
def mandelbrot_kernel_cpu_f64(
        real_grid: np.ndarray, imag_grid: np.ndarray,
        width: int, height: int, max_iter: int, samples: int):
    iter_buf = np.zeros((height, width), dtype=np.float64)
    eps = 1e-20

    for y in prange(height):
        for x in range(width):
            accum = 0.0
            for sx in range(samples):
                for sy in range(samples):
                    offset_x = (sx + 0.5) / samples
                    offset_y = (sy + 0.5) / samples
                    zr = real_grid[y, x] + offset_x * (
                                real_grid[0, 1] - real_grid[0, 0])
                    zi = imag_grid[y, x] + offset_y * (
                                imag_grid[1, 0] - imag_grid[0, 0])

                    i = 0
                    while zr * zr + zi * zi <= 4.0 and i < max_iter:
                        temp = zr * zr - zi * zi + real_grid[y, x]
                        zi = 2.0 * zr * zi + imag_grid[y, x]
                        zr = temp
                        i += 1

                    mag2 = zr * zr + zi * zi
                    if mag2 < eps:
                        mag2 = eps

                    log_zn = 0.5 * math.log(mag2)
                    nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
                    smooth_iter = (i + 1) - nu if i < max_iter else i

                    accum += smooth_iter / max_iter

            iter_buf[y, x] = accum / (samples * samples)

    return iter_buf

# ---------------- Perturbation ---------------
@njit(parallel=True, fastmath=True)
def mandelbrot_kernel_cpu_perturb(
    zref: np.ndarray,          # shape (ref_len, 2), float64 (C-contiguous)
    c_ref: complex,
    width: int, height: int,
    c0: complex, step_x: complex, step_y: complex,
    max_iter: int, order: int = 2, w_fallback_thresh: float = 1e-6
) -> np.ndarray:
    """
    Parallelized (Numba) perturbation CPU kernel.
    Returns iteration counts (int32) of shape (height, width).
    """
    out = np.empty((height, width), dtype=np.int32)

    # Pull reference arrays as float64 (Numba-friendly)
    zr = zref[:max_iter, 0]
    zi = zref[:max_iter, 1]

    c_ref_r = c_ref.real
    c_ref_i = c_ref.imag

    # Precompute real/imag parts of steps (complex not fully optimized in nopython mode)
    sx_r = step_x.real
    sx_i = step_x.imag
    sy_r = step_y.real
    sy_i = step_y.imag
    c0_r = c0.real
    c0_i = c0.imag

    for iy in prange(height):
        for ix in range(width):
            # Map pixel -> c and Δc
            c_r = c0_r + sx_r * ix + sy_r * iy
            c_i = c0_i + sx_i * ix + sy_i * iy
            dc_r = c_r - c_ref_r
            dc_i = c_i - c_ref_i

            wr = 0.0
            wi = 0.0
            n = 0

            while n < max_iter:
                zr_n = zr[n]
                zi_n = zi[n]

                # z = z*_n + w
                zrx = zr_n + wr
                zry = zi_n + wi
                if zrx * zrx + zry * zry > 4.0:
                    break

                # w_{n+1} = 2*z*_n*w_n [+ w_n^2] + Δc
                t_r = 2.0 * zr_n * wr - 2.0 * zi_n * wi
                t_i = 2.0 * zr_n * wi + 2.0 * zi_n * wr

                if order >= 2:
                    w2_r = wr * wr - wi * wi
                    w2_i = 2.0 * wr * wi
                    t_r += w2_r
                    t_i += w2_i

                wr = t_r + dc_r
                wi = t_i + dc_i

                # Fallback to raw iteration if perturbation grows too big
                # Using branchless abs for Numba
                if (wr if wr >= 0.0 else -wr) > w_fallback_thresh or (wi if wi >= 0.0 else -wi) > w_fallback_thresh:
                    zf_r = zrx
                    zf_i = zry
                    n += 1
                    while n < max_iter:
                        zr2 = zf_r * zf_r - zf_i * zf_i + c_r
                        zi2 = 2.0 * zf_r * zf_i + c_i
                        zf_r = zr2
                        zf_i = zi2
                        if zf_r * zf_r + zf_i * zf_i > 4.0:
                            break
                        n += 1
                    break

                n += 1

            out[iy, ix] = n

    return out

