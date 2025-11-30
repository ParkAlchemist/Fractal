import math
from mpmath import mp
from enum import Enum
import numpy as np
from numba import cuda, njit, prange
from multiprocessing import Pool, cpu_count

class Kernel(Enum):
    AUTO = 0
    OPENCL = 1
    CUDA = 2
    CPU = 3

# --------------------------------------------------------
# ---------------------- OpenCL KERNELS ------------------
# --------------------------------------------------------

# -------------- Single precision -------------------
mandelbrot_kernel_cl_float = """
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
mandelbrot_kernel_cl_double = """
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

# --------------------------------------------------------
# ---------------------- CPU KERNELS ---------------------
# --------------------------------------------------------

# ---------------- Single and Double precision ------------
@njit(parallel=True, fastmath=True)
def mandelbrot_kernel_cpu(real_grid, imag_grid, width, height, max_iter, samples, precision):
    iter_buf = np.zeros((height, width), dtype=precision)
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

# ---------------- Arbitrary precision ---------------
def mandelbrot_kernel_cpu_mp(min_x, max_x, min_y, max_y, width, height, max_iter, samples, zoom_factor):
    digits = max(50, int(math.log10(zoom_factor) * 4))
    mp.dps = digits

    pixel_size_x = mp.mpf(max_x - min_x) / width
    pixel_size_y = mp.mpf(max_y - min_y) / height

    # Split into tiles for parallel processing
    num_workers = max(1, cpu_count() - 1)
    tile_size = max(32, width // num_workers)

    tasks = []
    for x_start in range(0, width, tile_size):
        for y_start in range(0, height, tile_size):
            x_end = min(x_start + tile_size, width)
            y_end = min(y_start + tile_size, height)
            tasks.append((x_start, x_end, y_start, y_end,
                          min_x, min_y, pixel_size_x, pixel_size_y,
                          width, height, max_iter, samples))

    iter_buf = np.zeros((height, width), dtype=np.float64)

    with Pool(processes=num_workers) as pool:
        for tile_x_start, tile_y_start, tile_buf in pool.imap_unordered(_compute_tile, tasks):
            h, w = tile_buf.shape
            iter_buf[tile_y_start:tile_y_start+h, tile_x_start:tile_x_start+w] = tile_buf

    return iter_buf


def _compute_tile(args):
    (tile_x_start, tile_x_end, tile_y_start, tile_y_end,
     min_x, min_y, pixel_size_x, pixel_size_y,
     width, height, max_iter, samples) = args

    tile_buf = np.zeros((tile_y_end - tile_y_start, tile_x_end - tile_x_start), dtype=np.float64)

    for y in range(tile_y_start, tile_y_end):
        for x in range(tile_x_start, tile_x_end):
            accum = mp.mpf(0)
            for sx in range(samples):
                for sy in range(samples):
                    offset_x = (sx + 0.5) / samples
                    offset_y = (sy + 0.5) / samples

                    real = mp.mpf(min_x) + (x + offset_x) * pixel_size_x
                    imag = mp.mpf(min_y) + (y + offset_y) * pixel_size_y
                    zr = real
                    zi = imag

                    i = 0
                    while zr * zr + zi * zi <= 4 and i < max_iter:
                        temp = zr * zr - zi * zi + real
                        zi = 2 * zr * zi + imag
                        zr = temp
                        i += 1

                    mag2 = zr * zr + zi * zi
                    if mag2 < mp.mpf('1e-50'):
                        mag2 = mp.mpf('1e-50')

                    log_zn = mp.log(mag2) / 2
                    nu = mp.log(log_zn / mp.log(2)) / mp.log(2)
                    smooth_iter = (i + 1) - nu if i < max_iter else i

                    accum += smooth_iter / max_iter

            tile_buf[y - tile_y_start, x - tile_x_start] = float(accum / (samples * samples))

    return tile_x_start, tile_y_start, tile_buf
