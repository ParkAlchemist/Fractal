import math
import numpy as np
from numba import njit, prange

@njit(parallel=True, fastmath=True)
def mandelbrot_kernel_cpu_f32(real_grid, imag_grid, width, height, max_iter, samples):
    out = np.zeros((height, width), dtype=np.float32)
    eps = 1e-20
    for y in prange(height):
        for x in range(width):
            accum = 0.0
            for sx in range(samples):
                for sy in range(samples):
                    ox = (sx + 0.5) / samples
                    oy = (sy + 0.5) / samples
                    zr = real_grid[y, x] + ox * (real_grid[0, 1] - real_grid[0, 0])
                    zi = imag_grid[y, x] + oy * (imag_grid[1, 0] - imag_grid[0, 0])
                    i = 0
                    while zr*zr + zi*zi <= 4.0 and i < max_iter:
                        tmp = zr*zr - zi*zi + real_grid[y, x]
                        zi = 2.0*zr*zi + imag_grid[y, x]
                        zr = tmp
                        i += 1
                    mag2 = zr*zr + zi*zi
                    if mag2 < eps: mag2 = eps
                    log_zn = 0.5 * math.log(mag2)
                    nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
                    smooth = (i + 1) - nu if i < max_iter else i
                    accum += smooth / max_iter
            out[y, x] = accum / (samples * samples)
    return out

@njit(parallel=True, fastmath=True)
def mandelbrot_kernel_cpu_f64(real_grid, imag_grid, width, height, max_iter, samples):
    out = np.zeros((height, width), dtype=np.float64)
    eps = 1e-20
    for y in prange(height):
        for x in range(width):
            accum = 0.0
            for sx in range(samples):
                for sy in range(samples):
                    ox = (sx + 0.5) / samples
                    oy = (sy + 0.5) / samples
                    zr = real_grid[y, x] + ox * (real_grid[0, 1] - real_grid[0, 0])
                    zi = imag_grid[y, x] + oy * (imag_grid[1, 0] - imag_grid[0, 0])
                    i = 0
                    while zr*zr + zi*zi <= 4.0 and i < max_iter:
                        tmp = zr*zr - zi*zi + real_grid[y, x]
                        zi = 2.0*zr*zi + imag_grid[y, x]
                        zr = tmp
                        i += 1
                    mag2 = zr*zr + zi*zi
                    if mag2 < eps: mag2 = eps
                    log_zn = 0.5 * math.log(mag2)
                    nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
                    smooth = (i + 1) - nu if i < max_iter else i
                    accum += smooth / max_iter
            out[y, x] = accum / (samples * samples)
    return out
