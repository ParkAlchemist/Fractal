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

# Perturbation returns int32 iteration counts (will be normalized by backend)
def mandelbrot_kernel_cpu_perturb(zref, c_ref, width, height, c0, step_x, step_y,
                                  max_iter, order=2, w_fallback_thresh=1e-6):
    zr = zref[:max_iter, 0]
    zi = zref[:max_iter, 1]
    cr, ci = c_ref.real, c_ref.imag
    out = np.empty((height, width), dtype=np.int32)
    for iy in range(height):
        for ix in range(width):
            c = c0 + ix*step_x + iy*step_y
            dcr = c.real - cr
            dci = c.imag - ci
            wr = 0.0; wi = 0.0
            n = 0
            while n < max_iter:
                zrn, zin = zr[n], zi[n]
                zrx, zry = zrn + wr, zin + wi
                if zrx*zrx + zry*zry > 4.0: break
                tr = 2.0*zrn*wr - 2.0*zin*wi
                ti = 2.0*zrn*wi + 2.0*zin*wr
                if order >= 2:
                    w2r = wr*wr - wi*wi
                    w2i = 2.0*wr*wi
                    tr += w2r; ti += w2i
                wr = tr + dcr
                wi = ti + dci
                if max(abs(wr), abs(wi)) > w_fallback_thresh:
                    zf_r, zf_i = zrx, zry
                    n += 1
                    while n < max_iter:
                        zr2 = zf_r*zf_r - zf_i*zf_i + c.real
                        zi2 = 2.0*zf_r*zf_i + c.imag
                        zf_r, zf_i = zr2, zi2
                        if zf_r*zf_r + zf_i*zf_i > 4.0: break
                        n += 1
                    break
                n += 1
            out[iy, ix] = n
    return out
