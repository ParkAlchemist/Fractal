import math
from numba import cuda

@cuda.jit(fastmath=True)
def mandelbrot_kernel_cuda_f32(min_x, max_x, min_y, max_y, iter_buf, max_iter, samples):
    H, W = iter_buf.shape
    psx = (max_x - min_x) / float(W)
    psy = (max_y - min_y) / float(H)
    startX, startY = cuda.grid(2)
    gridX, gridY = cuda.gridsize(2)
    eps = 1e-20
    for x in range(startX, W, gridX):
        for y in range(startY, H, gridY):
            accum = 0.0
            for sx in range(samples):
                for sy in range(samples):
                    ox = (sx + 0.5) / samples
                    oy = (sy + 0.5) / samples
                    real = min_x + (x + ox) * psx
                    imag = min_y + (y + oy) * psy
                    zr = real; zi = imag; i = 0
                    while zr*zr + zi*zi <= 4.0 and i < max_iter:
                        tmp = zr*zr - zi*zi + real
                        zi = 2.0*zr*zi + imag
                        zr = tmp
                        i += 1
                    mag2 = zr*zr + zi*zi
                    if mag2 < eps: mag2 = eps
                    log_zn = 0.5 * math.log(mag2)
                    nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
                    smooth = (i + 1) - nu if i < max_iter else i
                    accum += smooth / max_iter
            iter_buf[y, x] = accum / (samples * samples)

@cuda.jit(fastmath=True)
def mandelbrot_kernel_cuda_f64(min_x, max_x, min_y, max_y, iter_buf, max_iter, samples):
    H, W = iter_buf.shape
    psx = (max_x - min_x) / float(W)
    psy = (max_y - min_y) / float(H)
    startX, startY = cuda.grid(2)
    gridX, gridY = cuda.gridsize(2)
    eps = 1e-20
    for x in range(startX, W, gridX):
        for y in range(startY, H, gridY):
            accum = 0.0
            for sx in range(samples):
                for sy in range(samples):
                    ox = (sx + 0.5) / samples
                    oy = (sy + 0.5) / samples
                    real = min_x + (x + ox) * psx
                    imag = min_y + (y + oy) * psy
                    zr = real; zi = imag; i = 0
                    while zr*zr + zi*zi <= 4.0 and i < max_iter:
                        tmp = zr*zr - zi*zi + real
                        zi = 2.0*zr*zi + imag
                        zr = tmp
                        i += 1
                    mag2 = zr*zr + zi*zi
                    if mag2 < eps: mag2 = eps
                    log_zn = 0.5 * math.log(mag2)
                    nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
                    smooth = (i + 1) - nu if i < max_iter else i
                    accum += smooth / max_iter
            iter_buf[y, x] = accum / (samples * samples)
