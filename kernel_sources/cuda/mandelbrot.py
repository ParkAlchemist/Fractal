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

@cuda.jit(device=True, inline=True)
def _cnorm2(x, y): return x*x + y*y

@cuda.jit
def mandelbrot_kernel_cuda_perturb_f64(
    zref, ref_len,
    c_ref_r, c_ref_i,
    c0_r, c0_i,
    step_x_r, step_x_i,
    step_y_r, step_y_i,
    width, height, max_iter,
    order, w_fallback_thresh,
    iter_out
):
    ix = cuda.blockIdx.x * cuda.blockDim.x + cuda.threadIdx.x
    iy = cuda.blockIdx.y * cuda.blockDim.y + cuda.threadIdx.y
    if ix >= width or iy >= height: return

    c_r = c0_r + step_x_r * ix + step_y_r * iy
    c_i = c0_i + step_x_i * ix + step_y_i * iy
    dc_r = c_r - c_ref_r
    dc_i = c_i - c_ref_i

    wr = 0.0; wi = 0.0; n = 0
    while n < max_iter:
        zr_n = zref[n, 0]; zi_n = zref[n, 1]
        zrx = zr_n + wr; zry = zi_n + wi
        if _cnorm2(zrx, zry) > 4.0: break
        t_r = 2.0 * zr_n * wr - 2.0 * zi_n * wi
        t_i = 2.0 * zr_n * wi + 2.0 * zi_n * wr
        if order >= 2:
            w2_r = wr*wr - wi*wi
            w2_i = 2.0 * wr * wi
            t_r += w2_r; t_i += w2_i
        wr = t_r + dc_r
        wi = t_i + dc_i
        if (wr if wr >= 0 else -wr) > w_fallback_thresh or (wi if wi >= 0 else -wi) > w_fallback_thresh:
            zf_r = zrx; zf_i = zry
            n += 1
            while n < max_iter:
                zr2 = zf_r*zf_r - zf_i*zf_i + c_r
                zi2 = 2.0*zf_r*zf_i + c_i
                zf_r = zr2; zf_i = zi2
                if _cnorm2(zf_r, zf_i) > 4.0: break
                n += 1
            break
        n += 1
    iter_out[iy, ix] = n
