import math
import numpy as np
from numba import cuda, float32, int32


@cuda.jit
def mandelbrot_kernel(min_x, max_x, min_y, max_y, image, max_iter, palette, samples):
    height, width, _ = image.shape
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    startX, startY = cuda.grid(2)
    gridX, gridY = cuda.gridsize(2)

    for x in range(startX, width, gridX):
        for y in range(startY, height, gridY):
            r_accum = 0
            g_accum = 0
            b_accum = 0

            for sx in range(samples):
                for sy in range(samples):
                    # Supersampling offset
                    offset_x = (sx + 0.5) / samples
                    offset_y = (sy + 0.5) / samples

                    real = min_x + (x + offset_x) * pixel_size_x
                    imag = min_y + (y + offset_y) * pixel_size_y
                    zr = real
                    zi = imag
                    i = 0
                    for i in range(max_iter):
                        if zr * zr + zi * zi >= 4.0:
                            break
                        temp = zr * zr - zi * zi + real
                        zi = 2.0 * zr * zi + imag
                        zr = temp

                    if i == max_iter:
                        # Distance estimation for interior coloring
                        modulus = math.sqrt(zr * zr + zi * zi)
                        if modulus == 0:
                            modulus = 1e-6
                        dist = 0.5 * math.log(modulus) * modulus
                        shade = int(255 * math.exp(-dist))
                        shade = max(0, min(255, shade))
                        r, g, b = shade, shade, shade
                    else:
                        # Smooth Coloring
                        mag = zr * zr + zi * zi
                        log_zn = math.log(mag) / 2
                        nu = math.log(log_zn / math.log(2)) / math.log(2)
                        smooth = i + 1 - nu
                        norm = smooth / max_iter
                        idx = int(norm * 255)
                        idx = max(0, min(255, idx))
                        r = palette[idx, 0]
                        g = palette[idx, 1]
                        b = palette[idx, 2]
                    r_accum += r
                    g_accum += g
                    b_accum += b
            total_samples = samples * samples
            image[y, x, 0] = int(r_accum / total_samples)
            image[y, x, 1] = int(g_accum / total_samples)
            image[y, x, 2] = int(b_accum / total_samples)



def render_mandelbrot(width, height, max_iter, min_x, max_x, min_y, max_y, palette, samples=2):

    # Allocate GPU memory
    image_gpu = cuda.device_array((height, width, 3), dtype=np.uint8)
    palette_device = cuda.to_device(np.array(palette, dtype=np.uint8))
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(width / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(height / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Launch kernel
    mandelbrot_kernel[blockspergrid, threadsperblock](float32(min_x),
                                                      float32(max_x),
                                                      float32(min_y),
                                                      float32(max_y),
                                                      image_gpu,
                                                      int32(max_iter),
                                                      palette_device,
                                                      int32(samples))
    cuda.synchronize()

    # Copy result to host
    image = image_gpu.copy_to_host()

    return image
