import numpy as np
from numba import cuda


@cuda.jit
def mandelbrot_kernel(min_x, max_x, min_y, max_y, image, max_iter):
    height, width = image.shape
    pixel_size_x = (max_x - min_x) / width
    pixel_size_y = (max_y - min_y) / height

    startX, startY = cuda.grid(2)
    gridX, gridY = cuda.gridsize(2)

    for x in range(startX, width, gridX):
        real = min_x + x * pixel_size_x
        for y in range(startY, height, gridY):
            imag = min_y + y * pixel_size_y
            zr = real
            zi = imag
            for i in range(max_iter):
                if zr * zr + zi * zi >= 4:
                    image[y, x] = i
                    break
                temp = zr * zr - zi * zi + real
                zi = 2.0 * zr * zi + imag
                zr = temp
            else:
                image[y, x] = max_iter

def render_mandelbrot(width, height, max_iter, min_x, max_x, min_y, max_y):

    # Allocate GPU memory
    image_gpu = cuda.device_array((height, width), dtype=np.uint16)
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(width / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(height / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    # Launch kernel
    mandelbrot_kernel[blockspergrid, threadsperblock](min_x, max_x, min_y, max_y, image_gpu, max_iter)
    cuda.synchronize()

    # Copy result to host
    image = image_gpu.copy_to_host()

    return image
