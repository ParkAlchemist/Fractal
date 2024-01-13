import PIL.ImageOps
from PIL import Image
from mandelbrot import MandelbrotSet
from julia import JuliaSet
import numpy as np
import multiprocessing as mp
from scipy.interpolate import interp1d


def create_smooth_gradient(palette, resolution=1000):
    indices = np.linspace(0, len(palette) - 1, len(palette))
    interp_func = interp1d(indices, palette, kind='cubic', axis=0, fill_value="extrapolate")
    smooth_gradient = interp_func(np.linspace(0, len(palette) - 1, resolution))
    return smooth_gradient.astype(int)


width, height = 1024, 1024
BLACK_AND_WHITE = "1"
GRAYSCALE = "L"
JULIA_CONST = complex(0.0, 0.8)
CENTER = 0 + 0j
VPWIDTH = 3.5
VPHEIGHT = (width / VPWIDTH)*height
VPSCALE = VPWIDTH / width
OFFSET = CENTER + complex(-VPWIDTH, VPHEIGHT) / 2
MAX_ITER = 1000

colors = [(0, 0, 0), (66, 30, 15), (25, 7, 26), (9, 1, 47),
          (4, 4, 73), (0, 7, 100), (12, 44, 138),
          (24, 82, 177), (57, 125, 209), (134, 181, 229),
          (211, 236, 248), (241, 233, 191), (248, 201, 95),
          (255, 170, 0), (204, 128, 0), (153, 87, 0), (106, 52, 3)]

PROCESS_COUNT = 8

# Create a smooth gradient with higher resolution
smooth_gradient = create_smooth_gradient(colors, resolution=1000)


def multiprocess_render_mandelbrot(pixel_start, pixel_count, arr):
    mandelbrot_set = MandelbrotSet(max_iterations=MAX_ITER, escape_radius=1000)
    k_values = np.arange(pixel_count)

    x = np.mod(pixel_start + k_values, width)
    y = (pixel_start + k_values) // width
    re = (x - width / 2)
    im = (height / 2 - y)
    c = (re + 1j * im) * VPSCALE + (0 if CENTER == complex(0, 0) else OFFSET)

    instability = 1 - mandelbrot_set.stability(c, smooth=True)
    indices = np.where((instability == 1.0) | (instability == 0.0), 0,
                       ((instability * MAX_ITER) % 16) + 1)
    arr[pixel_start:pixel_start + pixel_count] = indices.astype(int)


def multiprocess_render_julia(pixel_start, pixel_count, arr):
    julia_set = JuliaSet(max_iterations=MAX_ITER, escape_radius=1000,
                         julia_constant=JULIA_CONST)
    k_values = np.arange(pixel_count)

    x = np.mod(pixel_start + k_values, width)
    y = (pixel_start + k_values) // width
    re = (x - width / 2)
    im = (height / 2 - y)
    c = (re + 1j * im) * VPSCALE + 0 if CENTER == complex(0, 0) else OFFSET

    instability = 1 - julia_set.stability(c, smooth=True)
    indices = np.where((instability == 1.0) | (instability == 0.0), 0,
                       ((instability * MAX_ITER) % 16) + 1)
    arr[pixel_start:pixel_start + pixel_count] = indices.astype(int)


if __name__ == '__main__':
    processes = []
    work_set_size = int((width * height) / PROCESS_COUNT)

    indices_buff = mp.Array('i', width * height)

    for i in range(PROCESS_COUNT):
        p = mp.Process(target=multiprocess_render_julia,
                       args=(i * work_set_size, work_set_size, indices_buff))
        processes.append(p)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    # convert mp Array to list
    color_indices = np.frombuffer(indices_buff.get_obj(), dtype="int32")

    # Reshape color_indices to match image dimensions
    color_indices = color_indices.reshape((height, width))

    pixels = []
    for row in color_indices:
        pixel_row = []
        for col in row:
            pixel_row.append(colors[col])
        pixels.append(pixel_row)

    # Create an image using the smooth gradient
    img = Image.fromarray(np.uint8(pixels))

    # Show the resulting image
    img.show()

