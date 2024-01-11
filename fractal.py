from PIL import Image, ImageEnhance
from mandelbrot import MandelbrotSet
from julia import JuliaSet
import numpy as np
import multiprocessing as mp


width, height = 800, 600
scale = 0.0035
BLACK_AND_WHITE = "1"
GRAYSCALE = "L"
PROCESS_COUNT = 8
JULIA_CONST = complex(-0.75, 0.11)


def multiprocess_render_mandelbrot(pixel_start, pixel_count, arr):
    mandelbrot_set = MandelbrotSet(max_iterations=64, escape_radius=1000)
    k_values = np.arange(pixel_count)

    x = np.mod(pixel_start + k_values, width)
    y = (pixel_start + k_values) // width
    re = scale * (x - width / 2)
    im = scale * (height / 2 - y)
    c = re + 1j * im

    instability = 1 - mandelbrot_set.stability(c, smooth=True)
    arr[pixel_start:pixel_start+pixel_count] = instability * 255


def multiprocess_render_julia(pixel_start, pixel_count, arr):
    mandelbrot_set = JuliaSet(max_iterations=128, escape_radius=1000,
                              julia_constant=JULIA_CONST)
    k_values = np.arange(pixel_count)

    x = np.mod(pixel_start + k_values, width)
    y = (pixel_start + k_values) // width
    re = scale * (x - width / 2)
    im = scale * (height / 2 - y)
    c = re + 1j * im

    instability = 1 - mandelbrot_set.stability(c, smooth=True)
    arr[pixel_start:pixel_start+pixel_count] = instability * 255


if __name__ == '__main__':

    processes = []
    work_set_size = int((width*height)/PROCESS_COUNT)
    pixels = mp.Array('f', range(width*height))

    for i in range(PROCESS_COUNT):
        p = mp.Process(target=multiprocess_render_julia,
                       args=(i*work_set_size, work_set_size, pixels))
        processes.append(p)

    for process in processes:
        process.start()

    for process in processes:
        process.join()

    image = Image.new(mode=GRAYSCALE, size=(width, height))
    image.putdata(pixels)
    image.show()
