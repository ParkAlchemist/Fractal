from PIL import Image, ImageTk
import palettes
import numpy as np
from numba import cuda
import tkinter as tk
from tkinter import Canvas, Entry, Label, Button
import timeit


colors = [(0, 0, 0), (66, 30, 15), (25, 7, 26), (9, 1, 47),
          (4, 4, 73), (0, 7, 100), (12, 44, 138),
          (24, 82, 177), (57, 125, 209), (134, 181, 229),
          (211, 236, 248), (241, 233, 191), (248, 201, 95),
          (255, 170, 0), (204, 128, 0), (153, 87, 0), (106, 52, 3)]

MAX_ITER = 1000


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
            c = complex(real, imag)
            z = c
            for i in range(max_iter):
                if (z.real * z.real + z.imag * z.imag) >= 4:
                    image[y, x] = i
                    break
                z = z * z + c
            else:
                image[y, x] = max_iter


def render_mandelbrot(width, height, max_iter, min_x, max_x, min_y, max_y):
    image_gpu = cuda.device_array((height, width), dtype=np.uint8)
    threadsperblock = (16, 16)
    blockspergrid_x = int(np.ceil(width / threadsperblock[0]))
    blockspergrid_y = int(np.ceil(height / threadsperblock[1]))
    blockspergrid = (blockspergrid_x, blockspergrid_y)

    mandelbrot_kernel[blockspergrid, threadsperblock](min_x, max_x, min_y,
                                                      max_y, image_gpu,
                                                      max_iter)
    cuda.synchronize()

    # Copy the result back to the host
    image = image_gpu.copy_to_host()

    return image


def draw_mandelbrot(canvas, width, height, max_iter, min_x, max_x, min_y, max_y):
    image = render_mandelbrot(width, height, max_iter, min_x, max_x, min_y,
                              max_y)

    pixels = np.zeros((height, width, 3), dtype=np.uint8)
    for x in range(width):
        for y in range(height):
            color = colors[image[y, x] % len(colors)]
            pixels[y, x] = color

    img = Image.fromarray(pixels)
    img_tk = ImageTk.PhotoImage(img)
    canvas.create_image(0, 0, anchor=tk.NW, image=img_tk)
    canvas.image = img_tk  # Keep a reference to avoid garbage collection


class MandelbrotApp:
    def __init__(self, root):
        self.init_done = False
        self.root = root
        self.root.title("Mandelbrot Set")

        self.width, self.height = 512, 512
        self.canvas = Canvas(root, width=self.width,
                             height=self.height, bg='white')
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.min_x, self.max_x = -2.0, 1.0
        self.min_y, self.max_y = -1.5, 1.5
        self.max_iter = MAX_ITER

        self.zoom_factor = 1.5

        self.create_controls()

        self.canvas.bind("<Button-1>", self.zoom_in)
        self.canvas.bind("<Button-3>", self.zoom_out)
        self.root.bind("<Configure>", self.on_resize)

        self.init_done = True
        self.draw_mandelbrot()

    def create_controls(self):
        control_frame = tk.Frame(self.root)
        control_frame.pack(side=tk.TOP, fill=tk.X)

        Label(control_frame, text="Center (Real):").pack(side=tk.LEFT)
        self.real_entry = Entry(control_frame, width=10)
        self.real_entry.pack(side=tk.LEFT)
        self.real_entry.insert(0, "0.0")

        Label(control_frame, text="Center (Imag):").pack(side=tk.LEFT)
        self.imag_entry = Entry(control_frame, width=10)
        self.imag_entry.pack(side=tk.LEFT)
        self.imag_entry.insert(0, "0.0")

        Button(control_frame, text="Set Center", command=self.set_center).pack(
            side=tk.LEFT)
        Button(control_frame, text="Zoom in", command=self.zoom_in).pack(
            side=tk.RIGHT)
        Button(control_frame, text="Zoom out", command=self.zoom_out).pack(
            side=tk.RIGHT)

    def draw_mandelbrot(self):
        starttime = timeit.default_timer()
        draw_mandelbrot(self.canvas, self.width, self.height, self.max_iter,
                        self.min_x, self.max_x, self.min_y, self.max_y)
        print("The time difference is :", timeit.default_timer() - starttime)

    def zoom_in(self, event):
        self.zoom(event, self.zoom_factor)

    def zoom_out(self, event):
        self.zoom(event, 1 / self.zoom_factor)

    def zoom(self, event, zoom_factor):
        mouse_x, mouse_y = event.x, event.y
        mouse_re = self.min_x + (mouse_x / self.width) * (self.max_x - self.min_x)
        mouse_im = self.min_y + (mouse_y / self.height) * (self.max_y - self.min_y)

        new_width = (self.max_x - self.min_x) / zoom_factor
        new_height = (self.max_y - self.min_y) / zoom_factor

        self.min_x = mouse_re - new_width / 2
        self.max_x = mouse_re + new_width / 2
        self.min_y = mouse_im - new_height / 2
        self.max_y = mouse_im + new_height / 2

        self.draw_mandelbrot()

    def set_center(self):
        try:
            center_re = float(self.real_entry.get())
            center_im = float(self.imag_entry.get())
            center = complex(center_re, center_im)

            width = self.max_x - self.min_x
            height = self.max_y - self.min_y

            self.min_x = center.real - width / 2
            self.max_x = center.real + width / 2
            self.min_y = center.imag - height / 2
            self.max_y = center.imag + height / 2

            self.draw_mandelbrot()
        except ValueError:
            print("Invalid input for center coordinates")

    def on_resize(self, event):
        if self.width == event.width and self.height == event.height:
            return
        print(event)
        self.width = event.width
        self.height = event.height
        if self.init_done:
            self.draw_mandelbrot()


if __name__ == "__main__":
    root = tk.Tk()
    app = MandelbrotApp(root)
    root.mainloop()
