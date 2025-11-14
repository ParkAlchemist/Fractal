import math
import sys
import numpy as np
from numba import cuda, float32, int32
import pyopencl as cl

from utils import clear_cache_lock


# --------------------------------------------------------
# -------- OpenCL KERNEL ---------------------------------
# --------------------------------------------------------
mandelbrot_kernel_cl = """
__kernel void mandelbrot_kernel(
    const float min_x, const float max_x,
    const float min_y, const float max_y,
    __global uchar *image,
    const int width, const int height,
    const int max_iter,
    __global const uchar *palette,
    const int samples)
{
    int x = get_global_id(0);
    int y = get_global_id(1);
    if (x >= width || y >= height) return;

    float pixel_size_x = (max_x - min_x) / (float)width;
    float pixel_size_y = (max_y - min_y) / (float)height;

    float r_accum = 0.0f, g_accum = 0.0f, b_accum = 0.0f;

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

        float norm = smooth_iter / (float)max_iter;
        norm = fmax(0.0f, fmin(1.0f, norm));

        float idx_f = norm * 255.0f;
        int idx = (int)idx_f;
        int idx_next = min(idx + 1, 255);
        float t = idx_f - (float)idx;

        int base_idx0 = idx * 3;
        int base_idx1 = idx_next * 3;

        float r = (1.0f - t) * palette[base_idx0 + 0] + t * palette[base_idx1 + 0];
        float g = (1.0f - t) * palette[base_idx0 + 1] + t * palette[base_idx1 + 1];
        float b = (1.0f - t) * palette[base_idx0 + 2] + t * palette[base_idx1 + 2];

        r_accum += r;
        g_accum += g;
        b_accum += b;
    }

    float total_samples = (float)(samples * samples);
    int base_idx = (y * width + x) * 3;
    image[base_idx + 0] = (uchar)(r_accum / total_samples);
    image[base_idx + 1] = (uchar)(g_accum / total_samples);
    image[base_idx + 2] = (uchar)(b_accum / total_samples);
}
"""


# --------------------------------------------------------
# ---------------------- CUDA KERNEL ---------------------
# --------------------------------------------------------
@cuda.jit
def mandelbrot_kernel_cuda(min_x, max_x, min_y, max_y, image, max_iter, palette, samples):
    height, width, _ = image.shape
    pixel_size_x = (max_x - min_x) / float(width)
    pixel_size_y = (max_y - min_y) / float(height)

    startX, startY = cuda.grid(2)
    gridX, gridY = cuda.gridsize(2)

    eps = 1e-20

    for x in range(startX, width, gridX):
        for y in range(startY, height, gridY):
            r_accum = 0
            g_accum = 0
            b_accum = 0

            for sx in range(samples):
                for sy in range(samples):
                    offset_x = (sx + 0.5) / samples
                    offset_y = (sy + 0.5) / samples

                    real = min_x + (x + offset_x) * pixel_size_x
                    imag = min_y + (y + offset_y) * pixel_size_y
                    zr = real
                    zi = imag

                    escaped = False
                    i = 0

                    for i in range(max_iter):
                        zr2 = zr * zr
                        zi2 = zi * zi

                        if zr2 + zi2 >= 4.0:
                            escaped = True
                            break

                        temp = zr2 - zi2 + real
                        zi = 2.0 * zr * zi + imag
                        zr = temp

                    mag = zr * zr + zi * zi
                    if mag < eps:
                        mag = eps

                    if not escaped:
                        modulus = math.sqrt(mag)
                        dist = 0.5 * math.log(modulus) * modulus
                        shade = int(255 * math.exp(-dist))
                        shade = max(0, min(255, shade))
                        r, g, b = shade, shade, shade
                    else:
                        log_zn = 0.5 * math.log(mag)
                        log_zn = max(log_zn, eps)
                        nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
                        smooth = (i + 1) - nu
                        norm = smooth / float(max_iter)
                        norm = max(0.0, min(1.0, norm))

                        idx = int(norm * 255.0)
                        idx = max(0, min(255, idx))

                        r = int(palette[idx, 0])
                        g = int(palette[idx, 1])
                        b = int(palette[idx, 2])

                    r_accum += r
                    g_accum += g
                    b_accum += b

            total = samples * samples
            image[y, x, 0] = r_accum // total
            image[y, x, 1] = g_accum // total
            image[y, x, 2] = b_accum // total



# --------------------------------------------------------
# ------------------------ CLASS -------------------------
# --------------------------------------------------------
class Mandelbrot:
    def __init__(self, palette, kernel, img_width, img_height, max_iter):

        self.palette = palette
        self.kernel = kernel.lower()
        self.width = img_width
        self.height = img_height
        self.max_iter = max_iter

        if self.kernel == "opencl":

            clear_cache_lock()  # 🔧 Fix OpenCL cache warning once

            self.context = cl.create_some_context()
            self.queue = cl.CommandQueue(self.context)

            palette_np = np.array(palette, dtype=np.uint8).reshape(-1, 3).flatten()
            self.palette_buf = cl.Buffer(self.context,
                                         cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                         hostbuf=palette_np)

            self.image_np = np.zeros(self.width * self.height * 3, dtype=np.uint8)
            self.image_buf = cl.Buffer(self.context, cl.mem_flags.WRITE_ONLY, self.image_np.nbytes)

            self.program = cl.Program(self.context, mandelbrot_kernel_cl).build()

            self.render = self._render_opencl

        elif self.kernel == "cuda":

            if not cuda.is_available():
                print("CUDA not available")
                sys.exit(1)

            self.image_gpu = cuda.device_array((self.height, self.width, 3), dtype=np.uint8)

            palette_np = np.array(self.palette, dtype=np.uint8).reshape(-1, 3)
            self.palette_device = cuda.to_device(palette_np)

            self.threads_per_block = (16, 16)
            self.blocks_per_grid = (
                (self.width + 15) // 16,
                (self.height + 15) // 16
            )

            self.render = self._render_cuda

        else:
            raise ValueError("Kernel must be 'opencl' or 'cuda'.")

    # ---------------- CUDA ----------------
    def _render_cuda(self, min_x, max_x, min_y, max_y, samples=2):
        mandelbrot_kernel_cuda[self.blocks_per_grid, self.threads_per_block](
            float32(min_x), float32(max_x),
            float32(min_y), float32(max_y),
            self.image_gpu, int32(self.max_iter),
            self.palette_device, int32(samples)
        )
        cuda.synchronize()
        return self.image_gpu.copy_to_host()


    # ---------------- OPENCL ----------------
    def _render_opencl(self, min_x, max_x, min_y, max_y, samples=2):
        self.program.mandelbrot_kernel(
            self.queue, (self.width, self.height), None,
            np.float32(min_x), np.float32(max_x),
            np.float32(min_y), np.float32(max_y),
            self.image_buf,
            np.int32(self.width), np.int32(self.height),
            np.int32(self.max_iter),
            self.palette_buf,
            np.int32(samples)
        )

        cl.enqueue_copy(self.queue, self.image_np, self.image_buf)
        self.queue.finish()

        return self.image_np.reshape((self.height, self.width, 3))


    def change_palette(self, palette):
        """ Update gradient on GPU at runtime """
        self.palette = palette

        if self.kernel == "opencl":
            palette_np = np.array(palette, dtype=np.uint8).reshape(-1, 3).flatten()
            self.palette_buf = cl.Buffer(self.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=palette_np)

        elif self.kernel == "cuda":
            self.palette_device = cuda.to_device(np.array(self.palette, dtype=np.uint8).reshape(-1, 3))
