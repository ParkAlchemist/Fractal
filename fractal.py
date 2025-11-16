import math
import time
import numpy as np
from numba import cuda, njit, prange
import pyopencl as cl

from utils import clear_cache_lock


# --------------------------------------------------------
# ---------------------- OpenCL KERNEL -------------------
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
            r_accum = 0.0
            g_accum = 0.0
            b_accum = 0.0

            for sx in range(samples):
                for sy in range(samples):
                    offset_x = (sx + 0.5) / samples
                    offset_y = (sy + 0.5) / samples

                    real = min_x + (x + offset_x) * pixel_size_x
                    imag = min_y + (y + offset_y) * pixel_size_y
                    zr = real
                    zi = imag

                    i = 0
                    for i in range(max_iter):
                        zr2 = zr * zr
                        zi2 = zi * zi
                        if zr2 + zi2 >= 4.0:
                            break
                        temp = zr2 - zi2 + real
                        zi = 2.0 * zr * zi + imag
                        zr = temp

                    mag2 = zr * zr + zi * zi
                    if mag2 < eps:
                        mag2 = eps

                    # Smooth coloring
                    log_zn = 0.5 * math.log(mag2)
                    nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
                    smooth_iter = (i + 1) - nu if i < max_iter else i

                    norm = smooth_iter / float(max_iter)
                    norm = max(0.0, min(1.0, norm))

                    idx_f = norm * 255.0
                    idx = int(idx_f)
                    idx_next = min(idx + 1, 255)
                    t = idx_f - idx

                    base_idx0 = idx * 3
                    base_idx1 = idx_next * 3

                    r = (1.0 - t) * palette[base_idx0 + 0] + t * palette[base_idx1 + 0]
                    g = (1.0 - t) * palette[base_idx0 + 1] + t * palette[base_idx1 + 1]
                    b = (1.0 - t) * palette[base_idx0 + 2] + t * palette[base_idx1 + 2]

                    r_accum += r
                    g_accum += g
                    b_accum += b

            total_samples = samples * samples
            image[y, x, 0] = int(r_accum / total_samples)
            image[y, x, 1] = int(g_accum / total_samples)
            image[y, x, 2] = int(b_accum / total_samples)

# --------------------------------------------------------
# ---------------------- CPU KERNEL ---------------------
# --------------------------------------------------------
@njit(parallel=True, fastmath=True)
def mandelbrot_kernel_cpu(real_grid, imag_grid, width, height, max_iter, samples, palette):
    img = np.zeros((height, width, 3), dtype=np.uint8)
    eps = 1e-20

    for y in prange(height):
        for x in range(width):
            r_accum = g_accum = b_accum = 0.0
            for sx in range(samples):
                for sy in range(samples):
                    offset_x = (sx + 0.5) / samples
                    offset_y = (sy + 0.5) / samples
                    zr = real_grid[y, x] + offset_x * (
                                real_grid[0, 1] - real_grid[0, 0])
                    zi = imag_grid[y, x] + offset_y * (
                                imag_grid[1, 0] - imag_grid[0, 0])

                    i = 0
                    for i in range(max_iter):
                        zr2 = zr * zr
                        zi2 = zi * zi
                        if zr2 + zi2 >= 4.0:
                            break
                        temp = zr2 - zi2 + real_grid[y, x]
                        zi = 2.0 * zr * zi + imag_grid[y, x]
                        zr = temp

                    mag2 = zr * zr + zi * zi
                    if mag2 < eps:
                        mag2 = eps

                    log_zn = 0.5 * math.log(mag2)
                    nu = math.log(log_zn / math.log(2.0)) / math.log(2.0)
                    smooth_iter = (i + 1) - nu if i < max_iter else i

                    norm = smooth_iter / max_iter
                    norm = max(0.0, min(1.0, norm))

                    idx_f = norm * 255.0
                    idx = int(idx_f)
                    idx_next = min(idx + 1, 255)
                    t = idx_f - idx

                    base_idx0 = idx * 3
                    base_idx1 = idx_next * 3

                    r = (1.0 - t) * palette[base_idx0 + 0] + t * palette[
                        base_idx1 + 0]
                    g = (1.0 - t) * palette[base_idx0 + 1] + t * palette[
                        base_idx1 + 1]
                    b = (1.0 - t) * palette[base_idx0 + 2] + t * palette[
                        base_idx1 + 2]

                    r_accum += r
                    g_accum += g
                    b_accum += b

            total_samples = samples * samples
            img[y, x, 0] = int(r_accum / total_samples)
            img[y, x, 1] = int(g_accum / total_samples)
            img[y, x, 2] = int(b_accum / total_samples)

    return img


# --------------------------------------------------------
# ------------------------ CLASS -------------------------
# --------------------------------------------------------
class Mandelbrot:
    def __init__(self, palette, kernel="auto", img_width=800, img_height=600,
                 max_iter=1000, precision="float32", threads_per_block=(16, 16),
                 enable_timing=False):

        self.palette = np.array(palette, dtype=np.uint8).reshape(-1, 3).flatten()
        self.width = img_width
        self.height = img_height
        self.max_iter = max_iter
        self.precision = np.float32 if precision == "float32" else np.float64
        self.enable_timing = enable_timing

        # Auto fall-back logic
        if kernel == "auto":
            if cuda.is_available():
                self.kernel = "cuda"
            else:
                try:
                    platforms = cl.get_platforms()
                    if platforms:
                        self.kernel = "opencl"
                    else:
                        self.kernel = "cpu"
                except RuntimeError:
                    self.kernel = "cpu"
        else:
            self.kernel = kernel.lower()

        # Init backend
        if self.kernel == "opencl":
            self._init_opencl()
            self.render = self._render_opencl
            print("Using OpenCL backend.")
        elif self.kernel == "cuda":
            self._init_cuda(threads_per_block)
            self.render = self._render_cuda
            print("Using CUDA backend.")
        elif self.kernel == "cpu":
            self.render = self._render_cpu
        else:
            raise ValueError("Kernel must be 'opencl', 'cuda' or 'cpu'.")

    # ---------------- INIT OPENCL ----------------
    def _init_opencl(self):

        clear_cache_lock()

        try:
            platforms = cl.get_platforms()
            if not platforms:
                raise RuntimeError("No OpenCL platforms found.")

            # Prefer GPU device if available
            devices = []
            for platform in platforms:
                gpu_devices = platform.get_devices(
                    device_type=cl.device_type.GPU)
                if gpu_devices:
                    devices.extend(gpu_devices)
                else:
                    devices.extend(
                        platform.get_devices(device_type=cl.device_type.CPU))

            if not devices:
                raise RuntimeError("No OpenCL devices found.")

            # Pick the first GPU or CPU device
            self.device = devices[0]
            self.context = cl.Context([self.device])
            self.queue = cl.CommandQueue(self.context, self.device)

            print(f"Using OpenCL device: {self.device.name}")

        except Exception as e:
            raise RuntimeError(f"OpenCL initialization failed: {e}")

        # Allocate buffers
        self.palette_buf = cl.Buffer(self.context,
                                     cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                     hostbuf=self.palette)

        self.image_np = np.zeros(self.width * self.height * 3, dtype=np.uint8)
        self.image_buf = cl.Buffer(self.context,
                                   cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                   self.image_np.nbytes,
                                   hostbuf=self.image_np)

        # Build Kernel
        self.program = cl.Program(self.context, mandelbrot_kernel_cl).build()

    # ---------------- INIT CUDA ----------------
    def _init_cuda(self, threads_per_block):
        if not cuda.is_available():
            raise RuntimeError("CUDA not available on this system.")

        self.image_gpu = cuda.device_array((self.height, self.width, 3),
                                           dtype=np.uint8)
        self.palette_device = cuda.to_device(self.palette)
        self.threads_per_block = threads_per_block
        self.blocks_per_grid = (
            (self.width + threads_per_block[0] - 1) // threads_per_block[0],
            (self.height + threads_per_block[1] - 1) // threads_per_block[1]
        )

    # ---------------- CUDA RENDER ----------------
    def _render_cuda(self, min_x, max_x, min_y, max_y, samples=2):
        start = time.time() if self.enable_timing else None

        mandelbrot_kernel_cuda[self.blocks_per_grid, self.threads_per_block](
            self.precision(min_x), self.precision(max_x),
            self.precision(min_y), self.precision(max_y),
            self.image_gpu, np.int32(self.max_iter),
            self.palette_device, np.int32(samples)
        )
        cuda.synchronize()
        result = self.image_gpu.copy_to_host()

        if self.enable_timing:
            print(f"CUDA render time: {time.time() - start:.3f}s")

        return result

    # ---------------- OPENCL RENDER ----------------
    def _render_opencl(self, min_x, max_x, min_y, max_y, samples=2):
        start = time.time() if self.enable_timing else None

        self.program.mandelbrot_kernel(
            self.queue, (self.width, self.height), None,
            self.precision(min_x), self.precision(max_x),
            self.precision(min_y), self.precision(max_y),
            self.image_buf,
            np.int32(self.width), np.int32(self.height),
            np.int32(self.max_iter),
            self.palette_buf,
            np.int32(samples)
        )

        cl.enqueue_copy(self.queue, self.image_np, self.image_buf)
        self.queue.finish()

        if self.enable_timing:
            print(f"OpenCL render time: {time.time() - start:.3f}s")

        return self.image_np.reshape((self.height, self.width, 3))

    # ---------------- CPU RENDER -------------------
    def _render_cpu(self, min_x, max_x, min_y, max_y, samples=2):
        start = time.time() if self.enable_timing else None

        real = np.linspace(min_x, max_x, self.width, dtype=self.precision)
        imag = np.linspace(min_y, max_y, self.height, dtype=self.precision)
        real_grid, imag_grid = np.meshgrid(real, imag)
        img = mandelbrot_kernel_cpu(real_grid, imag_grid,
                                    self.width, self.height,
                                    self.max_iter, samples, self.palette)
        if self.enable_timing:
            print(f"CPU render time: {time.time() - start:.3f}s")
        return img

    # ---------------- PALETTE UPDATE ----------------
    def change_palette(self, palette):
        self.palette = np.array(palette, dtype=np.uint8).reshape(-1,
                                                                 3).flatten()
        if self.kernel == "opencl":
            self.palette_buf = cl.Buffer(self.context,
                                         cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                         hostbuf=self.palette)
        elif self.kernel == "cuda":
            self.palette_device = cuda.to_device(self.palette)

    # ---------------- CLEANUP ----------------
    def close(self):
        if self.kernel == "opencl":
            self.image_buf.release()
            self.palette_buf.release()
        # CUDA cleanup handled by Numba automatically

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
