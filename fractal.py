import time
import numpy as np
from numba import cuda
import pyopencl as cl

from utils import clear_cache_lock, available_backends
from kernel import (mandelbrot_kernel_cl_float,
                    mandelbrot_kernel_cl_double,
                    mandelbrot_kernel_cuda_f32,
                    mandelbrot_kernel_cuda_f64,
                    mandelbrot_kernel_cpu,
                    mandelbrot_kernel_cpu_mp,
                    Kernel)

class Precisions:
    single = np.float32
    double = np.float64
    arbitrary = 0

# --------------------------------------------------------
# ------------------------ CLASS -------------------------
# --------------------------------------------------------
class Mandelbrot:
    def __init__(self, kernel=Kernel.AUTO, img_width=800, img_height=600,
                 max_iter=1000, precision=Precisions.single, enable_timing=False, samples=2, zoom_factor=250):

        self.width = img_width
        self.height = img_height
        self.max_iter = max_iter
        self.precision = precision
        self.enable_timing = enable_timing
        self.kernel = kernel
        self.samples = samples
        self.zoom_factor = zoom_factor

        # Kernel warm up params
        self.wu_min_x = -2.0
        self.wu_max_x = 1.0
        self.wu_min_y = -1.5
        self.wu_max_y = 1.5
        self.wu_max_iter = 100
        self.wu_samples = 1
        self.wu_width = 256
        self.wu_height = 256
        self.cpu_warmed_up = False
        self.cpu_mp_warmed_up = False
        self.cuda_f32_warmed_up = False
        self.cuda_f64_warmed_up = False

        # Auto mode: select available backend opencl -> cuda -> cpu
        if self.kernel == Kernel.AUTO:
            # Detect available backends
            self.available_backends = available_backends()

            if Kernel.OPENCL.name in self.available_backends:
                self.kernel = Kernel.OPENCL
            elif Kernel.CUDA.name in self.available_backends:
                self.kernel = Kernel.CUDA
            else:
                self.kernel = Kernel.CPU

        self._init_kernel()

    # ----------------------------------------------
    # --------------- KERNEL INITS -----------------
    # ----------------------------------------------
    def _init_kernel(self):
        # Init backend
        if self.kernel == Kernel.OPENCL and self.precision != Precisions.arbitrary:
            self._init_opencl()
            self.render = self._render_opencl
            print("Using OpenCL backend.")
        elif self.kernel == Kernel.CUDA and self.precision != Precisions.arbitrary:
            self._init_cuda()
            self.render = self._render_cuda
            print("Using CUDA backend.")
        elif self.kernel == Kernel.CPU or self.precision == Precisions.arbitrary:
            self._init_cpu()
            self.render = self._render_cpu
            print("Using CPU backend.")
        else:
            raise ValueError(
                f"Kernel must be {Kernel.OPENCL.name}, {Kernel.CUDA.name} or {Kernel.CPU.name}.")

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
                gpu_devices = platform.get_devices(device_type=cl.device_type.GPU)
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
        self.iter_np = np.zeros(self.width * self.height, dtype=self.precision)
        self.iter_buf = cl.Buffer(self.context,
                                   cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                   self.iter_np.nbytes,
                                   hostbuf=self.iter_np)

        # Build program
        if self.precision == Precisions.single:
            self.program = cl.Program(self.context, mandelbrot_kernel_cl_float).build()
        elif self.precision == Precisions.double:
            self.program = cl.Program(self.context, mandelbrot_kernel_cl_double).build()

        # Store kernel
        self.kernel_prog = cl.Kernel(self.program, "mandelbrot_kernel")

    # ---------------- INIT CUDA ----------------
    def _init_cuda(self):
        if not cuda.is_available():
            raise RuntimeError("CUDA not available on this system.")

        self.iter_gpu = cuda.device_array((self.height, self.width),
                                           dtype=self.precision)
        self.threads_per_block = (16, 16)

        self.blocks_per_grid = (
            (self.width + self.threads_per_block[0] - 1) // self.threads_per_block[0],
            (self.height + self.threads_per_block[1] - 1) // self.threads_per_block[1]
        )

        print(f"CUDA block size: {self.threads_per_block}, grid size: {self.blocks_per_grid}")

        # Warm up
        if self.cuda_f32_warmed_up and self.cuda_f64_warmed_up: return
        print("Warming up CUDA kernel...")
        if self.precision == Precisions.single:
            self.cuda_kernel = mandelbrot_kernel_cuda_f32
            self.cuda_f32_warmed_up = True
        elif self.precision == Precisions.double:
            self.cuda_kernel = mandelbrot_kernel_cuda_f64
            self.cuda_f64_warmed_up = True

        self.cuda_kernel[self.blocks_per_grid, self.threads_per_block](
            self.precision(self.wu_min_x), self.precision(self.wu_max_x),
            self.precision(self.wu_min_y), self.precision(self.wu_max_y),
            self.iter_gpu, np.int32(self.wu_max_iter),
            np.int32(self.wu_samples)
        )
        print("Warmed up!")

    # ---------------- Init CPU --------------------
    def _init_cpu(self):
        if self.cpu_warmed_up and self.cpu_mp_warmed_up: return

        print("Warming up CPU kernel...")
        if self.precision != Precisions.arbitrary:
            real = np.linspace(self.wu_min_x, self.wu_max_x,
                               self.wu_width, dtype=self.precision)
            imag = np.linspace(self.wu_min_y, self.wu_max_y,
                               self.wu_height, dtype=self.precision)
            real_grid, imag_grid = np.meshgrid(real, imag)
            mandelbrot_kernel_cpu(real_grid, imag_grid,
                                  self.wu_width, self.wu_height,
                                  self.wu_max_iter, self.wu_samples, self.precision)
            self.cpu_warmed_up = True
        else:
            width, height = 10, 10
            min_x, max_x = -0.743643887037151, -0.743643887037151 + 1e-12
            min_y, max_y = 0.13182590420533, 0.13182590420533 + 1e-12
            zoom_factor = 1e12

            mandelbrot_kernel_cpu_mp(min_x, max_x, min_y, max_y,
                                      width,
                                      height, max_iter=1000, samples=1,
                                      zoom_factor=zoom_factor)
            self.cpu_mp_warmed_up = True
        print("Warmed up!")

    # ---------------- OPENCL RENDER ----------------
    def _render_opencl(self, min_x, max_x, min_y, max_y):
        start = time.time() if self.enable_timing else None

        self.kernel_prog.set_arg(0, self.precision(min_x))
        self.kernel_prog.set_arg(1, self.precision(max_x))
        self.kernel_prog.set_arg(2, self.precision(min_y))
        self.kernel_prog.set_arg(3, self.precision(max_y))
        self.kernel_prog.set_arg(4, self.iter_buf)
        self.kernel_prog.set_arg(5, np.int32(self.width))
        self.kernel_prog.set_arg(6, np.int32(self.height))
        self.kernel_prog.set_arg(7, np.int32(self.max_iter))
        self.kernel_prog.set_arg(8, np.int32(self.samples))

        global_size = (self.width, self.height)
        local_size = None  # Let OpenCL decide
        cl.enqueue_nd_range_kernel(self.queue, self.kernel_prog,
                                   global_size, local_size)

        cl.enqueue_copy(self.queue, self.iter_np, self.iter_buf)
        self.queue.finish()

        if self.enable_timing:
            print(f"OpenCL render time: {time.time() - start:.3f}s")

        return self.iter_np.reshape((self.height, self.width))

    # ---------------- CUDA RENDER ----------------
    def _render_cuda(self, min_x, max_x, min_y, max_y):
        start = time.time() if self.enable_timing else None

        self.cuda_kernel[
            self.blocks_per_grid, self.threads_per_block](
            self.precision(min_x), self.precision(max_x),
            self.precision(min_y), self.precision(max_y),
            self.iter_gpu, np.int32(self.max_iter),
            np.int32(self.samples)
        )
        cuda.synchronize()
        result = self.iter_gpu.copy_to_host()

        if self.enable_timing:
            print(f"CUDA render time: {time.time() - start:.3f}s")

        return result

    # ---------------- CPU RENDER -------------------
    def _render_cpu(self, min_x, max_x, min_y, max_y):
        start = time.time() if self.enable_timing else None

        img = None
        if self.precision != Precisions.arbitrary:
            real = np.linspace(min_x, max_x, self.width, dtype=self.precision)
            imag = np.linspace(min_y, max_y, self.height, dtype=self.precision)
            real_grid, imag_grid = np.meshgrid(real, imag)
            img = mandelbrot_kernel_cpu(real_grid, imag_grid,
                                        self.width, self.height,
                                        self.max_iter, self.samples, self.precision)
        else:
            img = mandelbrot_kernel_cpu_mp(min_x, max_x, min_y, max_y,
                                           self.width, self.height,
                                           self.max_iter, self.samples,
                                           self.zoom_factor)
        if self.enable_timing:
            print(f"CPU render time: {time.time() - start:.3f}s")
        return img

    # ------------ IMAGE SIZE UPDATE ---------------
    def change_image_size(self, width, height):
        self.width = width
        self.height = height

        if self.kernel == Kernel.OPENCL and self.precision != Precisions.arbitrary:
            self.iter_np = np.zeros(self.width * self.height,
                                     dtype=self.precision)
            if hasattr(self, "iter_buf"):
                self.iter_buf.release()
            self.iter_buf = cl.Buffer(self.context,
                                       cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                       self.iter_np.nbytes,
                                       hostbuf=self.iter_np)
        elif self.kernel == Kernel.CUDA and self.precision != Precisions.arbitrary:
            self.iter_gpu = cuda.device_array((self.height, self.width),
                                               dtype=self.precision)

    # ---------------- Kernel Change ----------------
    def change_kernel(self, new_kernel):
        if new_kernel == self.kernel:
            return

        # Cleanup
        self.close()

        # Init new kernel
        self.kernel = new_kernel
        self._init_kernel()
        print(f"Kernel switched to {self.kernel.name} successfully.")


    # ---------------- Precision change --------------------
    def change_precision(self, new_precison):
        if new_precison == self.precision:
            return

        # Cleanup
        self.close()

        # Init new kernel
        self.precision = new_precison
        self._init_kernel()
        print(f"Precision changed to {self.precision.name} successfully.")

    # ---------------- CLEANUP ----------------
    def close(self):
        # OpenCL cleanup
        if hasattr(self, "iter_buf") and self.iter_buf is not None:
            self.iter_buf.release()
            self.iter_buf = None
        if hasattr(self, "queue") and self.queue is not None:
            try:
                self.queue.finish()
            except Exception:
                pass
            self.queue = None
        if hasattr(self, "context"):
            self.context = None
        if hasattr(self, "program"):
            self.program = None
        if hasattr(self, "kernel_prog"):
            self.kernel_prog = None

        # CUDA cleanup
        if hasattr(self, "iter_gpu") and self.iter_gpu is not None:
            del self.iter_gpu
            self.iter_gpu = None

        # General cleanup
        clear_cache_lock()


    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()


if __name__ == "__main__":
    from mpmath import mp
    # Test with a small image and deep zoom

    width, height = 64, 64
    min_x, max_x = -0.743643887037151, -0.743643887037151 + 1e-10
    min_y, max_y = 0.13182590420533, 0.13182590420533 + 1e-10
    zoom_factor = 1e10

    result = mandelbrot_kernel_cpu_mp(min_x, max_x, min_y, max_y,
                                               width, height, max_iter=500,
                                               samples=1,
                                               zoom_factor=zoom_factor)
    print("Parallel arbitrary precision Mandelbrot computed for 64x64 region at zoom 1e10.")
    print("Precision digits:", mp.dps)
    print("Sample output:", result[:2, :2])
