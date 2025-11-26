import time
import numpy as np
from numba import cuda
import pyopencl as cl

from utils import clear_cache_lock, available_backends
from kernel import (mandelbrot_kernel_cl,
                    mandelbrot_kernel_cuda,
                    mandelbrot_kernel_cpu,
                    Kernel)


# --------------------------------------------------------
# ------------------------ CLASS -------------------------
# --------------------------------------------------------
class Mandelbrot:
    def __init__(self, kernel=Kernel.AUTO, img_width=800, img_height=600,
                 max_iter=1000, precision="float32", enable_timing=False, samples=2):

        self.width = img_width
        self.height = img_height
        self.max_iter = max_iter
        self.precision = np.float32 if precision == "float32" else np.float64
        self.enable_timing = enable_timing
        self.kernel = kernel
        self.samples = samples

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
        self.cuda_warmed_up = False

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
        if self.kernel == Kernel.OPENCL:
            self._init_opencl()
            self.render = self._render_opencl
            print("Using OpenCL backend.")
        elif self.kernel == Kernel.CUDA:
            self._init_cuda()
            self.render = self._render_cuda
            print("Using CUDA backend.")
        elif self.kernel == Kernel.CPU:
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
        self.iter_np = np.zeros(self.width * self.height, dtype=np.float32)
        self.iter_buf = cl.Buffer(self.context,
                                   cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                   self.iter_np.nbytes,
                                   hostbuf=self.iter_np)

        # Build program
        self.program = cl.Program(self.context, mandelbrot_kernel_cl).build()

        # Store kernel
        self.kernel_prog = cl.Kernel(self.program, "mandelbrot_kernel")

    # ---------------- INIT CUDA ----------------
    def _init_cuda(self):
        if not cuda.is_available():
            raise RuntimeError("CUDA not available on this system.")

        self.iter_gpu = cuda.device_array((self.height, self.width),
                                           dtype=np.float32)
        self.threads_per_block = (16, 16)

        self.blocks_per_grid = (
            (self.width + self.threads_per_block[0] - 1) // self.threads_per_block[0],
            (self.height + self.threads_per_block[1] - 1) // self.threads_per_block[1]
        )

        print(f"CUDA block size: {self.threads_per_block}, grid size: {self.blocks_per_grid}")

        # Warm up
        if self.cuda_warmed_up: return
        print("Warming up CUDA kernel...")
        mandelbrot_kernel_cuda[self.blocks_per_grid, self.threads_per_block](
            self.precision(self.wu_min_x), self.precision(self.wu_max_x),
            self.precision(self.wu_min_y), self.precision(self.wu_max_y),
            self.iter_gpu, np.int32(self.wu_max_iter),
            np.int32(self.wu_samples)
        )
        print("Warmed up!")
        self.cuda_warmed_up = True

    # ---------------- Init CPU --------------------
    def _init_cpu(self):
        if self.cpu_warmed_up: return
        print("Warming up CPU kernel...")
        real = np.linspace(self.wu_min_x, self.wu_max_x,
                           self.wu_width, dtype=self.precision)
        imag = np.linspace(self.wu_min_y, self.wu_max_y,
                           self.wu_height, dtype=self.precision)
        real_grid, imag_grid = np.meshgrid(real, imag)
        mandelbrot_kernel_cpu(real_grid, imag_grid,
                              self.wu_width, self.wu_height,
                              self.wu_max_iter, self.wu_samples)
        print("Warmed up!")
        self.cpu_warmed_up = True

    # ---------------- CUDA RENDER ----------------
    def _render_cuda(self, min_x, max_x, min_y, max_y):
        start = time.time() if self.enable_timing else None

        mandelbrot_kernel_cuda[self.blocks_per_grid, self.threads_per_block](
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

    # ---------------- CPU RENDER -------------------
    def _render_cpu(self, min_x, max_x, min_y, max_y):
        start = time.time() if self.enable_timing else None

        real = np.linspace(min_x, max_x, self.width, dtype=self.precision)
        imag = np.linspace(min_y, max_y, self.height, dtype=self.precision)
        real_grid, imag_grid = np.meshgrid(real, imag)
        img = mandelbrot_kernel_cpu(real_grid, imag_grid,
                                    self.width, self.height,
                                    self.max_iter, self.samples)
        if self.enable_timing:
            print(f"CPU render time: {time.time() - start:.3f}s")
        return img

    # ------------ IMAGE SIZE UPDATE ---------------
    def change_image_size(self, width, height):
        self.width = width
        self.height = height

        if self.kernel == Kernel.OPENCL:
            self.iter_np = np.zeros(self.width * self.height,
                                     dtype=np.float32)
            if hasattr(self, "iter_buf"):
                self.iter_buf.release()
            self.iter_buf = cl.Buffer(self.context,
                                       cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                       self.iter_np.nbytes,
                                       hostbuf=self.iter_np)
        elif self.kernel == Kernel.CUDA:
            self.iter_gpu = cuda.device_array((self.height, self.width),
                                               dtype=np.float32)

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
