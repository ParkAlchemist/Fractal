import time
import numpy as np
from numba import cuda
import pyopencl as cl

from utils import clear_cache_lock, available_backends, make_reference_orbit_hp
from kernel import (mandelbrot_kernel_cl_f32,
                    mandelbrot_kernel_cl_f64,
                    mandelbrot_kernel_cl_perturb,
                    mandelbrot_kernel_cuda_f32,
                    mandelbrot_kernel_cuda_f64,
                    mandelbrot_kernel_cuda_perturb,
                    mandelbrot_kernel_cpu_f32,
                    mandelbrot_kernel_cpu_f64,
                    mandelbrot_kernel_cpu_perturb,
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
                 max_iter=1000, precision=Precisions.single,
                 enable_timing=False, samples=2, zoom_factor=250,
                 use_perturb=False, order=2, w_fallback_thresh=1e-6, hp_dps=160):

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

        self.cpu_f32_warmed_up = False
        self.cpu_f64_warmed_up = False
        self.cpu_perturb_warmed_up = False
        self.cuda_f32_warmed_up = False
        self.cuda_f64_warmed_up = False
        self.cuda_perturb_warmed_up = False

        # Perturbation params
        self.use_perturb = use_perturb
        self.order = order
        self.w_fallback_thresh = w_fallback_thresh
        self.hp_dps = hp_dps

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
            if self.use_perturb:
                self.render = self._render_opencl_perturb
                print("Using OpenCL backend (perturbation).")
            else:
                self.render = self._render_opencl
                print("Using OpenCL backend.")
        elif self.kernel == Kernel.CUDA:
            self._init_cuda()
            if self.use_perturb:
                self.render = self._render_cuda_perturb
                print("Using CUDA backend (perturbation).")
            else:
                self.render = self._render_cuda
                print("Using CUDA backend.")
        elif self.kernel == Kernel.CPU:
            self._init_cpu()
            if self.use_perturb:
                self.render = self._render_cpu_perturb
                print("Using CPU backend (perturbation).")
            else:
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
        self.iter_np = np.zeros(self.width * self.height, dtype=np.float32 if self.precision == Precisions.single else np.float64)
        self.iter_buf = cl.Buffer(self.context,
                                   cl.mem_flags.WRITE_ONLY | cl.mem_flags.COPY_HOST_PTR,
                                   self.iter_np.nbytes,
                                   hostbuf=self.iter_np)

        # Build program
        if not self.use_perturb:
            src = mandelbrot_kernel_cl_f32 if self.precision == Precisions.single else mandelbrot_kernel_cl_f64
            self.program = cl.Program(self.context, src).build()
            self.kernel_prog = cl.Kernel(self.program, "mandelbrot_kernel")
            # Store kernel
            self.kernel_prog = cl.Kernel(self.program, "mandelbrot_kernel")
        else:
            self.program = cl.Program(self.context, mandelbrot_kernel_cl_perturb).build()
            self.kernel_prog = cl.Kernel(self.program,
                                         "mandelbrot_perturb_kernel")
            # Store kernel
            self.kernel_prog = cl.Kernel(self.program, "mandelbrot_perturb_kernel")



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
        self.cuda_perturb_kernel = mandelbrot_kernel_cuda_perturb
        self.cuda_perturb_warmed_up = True

        self.cuda_kernel[self.blocks_per_grid, self.threads_per_block](
            self.precision(self.wu_min_x), self.precision(self.wu_max_x),
            self.precision(self.wu_min_y), self.precision(self.wu_max_y),
            self.iter_gpu, np.int32(self.wu_max_iter),
            np.int32(self.wu_samples)
        )
        print("Warmed up!")

    # ---------------- Init CPU --------------------
    def _init_cpu(self):

        print("Warming up CPU kernel...")
        real = np.linspace(self.wu_min_x, self.wu_max_x, self.wu_width,
                           dtype=self.precision)
        imag = np.linspace(self.wu_min_y, self.wu_max_y, self.wu_height,
                           dtype=self.precision)
        real_grid, imag_grid = np.meshgrid(real, imag)
        if self.precision == Precisions.single:
            if self.cpu_f32_warmed_up: return
            mandelbrot_kernel_cpu_f32(real_grid, imag_grid,
                                      self.wu_width, self.wu_height,
                                      self.wu_max_iter, self.wu_samples)
            self.cpu_f32_warmed_up = True
            self.cpu_kernel = mandelbrot_kernel_cpu_f32
        elif self.precision == Precisions.double:
            if self.cpu_f64_warmed_up: return
            mandelbrot_kernel_cpu_f64(real_grid, imag_grid,
                                      self.wu_width, self.wu_height,
                                      self.wu_max_iter, self.wu_samples)
            self.cpu_f64_warmed_up = True
            self.cpu_kernel = mandelbrot_kernel_cpu_f64
        if self.cpu_perturb_warmed_up: return
        self.cpu_perturb_warmed_up = True
        self.cpu_perturb_kernel = mandelbrot_kernel_cpu_perturb
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

    def _render_opencl_perturb(self, min_x, max_x, min_y, max_y):
        start = time.time() if self.enable_timing else None

        # reference at viewport center
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        c_ref = complex(cx, cy)

        # high-precision orbit -> downcast to target precision
        zref = make_reference_orbit_hp(c_ref, self.max_iter,
                                       mp_dps=self.hp_dps)  # shape (N,2), float64

        zref_r = np.ascontiguousarray(zref[:, 0].astype(np.float64, copy=False))
        zref_i = np.ascontiguousarray(zref[:, 1].astype(np.float64, copy=False))
        cRef_r = np.float64(c_ref.real)
        cRef_i = np.float64(c_ref.imag)
        minx = np.float64(min_x)
        maxx = np.float64(max_x)
        miny = np.float64(min_y)
        maxy = np.float64(max_y)
        wth = np.float64(self.w_fallback_thresh)

        # Upload zref
        zref_r_buf = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=zref_r)
        zref_i_buf = cl.Buffer(self.context,
                               cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR,
                               hostbuf=zref_i)

        # Set kernel args (match kernel signature)
        # (cRef_r, cRef_i, zref_r, zref_i, refLen, min/max, out, W,H, max_iter, samples, order, wThresh)
        self.kernel_prog.set_arg(0, cRef_r)
        self.kernel_prog.set_arg(1, cRef_i)
        self.kernel_prog.set_arg(2, zref_r_buf)
        self.kernel_prog.set_arg(3, zref_i_buf)
        self.kernel_prog.set_arg(4, np.int32(zref.shape[0]))
        self.kernel_prog.set_arg(5, minx)
        self.kernel_prog.set_arg(6, maxx)
        self.kernel_prog.set_arg(7, miny)
        self.kernel_prog.set_arg(8, maxy)
        self.kernel_prog.set_arg(9, self.iter_buf)
        self.kernel_prog.set_arg(10, np.int32(self.width))
        self.kernel_prog.set_arg(11, np.int32(self.height))
        self.kernel_prog.set_arg(12, np.int32(self.max_iter))
        self.kernel_prog.set_arg(13, np.int32(self.samples))
        self.kernel_prog.set_arg(14, np.int32(self.order))
        self.kernel_prog.set_arg(15, wth)

        global_size = (self.width, self.height)
        local_size = None
        cl.enqueue_nd_range_kernel(self.queue, self.kernel_prog, global_size,
                                   local_size)

        # Download
        cl.enqueue_copy(self.queue, self.iter_np, self.iter_buf)
        self.queue.finish()

        if self.enable_timing:
            print(f"OpenCL perturb render time: {time.time() - start:.3f}s")

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

    def _render_cuda_perturb(self, min_x, max_x, min_y, max_y):
        start = time.time() if self.enable_timing else None

        width, height = self.width, self.height
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        c_ref = complex(cx, cy)

        step_x = complex((max_x - min_x) / float(width), 0.0)
        step_y = complex(0.0, (max_y - min_y) / float(height))
        c0 = complex(min_x, min_y)

        # High-precision orbit on CPU -> float64 (host)
        zref_f64 = make_reference_orbit_hp(c_ref, self.max_iter,
                                           mp_dps=self.hp_dps)

        # Device buffers
        d_zref = cuda.to_device(zref_f64)
        d_out = cuda.device_array((height, width), dtype=np.int32)

        # Launch
        self.cuda_perturb_kernel[self.blocks_per_grid, self.threads_per_block](
        d_zref, np.int32(len(d_zref)),
        np.float64(c_ref.real), np.float64(c_ref.imag),
        np.float64(c0.real), np.float64(c0.imag),
        np.float64(step_x.real), np.float64(step_x.imag),
        np.float64(step_y.real), np.float64(step_y.imag),
        np.int32(width), np.int32(height), np.int32(self.max_iter),
        np.int32(self.order), np.float64(self.w_fallback_thresh),
        d_out
        )
        cuda.synchronize()
        counts = d_out.copy_to_host()

        # Optional supersampling like CPU path (repeat kernel per subpixel)
        if self.samples > 1:
            total_samples = self.samples * self.samples
            accum = counts.astype(np.float64) / float(self.max_iter)
            for sx in range(self.samples):
                for sy in range(self.samples):
                    if sx == 0 and sy == 0:
                        continue
                    off_x = (sx + 0.5) / self.samples
                    off_y = (sy + 0.5) / self.samples
                    c0_off = complex(
                        c0.real + off_x * step_x.real + off_y * step_y.real,
                        c0.imag + off_x * step_x.imag + off_y * step_y.imag)

                    self.cuda_perturb_kernel[
                        self.blocks_per_grid, self.threads_per_block](
                        d_zref, np.int32(len(d_zref)),
                        np.float64(c_ref.real), np.float64(c_ref.imag),
                        np.float64(c0_off.real), np.float64(c0_off.imag),
                        np.float64(step_x.real), np.float64(step_x.imag),
                        np.float64(step_y.real), np.float64(step_y.imag),
                        np.int32(width), np.int32(height),
                        np.int32(self.max_iter),
                        np.int32(self.order),
                        np.float64(self.w_fallback_thresh),
                        d_out
                    )
                    cuda.synchronize()
                    counts = d_out.copy_to_host()
                    accum += counts.astype(np.float64) / float(self.max_iter)

                img = (accum / float(total_samples)).astype(self.precision)
        else:
            img = (counts.astype(np.float64) / float(
                self.max_iter)).astype(self.precision)

        if self.enable_timing:
            print(f"CUDA perturb render time: {time.time() - start:.3f}s")
        return img

    # ---------------- CPU RENDER -------------------
    def _render_cpu(self, min_x, max_x, min_y, max_y):
        start = time.time() if self.enable_timing else None

        real = np.linspace(min_x, max_x, self.width, dtype=self.precision)
        imag = np.linspace(min_y, max_y, self.height, dtype=self.precision)
        real_grid, imag_grid = np.meshgrid(real, imag)
        img = self.cpu_kernel(real_grid, imag_grid,
                              self.width, self.height,
                              self.max_iter, self.samples)
        if self.enable_timing:
            print(f"CPU render time: {time.time() - start:.3f}s")
        return img

    def _render_cpu_perturb(self, min_x, max_x, min_y, max_y):
        start = time.time() if self.enable_timing else None

        width, height = self.width, self.height

        # Reference parameter at viewport center
        cx = 0.5 * (min_x + max_x)
        cy = 0.5 * (min_y + max_y)
        c_ref = complex(cx, cy)

        # Axis-aligned steps (no rotation here; add rotation if needed)
        step_x = complex((max_x - min_x) / float(width), 0.0)
        step_y = complex(0.0, (max_y - min_y) / float(height))
        c0 = complex(min_x, min_y)

        # High-precision reference orbit -> downcast and ensure contiguity
        zref = make_reference_orbit_hp(c_ref, self.max_iter,
                                       mp_dps=self.hp_dps)  # (N,2) float64
        # Keep as float64 for the CPU perturb kernel (it expects float64), and ensure C-contiguous
        zref = np.ascontiguousarray(zref, dtype=np.float64)

        total_samples = self.samples * self.samples
        accum = np.zeros((height, width), dtype=np.float64)

        # Supersampling: reuse the same zref; offset c0 per subpixel
        for sx in range(self.samples):
            for sy in range(self.samples):
                off_x = (sx + 0.5) / self.samples
                off_y = (sy + 0.5) / self.samples
                c0_off = c0 + off_x * step_x + off_y * step_y

                counts = mandelbrot_kernel_cpu_perturb(
                    zref=zref, c_ref=c_ref,
                    width=width, height=height,
                    c0=c0_off, step_x=step_x, step_y=step_y,
                    max_iter=self.max_iter,
                    order=self.order, w_fallback_thresh=self.w_fallback_thresh
                )
                # Normalize to [0,1] like your OpenCL paths (which average smooth iterations)
                accum += counts.astype(np.float64) / float(self.max_iter)

        img = (accum / float(total_samples)).astype(self.precision)

        if self.enable_timing:
            print(f"CPU perturb render time: {time.time() - start:.3f}s")
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

        self.use_perturb = True if self.precision == Precisions.arbitrary else False

        self._init_kernel()
        print(f"Precision changed to {self.precision} successfully.")

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
    mb = Mandelbrot(kernel=Kernel.OPENCL, img_width=800, img_height=600,
                    max_iter=1500, precision=np.float64,
                    samples=2, use_perturb=True, order=2,
                    w_fallback_thresh=1e-6, hp_dps=160)

    # Seahorse Valley neighborhood (example)
    center_x, center_y = -0.743643887037151, 0.131825904205330
    span_x = 2e-9
    span_y = 1.5e-9

    img = mb.render(min_x=center_x - span_x, max_x=center_x + span_x,
                    min_y=center_y - span_y, max_y=center_y + span_y)

    print(img.shape, img.dtype, np.nanmin(img), np.nanmax(img))
