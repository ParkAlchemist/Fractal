import numpy as np
import pyopencl as cl
import threading
from typing import Dict, Any, Optional

from utils import clear_cache_lock
from fractals.fractal_base import Fractal, Viewport, RenderSettings
from backends.backend_base import Backend
from kernel_sources.opencl.mandelbrot import (
    mandelbrot_kernel_cl_f32, mandelbrot_kernel_cl_f64,
    mandelbrot_kernel_cl_perturb_f32, mandelbrot_kernel_cl_perturb_f64,
)

class OpenClBackend(Backend):
    name = "OPENCL"

    def __init__(self, prefer_cpu=False):

        clear_cache_lock()

        plats = cl.get_platforms()
        self.device = None
        if prefer_cpu:
            for p in plats:
                devs = p.get_devices(cl.device_type.CPU)
                if devs: self.device = devs[0]; break
        if self.device is None:
            for p in plats:
                devs = p.get_devices(cl.device_type.GPU)
                if devs: self.device = devs[0]; break
        if self.device is None:
            for p in plats:
                devs = p.get_devices()
                if devs: self.device = devs[0]; break
        if self.device is None:
            raise RuntimeError("No OpenCL devices found.")

        self._tls = threading.local()

    def _get_ctx_queue(self):
        if not hasattr(self._tls, "ctx"):
            self._tls.ctx = cl.Context([self.device])
            self._tls.queue = cl.CommandQueue(self._tls.ctx, self.device)
        return self._tls.ctx, self._tls.queue

    def compile(self, fractal: Fractal, settings: RenderSettings) -> None:
        pass  # build per render

    def render(self, fractal: Fractal, vp: Viewport, settings: RenderSettings,
               reference: Optional[Dict[str, Any]]) -> np.ndarray:
        ctx, queue = self._get_ctx_queue()

        # Build program per call on this thread: maximal isolation
        if fractal.name != "mandelbrot":
            raise ValueError("Only Mandelbrot wired")
        if settings.precision == np.float32:
            src = mandelbrot_kernel_cl_perturb_f32 if settings.use_perturb else mandelbrot_kernel_cl_f32
            cast = np.float32
        else:
            src = mandelbrot_kernel_cl_perturb_f64 if settings.use_perturb else mandelbrot_kernel_cl_f64
            cast = np.float64

        program = cl.Program(ctx, src).build()
        kernel = cl.Kernel(program, "mandelbrot_perturb_kernel" if settings.use_perturb else "mandelbrot_kernel")

        out_np = np.empty(vp.width * vp.height, dtype=settings.precision)
        out_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, out_np.nbytes)

        zref_r_buf = zref_i_buf = None
        try:
            if settings.use_perturb:
                args = fractal.kernel_args(vp, settings, reference)
                zref = args["zref"]
                zref_r = np.ascontiguousarray(zref[:, 0], dtype=cast)
                zref_i = np.ascontiguousarray(zref[:, 1], dtype=cast)
                zref_r_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=zref_r)
                zref_i_buf = cl.Buffer(ctx, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=zref_i)

                kernel.set_arg(0, cast(args["c_ref"].real))
                kernel.set_arg(1, cast(args["c_ref"].imag))
                kernel.set_arg(2, zref_r_buf)
                kernel.set_arg(3, zref_i_buf)
                kernel.set_arg(4, np.int32(zref.shape[0]))
                kernel.set_arg(5, cast(vp.min_x)); kernel.set_arg(6, cast(vp.max_x))
                kernel.set_arg(7, cast(vp.min_y)); kernel.set_arg(8, cast(vp.max_y))
                kernel.set_arg(9, out_buf)
                kernel.set_arg(10, np.int32(vp.width)); kernel.set_arg(11, np.int32(vp.height))
                kernel.set_arg(12, np.int32(settings.max_iter))
                kernel.set_arg(13, np.int32(settings.samples))
                kernel.set_arg(14, np.int32(settings.perturb_order))
                kernel.set_arg(15, cast(settings.perturb_thresh))
            else:
                kernel.set_arg(0, cast(vp.min_x)); kernel.set_arg(1, cast(vp.max_x))
                kernel.set_arg(2, cast(vp.min_y)); kernel.set_arg(3, cast(vp.max_y))
                kernel.set_arg(4, out_buf)
                kernel.set_arg(5, np.int32(vp.width)); kernel.set_arg(6, np.int32(vp.height))
                kernel.set_arg(7, np.int32(settings.max_iter))
                kernel.set_arg(8, np.int32(settings.samples))

            # Conservative local and padded global
            lx, ly = 8, 8
            gx = ((vp.width  + lx - 1) // lx) * lx
            gy = ((vp.height + ly - 1) // ly) * ly
            cl.enqueue_nd_range_kernel(queue, kernel, (gx, gy), (lx, ly))
            cl.enqueue_copy(queue, out_np, out_buf)
            queue.finish()
        finally:
            try: out_buf.release()
            except Exception as e:
                print("Error when releasing output buffer: ", e)
                pass

            if zref_r_buf is not None:
                try: zref_r_buf.release()
                except Exception as e:
                    print("Error when releasing zref_r buffer: ", e)
                    pass
            if zref_i_buf is not None:
                try: zref_i_buf.release()
                except Exception as e:
                    print("Error when releasing zref_i buffer: ", e)
                    pass

        return out_np.reshape((vp.height, vp.width))
