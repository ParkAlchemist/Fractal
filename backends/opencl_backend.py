import numpy as np
import pyopencl as cl
import threading
from typing import Dict, Any, Optional

from utils.utils import clear_cache_lock
from fractals.fractal_base import Fractal, Viewport, RenderSettings
from backends.backend_base import Backend

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
        self._kernel = None

    def _get_ctx_queue(self):
        if not hasattr(self._tls, "ctx"):
            self._tls.ctx = cl.Context([self.device])
            self._tls.queue = cl.CommandQueue(self._tls.ctx, self.device)
        return self._tls.ctx, self._tls.queue

    def compile(self, fractal: Fractal, settings: RenderSettings) -> None:

        params = fractal.get_kernel_source(settings, self.name)
        self._cast = params["precision"]

        ctx, _ = self._get_ctx_queue()
        program = cl.Program(ctx, params["kernel_source"]).build()
        self._kernel = cl.Kernel(program, params["kernel_name"])

    def render(self, fractal: Fractal, vp: Viewport, settings: RenderSettings,
               reference: Optional[Dict[str, Any]] = None) -> np.ndarray:
        ctx, queue = self._get_ctx_queue()

        args = fractal.get_kernel_args(vp, settings)

        out_np = np.empty(args["width"] * args["height"], dtype=settings.precision)
        out_buf = cl.Buffer(ctx, cl.mem_flags.WRITE_ONLY, out_np.nbytes)

        try:
            self._kernel.set_arg(0, self._cast(args["min_x"]))
            self._kernel.set_arg(1, self._cast(args["max_x"]))
            self._kernel.set_arg(2, self._cast(args["min_y"]))
            self._kernel.set_arg(3, self._cast(args["max_y"]))
            self._kernel.set_arg(4, out_buf)
            self._kernel.set_arg(5, np.int32(args["width"]))
            self._kernel.set_arg(6, np.int32(args["height"]))
            self._kernel.set_arg(7, np.int32(args["max_iter"]))
            self._kernel.set_arg(8, np.int32(args["samples"]))

            # Conservative local and padded global
            lx, ly = 8, 8
            gx = ((args["width"]  + lx - 1) // lx) * lx
            gy = ((args["height"] + ly - 1) // ly) * ly
            cl.enqueue_nd_range_kernel(queue, self._kernel, (gx, gy), (lx, ly))
            cl.enqueue_copy(queue, out_np, out_buf)
            queue.finish()
        finally:
            try: out_buf.release()
            except Exception as e:
                print("Error when releasing output buffer: ", e)
                pass

        return out_np.reshape((args["height"], args["width"]))
