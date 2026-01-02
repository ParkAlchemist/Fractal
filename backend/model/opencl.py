import numpy as np
import pyopencl as cl
from typing import Dict, Any, Optional, Tuple, List

from fractals.mandelbrot import MandelbrotFractal
from utils.backend_helpers import clear_cache_lock
from fractals.base import Fractal, Viewport, RenderSettings
from backend.model.base import Backend

class OpenClBackend(Backend):
    """
    Backend for OpenCL-based fractal rendering.
    """
    name = "OPENCL"

    def __init__(self, device: Optional[int] = None, queues: int = 3, out_of_order: bool = False):

        clear_cache_lock()

        plats: list[cl.Platform] = cl.get_platforms()
        all_devs: list[tuple[int, cl.Device]] = []
        ordinal = 0
        for p in plats:
            for d in p.get_devices():
                all_devs.append((ordinal, d))
                ordinal += 1

        if not all_devs:
            raise RuntimeError("No OpenCL devices found.")

        if device is None:
            chosen = None
            for ord_id, d in all_devs:
                if d.type == cl.device_type.GPU:
                    chosen = (ord_id, d)
                    break
            if chosen is None:
                chosen = all_devs[0]
        else:
            chosen = next(((ord_id, d) for ord_id, d in all_devs if ord_id == device), None)
            if chosen is None:
                raise RuntimeError(f"No OpenCL device with ordinal {device} found.")

        self.device_ordinal, self.device = chosen

        # Context
        self.ctx: cl.Context | None = cl.Context([self.device])

        # Queue properties
        props: cl.command_queue_properties = 0
        if out_of_order:
            props |= cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE

        self.queues: list[cl.CommandQueue] | None = [cl.CommandQueue(self.ctx, self.device, properties=props) for _ in range(queues)] if queues > 0 else []

        self._kernel: cl.Kernel | None = None
        self._precision_cast = np.float32

    @staticmethod
    def enumerate_devices() -> List[dict]:
        devs: list[dict] = []
        ordinal = 0
        for p in cl.get_platforms():
            for d in p.get_devices():
                name = d.name.strip()
                vendor = d.vendor.strip()
                driver = getattr(d, "driver_version", None)
                cl_ver = getattr(p, "version", None)
                total_mb = int(getattr(d, "global_mem_size", 0) // (1024 ** 2))
                devs.append({
                    "device_id": ordinal,
                    "name": name,
                    "vendor": vendor,
                    "driver": driver,
                    "compute_capability": cl_ver,
                    "memory_total_mb": total_mb,
                    "memory_free_mb": None,
                    "is_available": True
                })
                ordinal += 1
        return devs

    def _get_queue(self) -> cl.CommandQueue:
        if not self.queues:
            raise RuntimeError("Backend has been closed or not initialized")

        q = self.queues.pop(0)
        self.queues.append(q)
        return q

    def compile(self, fractal: Fractal, settings: RenderSettings) -> None:

        spec = fractal.get_backend_spec(settings, self.name)
        self._precision_cast = spec["precision"]
        program = cl.Program(self.ctx, spec["kernel_source"]).build()
        self._kernel = cl.Kernel(program, spec["kernel_name"])
        self._warmup()

    def _warmup(self) -> None:
        if self._kernel is None:
            return
        q = self._get_queue()
        w, h = 64, 64
        vp = Viewport(-0.5 - 1.0, 0.5 + 1.0, -1.0, 1.0, w, h)
        st = RenderSettings(max_iter=64, samples=1, precision=self._precision_cast)
        arr, evt = self.render_async(fractal=MandelbrotFractal(), vp=vp, settings=st, queue=q)
        evt.wait()

    def render_async(self,
                     fractal: Fractal,
                     vp: Viewport,
                     settings: RenderSettings,
                     reference: Optional[Dict[str, Any]] = None,
                     queue: Optional[cl.CommandQueue] = None) -> Tuple[np.ndarray, cl.Event]:
        """
        Asynchronous rendering.
        - Enqueue kernel on a chosen queue
        - Enqueue non-blocking read into a host array
        - Return (host_array, completion_event)
        """

        if self._kernel is None:
            raise RuntimeError("Backend has not been compiled yet")

        q = queue or self._get_queue()
        params = fractal.get_backend_params(vp, settings)

        # Output buffer (device)
        out_np = np.empty(params["width"] * params["height"],
                          dtype=settings.precision)

        # Device buffer
        out_buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, out_np.nbytes)

        # Set arguments
        self._kernel.set_arg(0, self._precision_cast(params["min_x"]))
        self._kernel.set_arg(1, self._precision_cast(params["max_x"]))
        self._kernel.set_arg(2, self._precision_cast(params["min_y"]))
        self._kernel.set_arg(3, self._precision_cast(params["max_y"]))
        self._kernel.set_arg(4, out_buf)
        self._kernel.set_arg(5, np.int32(params["width"]))
        self._kernel.set_arg(6, np.int32(params["height"]))
        self._kernel.set_arg(7, np.int32(params["max_iter"]))
        self._kernel.set_arg(8, np.int32(params["samples"]))

        # Local/Global sizes
        lx, ly = 8, 8
        gx = ((params["width"] + lx - 1) // lx) * lx
        gy = ((params["height"] + ly - 1) // ly) * ly

        # Enqueue kernel
        kernel_evt = cl.enqueue_nd_range_kernel(q,
                                                self._kernel,
                                                global_work_size=(gx, gy),
                                                local_work_size=(lx, ly))

        # Non-blocking read: returns an event that depends on kernel_evt
        read_evt = cl.enqueue_copy(q, out_np, out_buf,
                                   is_blocking=False, wait_for=[kernel_evt])

        # Returns reshaped view and completion event
        return out_np.reshape((params["height"], params["width"])), read_evt

    def render(self, fractal: Fractal, vp: Viewport, settings: RenderSettings,
               reference: Optional[Dict[str, Any]] = None) -> np.ndarray:
        view, evt = self.render_async(fractal, vp, settings, reference)
        evt.wait()
        return view

    def close(self) -> None:
        if self.queues:
            for q in self.queues:
                try:
                    q.finish()
                except Exception as e:
                    print(f"Error in closing queue: {e}")
                    pass
            self.queues.clear()
            self.queues = None

        self._kernel = None
        self.device = None
        self.ctx = None
