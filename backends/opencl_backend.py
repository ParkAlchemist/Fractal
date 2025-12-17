import numpy as np
import pyopencl as cl
from typing import Dict, Any, Optional, Tuple

from utils.backend_helpers import clear_cache_lock
from fractals.fractal_base import Fractal, Viewport, RenderSettings
from backends.backend_base import Backend

class OpenClBackend(Backend):
    name = "OPENCL"

    def __init__(self, prefer_cpu=False, queues: int = 0, out_of_order: bool = False):

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

        # Context
        self.ctx = cl.Context([self.device])

        # Queue properties
        props: cl.command_queue_properties = 0
        if out_of_order:
            props |= cl.command_queue_properties.OUT_OF_ORDER_EXEC_MODE_ENABLE

        self.queues = [cl.CommandQueue(self.ctx, self.device, properties=props) for _ in range(queues)] if queues > 0 else []

        self._kernel = None
        self._precision_cast = np.float32

    def _get_queue(self) -> cl.CommandQueue:
        if not self.queues:
            return cl.CommandQueue(self.ctx, self.device)

        q = self.queues.pop(0)
        self.queues.append(q)
        return q

    def compile(self, fractal: Fractal, settings: RenderSettings) -> None:

        spec = fractal.get_backend_spec(settings, self.name)
        self._precision_cast = spec["precision"]
        program = cl.Program(self.ctx, spec["kernel_source"]).build()
        self._kernel = cl.Kernel(program, spec["kernel_name"])

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
