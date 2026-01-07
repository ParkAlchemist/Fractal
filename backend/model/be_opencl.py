import numpy as np
import pyopencl as cl
from typing import Dict, Any, Optional, Tuple, List

from utils.backend_helpers import clear_cache_lock
from fractals.base import Fractal, Viewport, RenderSettings, ProgramSpec, KernelStep, ArgSpec
from backend.model.be_base import Backend

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
                if d.type & cl.device_type.GPU:
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

        self.program_spec: Optional[ProgramSpec] = None
        self._kernels: Dict[str, cl.Kernel] = {}    # step.name -> kernel
        self._precision_cast = np.float32

        self._fractal: Optional[Fractal] = None

        # Warmup parameters
        self._wu_w, self._wu_h = 64, 64
        self._wu_bounds = (-2.0, 1.0, -1.5, 1.5)
        self._wu_max_iter, self._wu_samples = 64, 1
        self._warmed_up = False

        self.local_size = (8, 8)

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

    @staticmethod
    def _np_dtype_of(dtype: Any) -> np.dtype:
        return np.dtype(dtype)

    def compile(self, fractal: Fractal, settings: RenderSettings) -> None:
        """
        Build kernels for all ProgramSpec steps.
        Expects the fractal to supply ProgramSpec via get_program_spec(settings, "OPENCL").
        For OpenCL, KernelStep.func can be:
            - a dict {"src": <str>, "kernel_name": <str>"}
            - a plain kernel source string (kernel name taken from step.name)
        """
        self.program_spec = fractal.get_program_spec(settings, self.name)
        self._precision_cast = self.program_spec.precision
        self._fractal = fractal
        self._kernels.clear()

        # Build one cl.Program for each distinct source in steps
        for step in self.program_spec.steps:
            src, kname = self._extract_opencl_source_and_name(step)
            program = cl.Program(self.ctx, src).build()
            kernel = cl.Kernel(program, kname)
            self._kernels[step.name] = kernel

        self._warmup()

    @staticmethod
    def _extract_opencl_source_and_name(step: KernelStep) -> Tuple[str, str]:
        """
        Extracts (source, kernel_name) for a KernelStep.
        Supports:
            - func is dict with keys: "src" and optional "kernel_name"
            - func is str (source), kernel_name taken from step.name
        """
        if isinstance(step.func, dict):
            src = step.func.get("src")
            if not isinstance(src, str) or not src.strip():
                raise ValueError(f"Invalid kernel source for step {step.name}: {src}")
            kname = step.func.get("kernel_name", step.name)
            if not isinstance(kname, str) or not kname.strip():
                raise ValueError(f"Invalid kernel name for step {step.name}: {kname}")
            return src, kname
        elif isinstance(step.func, str):
            return step.func, step.name
        else:
            raise TypeError(
                f"KernelStep {step.name}: unsupported func type {type(step.func)}; "
                f"use dict {{'src': ..., 'kernel_name': ...}} or plain source string."
            )

    def _warmup(self) -> None:
        if self._warmed_up or not self.program_spec or not self._fractal:
            return

        w, h = self._wu_w, self._wu_h
        minx, maxx, miny, maxy = self._wu_bounds
        vp = Viewport(min_x=minx, max_x=maxx, min_y=miny, max_y=maxy, width=w, height=h)
        st = RenderSettings(max_iter=self._wu_max_iter, samples=self._wu_samples, precision=self._precision_cast)

        scalars = self._fractal.build_arg_values(vp, st)

        step = self.program_spec.steps[0]
        kernel = self._kernels[step.name]

        arg_map, out_dev, _ = self._prepare_arg_map(self.program_spec.args, scalars, vp, st)

        self._bind_kernel_args(kernel, step.args, arg_map)

        q = self._get_queue()
        gxs, gys = self._compute_global_sizes(vp.width, vp.height, step)
        lxs, lys = self._compute_local_sizes(step)
        cl.enqueue_nd_range_kernel(q, kernel, global_work_size=(gxs, gys), local_work_size=(lxs, lys))
        q.finish()
        self._warmed_up = True

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

        if self.program_spec is None:
            raise RuntimeError("Backend has not been compiled yet")

        q = queue or self._get_queue()

        scalars = fractal.build_arg_values(vp, settings)

        arg_map, out_dev, out_host = self._prepare_arg_map(self.program_spec.args, scalars, vp, settings)

        last_read_evt: Optional[cl.Event] = None

        for step in self.program_spec.steps:
            kernel = self._kernels[step.name]
            self._bind_kernel_args(kernel, step.args, arg_map)

            gxs, gys = self._compute_global_sizes(vp.width, vp.height, step)
            lxs, lys = self._compute_local_sizes(step)

            k_evt = cl.enqueue_nd_range_kernel(q, kernel, global_work_size=(gxs, gys), local_work_size=(lxs, lys))

            if step is self.program_spec.steps[-1] and out_dev is not None and out_host is not None:
                last_read_evt = cl.enqueue_copy(q, out_host, out_dev, is_blocking=False, wait_for=[k_evt])

        if last_read_evt is None:
            out_host = np.zeros((vp.height, vp.width), dtype=self.program_spec.precision)
            last_read_evt = cl.enqueue_barrier(q)

        return out_host, last_read_evt

    def render(self,
               fractal: Fractal,
               vp: Viewport,
               settings: RenderSettings,
               reference: Optional[Dict[str, Any]] = None) -> np.ndarray:
        view, evt = self.render_async(fractal, vp, settings, reference)
        evt.wait()
        return view

    def _prepare_arg_map(
            self,
            args_spec: Dict[str, ArgSpec],
            scalars: Dict[str, Any],
            vp: Viewport,
            st: RenderSettings
    ) -> Tuple[Dict[str, Any], Optional[cl.Buffer], Optional[np.ndarray]]:
        """
        From ArgSpecs + scalar values, build a dict name->value for binding.
        Allocates device output buffers; returns (arg_map, out_dev, out_host).
        """
        H, W = int(vp.height), int(vp.width)
        arg_map: Dict[str, Any] = {}
        out_dev: Optional[cl.Buffer] = None
        out_host: Optional[np.ndarray] = None

        for name, spec in args_spec.items():
            if spec.role == "buffer_out":
                np_dt = self._np_dtype_of(spec.dtype)
                nbytes = H * W * np_dt.itemsize
                buf = cl.Buffer(self.ctx, cl.mem_flags.WRITE_ONLY, nbytes)
                arg_map[name] = buf

                if name == getattr(self.program_spec, "output_arg", name):
                    out_dev = buf
                    out_host = np.empty((H, W), dtype=np_dt)
            elif spec.role == "buffer_in":
                raise NotImplementedError("buffer_in not yet supported")
            elif spec.role == "scalar":
                if name in scalars:
                    val = scalars[name]
                else:
                    raise KeyError(
                            f"Missing scalar value for '{name}' – "
                            f"ensure fractal.build_arg_values() provides it."
                    )
                np_dt = self._np_dtype_of(spec.dtype)
                if np.issubdtype(np_dt, np.integer):
                    val = np_dt.type(int(val))
                else:
                    val = np_dt.type(float(val))
                arg_map[name] = val
            else:
                raise ValueError(f"Unsupported arg role {spec.role}")

        return arg_map, out_dev, out_host

    @staticmethod
    def _bind_kernel_args(kernel: cl.Kernel,
                          ordered_names: List[str],
                          arg_map: Dict[str, Any]) -> None:
        """
        Set kernel arguments by index according to ordered_names.
        """
        for idx, name in enumerate(ordered_names):
            if name not in arg_map:
                raise KeyError(f"Kernel arg {name} not found in arg_map")
            kernel.set_arg(idx, arg_map[name])

    def _compute_local_sizes(self, step: KernelStep) -> Tuple[int, int]:
        return int(self.local_size[0]), int(self.local_size[1])

    def _compute_global_sizes(self, w: int, h: int, step: KernelStep) -> Tuple[int, int]:
        lx, ly = self._compute_local_sizes(step)
        gx = ((w + lx - 1) // lx) * lx
        gy = ((h + ly - 1) // ly) * ly
        return gx, gy

    def close(self) -> None:
        if self.queues:
            for q in self.queues:
                try:
                    q.finish()
                except Exception as e:
                    print(f"Error in closing queue: {e}")
                    pass
            self.queues.clear()
        self.queues = []
        self._kernels.clear()
        self.program_spec = None
        self._fractal = None

if __name__ == "__main__":
    with OpenClBackend() as be:
        devs = be.enumerate_devices()
        if len(devs) == 0:
            raise RuntimeError("No devices found.")
        print("Available devices:")
        for i, d in enumerate(devs):
            print(f"{i}: {d}")
