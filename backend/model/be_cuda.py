import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from numba import cuda
import logging

from fractals.base import Fractal, Viewport, RenderSettings, ProgramSpec
from fractals.spec_validator import validate_program_spec
from backend.model.be_base import Backend
from utils.shape_helper import eval_shape_expr, ShapeExprError


logger = logging.getLogger(__name__)


class CudaBackend(Backend):
    """
    Backend for CUDA-based fractal rendering.
    """
    name = "CUDA"

    def __init__(self, device: Optional[int] = None, streams: int = 3):
        if not cuda.is_available():
            raise RuntimeError("CUDA not available")
        if device is not None:
            cuda.select_device(device)
        # Stream pool: 0 -> create on demand; else create N streams
        self.streams = [cuda.stream() for _ in range(streams)] if streams > 0 else []
        self.threads_per_block = (16, 16)
        self.blocks_per_grid = None
        self.program: Optional[ProgramSpec] = None
        self._cast = None

        # Warm up params
        self._wu_min_x = -2.0
        self._wu_max_x = 1.0
        self._wu_min_y = -1.5
        self._wu_max_y = 1.5
        self._wu_width = 64
        self._wu_height = 64
        self._wu_max_iter = 64
        self._wu_samples = 1
        self._warmed_up = False

    # ------ stream helpers ---------
    def _get_stream(self) -> Any:
        # Round-robin allocation
        if not self.streams:
            raise RuntimeError("CUDA backend is closed or has no streams")
        s = self.streams.pop(0)
        self.streams.append(s)
        return s

    # ------ compilation --------
    def compile(self, fractal: Fractal, settings: RenderSettings) -> None:
        """
        Pull a generic program spec from the fractal and prepare warmup.
        """
        spec = fractal.get_program_spec(settings, self.name)
        validate_program_spec(spec, backend_hint=self.name)
        self.program = spec
        self._cast = spec.precision
        self._warmup()

    def _warmup(self) -> None:
        """
        Used to warm up the backend with a small sample of the fractal to ensure the kernel is compiled by numba.
        """
        if self._warmed_up or self.program is None:
            return

        step = self.program.steps[0]

        self.blocks_per_grid = (
            (self._wu_width + self.threads_per_block[0] - 1) // self.threads_per_block[0],
            (self._wu_height + self.threads_per_block[1] - 1) // self.threads_per_block[1],
        )

        s = self._get_stream()
        d_out = cuda.device_array((self._wu_height, self._wu_width), dtype=self._cast)

        arg_map = {
            "min_x": self._cast(self._wu_min_x),
            "max_x": self._cast(self._wu_max_x),
            "min_y": self._cast(self._wu_min_y),
            "max_y": self._cast(self._wu_max_y),
            "out": d_out,
            "max_iter": self._cast(self._wu_max_iter),
            "samples": self._cast(self._wu_samples),
        }
        ordered = [arg_map[name] for name in step.args]

        step.func[self.blocks_per_grid, self.threads_per_block, s](*ordered)
        s.synchronize()
        self._warmed_up = True

    # ------- rendering --------

    def render_async(
            self,
            fractal: Fractal,
            vp: Viewport,
            settings: RenderSettings,
            reference: Optional[Dict[str, Any]] = None,
            stream: Optional[cuda.stream] = None,
    ) -> Tuple[np.ndarray, Any]:
        """
        Generic asynchronous rendering with shape_expr-aware allocation.
        """
        if self.program is None:
            raise RuntimeError("Backend has not been compiled yet")

        # launch dims for 2D kernels (most of yours)
        self.blocks_per_grid = (
            (vp.width + self.threads_per_block[0] - 1) //
            self.threads_per_block[0],
            (vp.height + self.threads_per_block[1] - 1) //
            self.threads_per_block[1],
        )
        s = stream or self._get_stream()

        scalars = fractal.build_arg_values(vp, settings)
        H, W = int(vp.height), int(vp.width)
        S = int(scalars.get("ssaa", 1))
        base_vars = {"H": H, "W": W, "width": W, "height": H, "S": S,
                     "C": int(scalars.get("channels", 1)),
                     "N": int(scalars.get("N", scalars.get("palette_ex_len", scalars.get("palette_len",)))),
                     "M": int(scalars.get("M", scalars.get("palette_in_len",scalars.get("palette_len",0))))}

        out_dev = None
        out_host = None
        arg_map: Dict[str, Any] = {}

        # Allocate & bind
        for name, spec in self.program.args.items():
            if spec.role == "buffer_out":
                np_dt = np.dtype(spec.dtype)
                shape = (H, W)
                if spec.shape_expr:
                    try:
                        shape = eval_shape_expr(spec.shape_expr, base_vars)
                    except ShapeExprError as e:
                        raise KeyError(f"Failed to allocate '{name}': {e}")
                dev_arr = cuda.device_array(shape, dtype=np_dt)
                arg_map[name] = dev_arr
                if name == self.program.output_arg:
                    out_dev = dev_arr
                    out_host = cuda.pinned_array(shape=shape, dtype=np_dt)

            elif spec.role == "buffer_in":
                if name not in scalars:
                    raise KeyError(
                        f"buffer_in '{name}' not provided in scalars.")
                host = scalars[name]
                if hasattr(host, "device_ctypes_pointer"):
                    # already a device array
                    arg_map[name] = host
                else:
                    arr = np.asarray(host)
                    if spec.shape_expr:
                        try:
                            exp_shape = eval_shape_expr(spec.shape_expr,
                                                        base_vars)
                            if tuple(arr.shape) != tuple(exp_shape):
                                raise ValueError(
                                    f"'{name}' expected shape {exp_shape}, got {arr.shape}.")
                        except ShapeExprError:
                            pass
                    if arr.dtype != np.dtype(spec.dtype):
                        arr = arr.astype(spec.dtype, copy=False)
                    arg_map[name] = cuda.to_device(arr, stream=s)

            elif spec.role == "scalar":
                val = scalars.get(name)
                if val is None and reference is not None:
                    val = reference.get(name)
                if val is None:
                    raise KeyError(
                        f"Missing scalar value for '{name}' – "
                        f"ensure fractal.build_arg_values() provides it."
                    )
                if spec.dtype in (np.float32, np.float64):
                    val = spec.dtype(val)
                elif spec.dtype in (np.int32, np.int64):
                    val = spec.dtype(val)
                arg_map[name] = val

            else:
                raise ValueError(
                    f"Unknown ArgSpec.role for '{name}': {spec.role}")

        # Ordered args & launch
        step = self.program.steps[0]
        ordered = [arg_map[name] for name in step.args]
        step.func[self.blocks_per_grid, self.threads_per_block, s](*ordered)

        # Async copy back only for final output buffer
        if out_dev is not None and out_host is not None:
            out_dev.copy_to_host(out_host, stream=s)
        done_evt = cuda.event(timing=False)
        done_evt.record(stream=s)
        return out_host, done_evt

    def render(self, fractal: Fractal, vp: Viewport, settings: RenderSettings,
               reference: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Synchronous rendering.
        """
        out, evt = self.render_async(fractal, vp, settings, reference)
        evt.synchronize()
        return out

    def close(self) -> None:
        if self.streams:
            for s in self.streams:
                try:
                    s.synchronize()
                except Exception as e:
                    print(f"Error in closing stream: {e}")
                    pass
            self.streams.clear()
        self.streams = None
        self.program = None
        self._cast = None


if __name__ == "__main__":
    with CudaBackend() as be:
        devs = be.enumerate_devices()
        if len(devs) == 0:
            raise RuntimeError("No devices found.")
        print("Available devices:")
        for i, d in enumerate(devs):
            print(f"{i}: {d}")
