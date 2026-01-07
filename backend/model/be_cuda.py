import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from numba import cuda

from fractals.base import Fractal, Viewport, RenderSettings, ProgramSpec
from backend.model.be_base import Backend


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

    @staticmethod
    def enumerate_devices() -> List[dict]:
        """
        Return a list of dicts compatible with DeviceInfo(**dict)
        """
        if not cuda.is_available():
            return []
        devices = []
        for i, gpu in enumerate(cuda.gpus):
            try:
                with gpu:
                    dev = cuda.get_current_device()
                    name = dev.name
                    cc = dev.compute_capability
                    cc_str = f"{cc[0]}.{cc[1]}"
                    free_b, total_b = cuda.current_context().get_memory_info()
                    total_mb = int(total_b // (1024.0 ** 2))
                    free_mb = int(free_b // (1024.0 ** 2))
                    devices.append({
                        "device_id": i,
                        "name": name,
                        "vendor": "NVIDIA",
                        "driver": None,
                        "compute_capability": cc_str,
                        "memory_total_mb": total_mb,
                        "memory_free_mb": free_mb,
                        "is_available": True,
                    })
            except Exception:
                devices.append({
                    "device_id": i,
                    "name": f"CUDA Device {i}",
                    "vendor": "NVIDIA",
                    "driver": None,
                    "compute_capability": None,
                    "memory_total_mb": None,
                    "memory_free_mb": None,
                    "is_available": False,
                })
        return devices

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
    def render_async(self,
                     fractal: Fractal,
                     vp: Viewport,
                     settings: RenderSettings,
                     reference: Optional[Dict[str, Any]] = None,
                     stream: Optional[cuda.stream] = None,
                     ) -> Tuple[np.ndarray, Any]:
        """
        Asynchronous rendering.
        - Allocates device output
        - Launches kernel into a stream
        - Copies back into a pinned host array asynchronously
        - Returns (host_array_view, completion_event)
        """

        if self.program is None:
            raise RuntimeError("Backend has not been compiled yet")

        # Compute launch dims
        self.blocks_per_grid = (
            (vp.width + self.threads_per_block[0] - 1) // self.threads_per_block[0],
            (vp.height + self.threads_per_block[1] - 1) // self.threads_per_block[1],
        )
        s = stream or self._get_stream()

        # Prepare scalars
        scalars = fractal.build_arg_values(vp, settings)

        # Allocate buffers
        H, W = vp.height, vp.width
        out_dev = cuda.device_array((H, W), dtype=self.program.precision)
        out_host = cuda.pinned_array(shape=(H, W), dtype=self.program.precision)

        # Build arg map
        arg_map: Dict[str, Any] = {}
        for name, spec in self.program.args.items():
            if spec.role == "buffer_out":
                arg_map[name] = out_dev
            elif spec.role == "scalar":
                val = scalars.get(name)

                if val is None and reference is not None:
                    val = reference.get(name)
                if val is None:
                    raise KeyError(
                        f"Missing scalar value for '{name}' - "
                        f"ensure fractal.build_arg_values() provides it."
                    )

                # cast to the required dtype
                if spec.dtype in (np.float32, np.float64):
                    val = spec.dtype(val)
                elif spec.dtype in (np.int32, np.int64):
                    val = spec.dtype(val)

                arg_map[name] = val
            else:
                arg_map[name] = scalars.get(name)

        # ordered args for the kernel
        step = self.program.steps[0]
        ordered = [arg_map[name] for name in step.args]

        # Launch kernel
        step.func[self.blocks_per_grid, self.threads_per_block, s](*ordered)

        # Async copy device -> host
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
