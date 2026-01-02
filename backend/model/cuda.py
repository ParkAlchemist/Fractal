import numpy as np
from typing import Dict, Any, Optional, Tuple, List
from numba import cuda

from fractals.base import Fractal, Viewport, RenderSettings
from backend.model.base import Backend


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
        self.kernel_func = None
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

    def _get_stream(self) -> Any:
        # Round-robin allocation
        if not self.streams:
            raise RuntimeError("CUDA backend is closed or has no streams")
        s = self.streams.pop(0)
        self.streams.append(s)
        return s

    def compile(self, fractal: Fractal, settings: RenderSettings) -> None:

        spec = fractal.get_backend_spec(settings, self.name)
        self.kernel_func = spec["kernel_source"]
        self._cast = spec["precision"]
        self._warmup()

    def _warmup(self) -> None:
        """
        Used to warm up the backend with a small sample of the fractal to ensure the kernel is compiled by numba.
        """
        if self._warmed_up: return
        self.blocks_per_grid = (
            (self._wu_width + self.threads_per_block[0] - 1) // self.threads_per_block[0],
            (self._wu_height + self.threads_per_block[1] - 1) // self.threads_per_block[1],
        )

        s = self._get_stream()

        d_out = cuda.device_array((self._wu_height, self._wu_width), dtype=self._cast)
        self.kernel_func[self.blocks_per_grid, self.threads_per_block, s](
            self._cast(self._wu_min_x), self._cast(self._wu_max_x),
            self._cast(self._wu_min_y), self._cast(self._wu_max_y),
            d_out, self._cast(self._wu_max_iter), self._cast(self._wu_samples)
        )

        s.synchronize()
        self._warmed_up = True

    def render_async(self,
                     fractal: Fractal,
                     vp: Viewport,
                     settings: RenderSettings,
                     reference: Optional[Dict[str, Any]] = None,
                     stream: Optional[cuda.stream] = None,
                     ) -> Tuple[np.ndarray, cuda.event]:
        """
        Asynchronous rendering.
        - Allocates device output
        - Launches kernel into a stream
        - Copies back into a pinned host array asynchronously
        - Returns (host_array_view, completion_event)
        """

        if self.kernel_func is None:
            raise RuntimeError("Backend has not been compiled yet")

        params = fractal.get_backend_params(vp, settings)

        self.blocks_per_grid = (
            (params["width"] + self.threads_per_block[0] - 1) // self.threads_per_block[0],
            (params["height"] + self.threads_per_block[1] - 1) // self.threads_per_block[1],
        )

        s = stream or self._get_stream()

        # Device output
        d_out = cuda.device_array((params["height"], params["width"]),
                                  dtype=settings.precision)

        # Kernel launch on chosen stream
        self.kernel_func[self.blocks_per_grid, self.threads_per_block, s](
            self._cast(params["min_x"]), self._cast(params["max_x"]),
            self._cast(params["min_y"]), self._cast(params["max_y"]),
            d_out, self._cast(params["max_iter"]), self._cast(params["samples"])
        )

        # Pinned host buffer for async copy
        h_out = cuda.pinned_array(shape=(params["height"], params["width"]),
                                  dtype=settings.precision)

        # Async copy to host in the same stream
        d_out.copy_to_host(h_out, stream=s)

        # Create an event and record it after copy
        done_evt = cuda.event(timing=False)
        done_evt.record(stream=s)

        return h_out, done_evt

    def render(self, fractal: Fractal, vp: Viewport, settings: RenderSettings,
               reference: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Synchronous rendering.
        """
        h_out, evt = self.render_async(fractal, vp, settings, reference)
        evt.synchronize()
        return h_out

    def close(self) -> None:
        if self.streams:
            for s in self.streams:
                try:
                    s.close()
                except Exception as e:
                    print(f"Error in closing stream: {e}")
                    pass
            self.streams.clear()
            self.streams = None

        self.kernel_func = None
        self._cast = None
