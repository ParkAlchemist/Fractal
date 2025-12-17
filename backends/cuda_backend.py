import numpy as np
from typing import Dict, Any, Optional, Tuple
from numba import cuda

from fractals.fractal_base import Fractal, Viewport, RenderSettings
from backends.backend_base import Backend


class CudaBackend(Backend):
    name = "CUDA"

    def __init__(self, streams: int = 0):
        if not cuda.is_available():
            raise RuntimeError("CUDA not available")
        # Stream pool: 0 -> create on demand; else create N streams
        self.streams = [cuda.stream() for _ in range(streams)] if streams > 0 else []
        self.threads_per_block = (16, 16)
        self.blocks_per_grid = None
        self.kernel_func = None
        self._cast = None

    def _get_stream(self) -> cuda.cudadrv.driver.Stream:
        # Round-robin allocation or creation on demand
        if not self.streams:
            return cuda.stream()
        s = self.streams.pop(0)
        self.streams.append(s)
        return s

    def compile(self, fractal: Fractal, settings: RenderSettings) -> None:

        sepc = fractal.get_backend_spec(settings, self.name)
        self.kernel_func = sepc["kernel_source"]
        self._cast = sepc["precision"]

    def render_async(self,
                     fractal: Fractal,
                     vp: Viewport,
                     settings: RenderSettings,
                     reference: Optional[Dict[str, Any]] = None,
                     stream: Optional[cuda.cudadrv.driver.Stream] = None,
                     ) -> Tuple[np.ndarray, cuda.cudadrv.driver.Stream]:
        """
        Asynchronous rendering.
        - Allocates device output
        - Launches kernel into a stream
        - Copies back into pinned host array asynchronously
        - Returns (host_array_view, completion_event)
        """
        params = fractal.get_backend_params(vp, settings)

        if self.blocks_per_grid is None:
            self.blocks_per_grid = (
                (params["width"] + self.threads_per_block[0] - 1) // self.threads_per_block[0],
                (params["height"] + self.threads_per_block[1] - 1) // self.threads_per_block[1],
            )

        s = stream or self._get_stream()

        # Device output
        d_out = cuda.device_array((params["height"], params["width"]),
                                  dtype=settings.precision)

        # Kernel launch on chose stream
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
        s.record_event(done_evt)

        return h_out, done_evt

    def render(self, fractal: Fractal, vp: Viewport, settings: RenderSettings,
               reference: Optional[Dict[str, Any]] = None) -> np.ndarray:
        """
        Synchronous rendering.
        """
        h_out, evt = self.render_async(fractal, vp, settings, reference)
        evt.synchronize()
        return h_out
