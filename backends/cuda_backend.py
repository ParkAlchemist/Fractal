import numpy as np
from typing import Dict, Any, Optional
from numba import cuda

from fractals.fractal_base import Fractal, Viewport, RenderSettings
from backends.backend_base import Backend


class CudaBackend(Backend):
    name = "CUDA"

    def __init__(self):
        if not cuda.is_available():
            raise RuntimeError("CUDA not available")
        self.threads_per_block = (16, 16)
        self.blocks_per_grid = None
        self.kernel_func = None
        self._cast = None

    def compile(self, fractal: Fractal, settings: RenderSettings) -> None:

        params = fractal.get_kernel_source(settings, self.name)
        self.kernel_func = params["kernel_source"]
        self._cast = params["precision"]

    def render(self, fractal: Fractal, vp: Viewport, settings: RenderSettings,
               reference: Optional[Dict[str, Any]] = None) -> np.ndarray:

        args = fractal.get_kernel_args(vp, settings)

        if self.blocks_per_grid is None:
            self.blocks_per_grid = (
                (args["width"] + self.threads_per_block[0] - 1) // self.threads_per_block[0],
                (args["height"] + self.threads_per_block[1] - 1) // self.threads_per_block[1],
            )

        d_out = cuda.device_array((args["height"], args["width"]), dtype=settings.precision)
        self.kernel_func[self.blocks_per_grid, self.threads_per_block](
            self._cast(args["min_x"]), self._cast(args["max_x"]),
            self._cast(args["min_y"]), self._cast(args["max_y"]),
            d_out, self._cast(args["max_iter"]), self._cast(args["samples"])
        )
        cuda.synchronize()
        return d_out.copy_to_host()
