import numpy as np
from typing import Dict, Any, Optional, Tuple

from fractals.base import Fractal, Viewport, RenderSettings, ProgramSpec, ArgSpec
from fractals.spec_validator import validate_program_spec
from backend.model.be_base import Backend
from utils.shape_helper import eval_shape_expr, ShapeExprError


class CpuEvent:
    """A minimal event-like object to mirror async interfaces (no-op)."""
    @staticmethod
    def wait() -> None:
        return None


class CpuBackend(Backend):
    """
    Backend for CPU-based fractal rendering.
    """
    name = "CPU"

    def __init__(self):
        self.program_spec: Optional[ProgramSpec] = None
        self._precision_cast = np.float32
        self._fractal: Optional[Fractal] = None
        self._warmed_up = False

        # Warmup configuration
        self._wu_w, self._wu_h = 64, 64
        self._wu_bounds = (-2.0, 1.0, -1.5, 1.5)
        self._wu_max_iter, self._wu_samples = 64, 1

    def compile(self, fractal: Fractal, settings: RenderSettings) -> None:
        """
        Pull the generic ProgramSpec from the fractal for the backend.
        Expects KernelStep.func to be a Python callable (Numba njit or pure Python).
        """
        spec = fractal.get_program_spec(settings, self.name)
        validate_program_spec(spec, backend_hint=self.name)
        self.program_spec = spec
        self._precision_cast = spec.precision
        self._fractal = fractal
        self._warmup()

    def _warmup(self) -> None:
        if self._warmed_up or not self.program_spec or not self._fractal:
            return

        w, h = self._wu_w, self._wu_h
        minx, maxx, miny, maxy = self._wu_bounds
        vp = Viewport(minx, maxx, miny, maxy, w, h)
        st = RenderSettings(
            max_iter=self._wu_max_iter,
            samples=self._wu_samples,
            precision=self._precision_cast
        )

        scalars = self._fractal.build_arg_values(vp, st)
        arg_map, _, _ = self._prepare_arg_map(self.program_spec.args, scalars,
                                              vp, st)

        step = self.program_spec.steps[0]
        ordered = [arg_map[name] for name in step.args]

        # Call CPU kernel (Numba njit or Python).
        ret = step.func(*ordered)
        _ = ret  # ignore; just ensure it executes

        self._warmed_up = True

    def render_async(
            self,
            fractal: Fractal,
            vp: Viewport,
            settings: RenderSettings,
            reference: Optional[Dict[str, Any]] = None,
    ) -> Tuple[np.ndarray, CpuEvent]:
        """
        Synchronous CPU execution, but returns (host_array, CpuEvent) for API parity.
        """
        if self.program_spec is None:
            raise RuntimeError("Backend has not been compiled yet")

        scalars = fractal.build_arg_values(vp, settings)
        arg_map, out_arr, _ = self._prepare_arg_map(self.program_spec.args,
                                                    scalars, vp, settings)

        # Execute steps in order; keep arg_map across steps so later steps
        # can reuse buffers written by earlier ones (buffer_in/out pattern).
        last_result = None
        for step in self.program_spec.steps:
            ordered = [arg_map[name] for name in step.args]
            last_result = step.func(*ordered)

            # If the CPU kernel returns an array, allow it to override the out buffer
            if last_result is not None and self.program_spec.output_arg in self.program_spec.args:
                out_arg = self.program_spec.output_arg
                if isinstance(last_result, np.ndarray):
                    # Replace mapped buffer with returned one
                    arg_map[out_arg] = last_result
                    out_arr = last_result
        if out_arr is None:
            # If nothing was produced, return a zero array to avoid None
            out_arr = np.zeros((vp.height, vp.width),
                               dtype=self.program_spec.precision)
        return out_arr, CpuEvent()

    def render(
            self,
            fractal: Fractal,
            vp: Viewport,
            settings: RenderSettings,
            reference: Optional[Dict[str, Any]] = None
    ) -> np.ndarray:
        view, evt = self.render_async(fractal, vp, settings, reference)
        evt.wait()
        return view

    def _prepare_arg_map(
            self,
            args_spec: Dict[str, ArgSpec],
            scalars: Dict[str, Any],
            vp: Viewport,
            st: RenderSettings
    ) -> Tuple[Dict[str, Any], Optional[np.ndarray], Optional[np.ndarray]]:
        """
        Build a dict name->value for function call.
        Returns (arg_map, out_array_view, tmp_host) where out_array_view is the final
        output buffer (NumPy ndarray) if a 'buffer_out' matches output_arg.
        """
        H, W = int(vp.height), int(vp.width)
        arg_map: Dict[str, Any] = {}
        out_arr: Optional[np.ndarray] = None

        # Vars for shape expressions
        S = int(scalars.get("ssaa", 1))
        base_vars = {
            "H": H, "W": W, "width": W, "height": H,
            "S": S,
            "C": int(scalars.get("channels", 3)),  # default 3 for rgb
            "N": int(scalars.get("N", scalars.get("palette_ex_len",scalars.get("palette_len",0)))),
            "M": int(scalars.get("M", scalars.get("palette_in_len",scalars.get("palette_len",0)))),
        }

        for name, spec in args_spec.items():
            if spec.role == "buffer_out":
                np_dt = np.dtype(spec.dtype)
                # special-case rgb default channels if not provided
                if name.lower() == "rgb" and "channels" not in scalars:
                    base_vars["C"] = 3
                shape = (H, W)
                if spec.shape_expr:
                    try:
                        shape = eval_shape_expr(spec.shape_expr, base_vars)
                    except ShapeExprError as e:
                        raise KeyError(f"Failed to allocate '{name}': {e}")
                arr = np.zeros(shape, dtype=np_dt)
                arg_map[name] = arr
                if name == getattr(self.program_spec, "output_arg", name):
                    out_arr = arr

            elif spec.role == "buffer_in":
                if name not in scalars:
                    raise KeyError(
                        f"buffer_in '{name}' not provided in scalars.")
                arr = np.asarray(scalars[name])
                if spec.shape_expr:
                    try:
                        exp_shape = eval_shape_expr(spec.shape_expr, base_vars)
                        if tuple(arr.shape) != tuple(exp_shape):
                            raise ValueError(
                                f"'{name}' expected shape {exp_shape}, got {arr.shape}.")
                    except ShapeExprError:
                        pass
                if arr.dtype != np.dtype(spec.dtype):
                    arr = arr.astype(spec.dtype, copy=False)
                arg_map[name] = arr

            elif spec.role == "scalar":
                if name in scalars:
                    val = scalars[name]
                else:
                    raise KeyError(
                        f"Missing scalar value for '{name}' – "
                        f"ensure fractal.build_arg_values() provides it."
                    )
                np_dt = np.dtype(spec.dtype)
                if np.issubdtype(np_dt, np.integer):
                    val = np_dt.type(int(val))
                else:
                    val = np_dt.type(float(val))
                arg_map[name] = val

            else:
                raise ValueError(
                    f"Unknown ArgSpec.role for '{name}': {spec.role}")

        return arg_map, out_arr, None

    def close(self) -> None:
        self.program_spec = None
        self._fractal = None
        self._warmed_up = False


if __name__ == "__main__":
    with CpuBackend() as be:
        devs = be.enumerate_devices()
        if len(devs) == 0:
            raise RuntimeError("No devices found.")
        print("Available devices:")
        for i, d in enumerate(devs):
            print(f"{i}: {d}")
