from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np

from fractals.fractal_base import Fractal, Viewport, RenderSettings
from utils import make_reference_orbit_hp

@dataclass
class MandelbrotFractal(Fractal):
    name: str = "mandelbrot"

    def parameters(self) -> Dict[str, Any]:
        return {}

    def supports_perturbation(self) -> bool:
        return True

    def build_reference(self, vp: Viewport, settings: RenderSettings) -> Optional[Dict[str, Any]]:
        cx = 0.5 * (vp.min_x + vp.max_x)
        cy = 0.5 * (vp.min_y + vp.max_y)
        c_ref = complex(cx, cy)
        zref = make_reference_orbit_hp(c_ref, settings.max_iter, mp_dps=settings.hp_dps)
        return {"c_ref": c_ref, "zref": np.ascontiguousarray(zref, dtype=np.float64)}

    def kernel_args(self, vp: Viewport, settings: RenderSettings,
                    reference: Optional[Dict[str, Any]]) -> Dict[str, Any]:
        step_x = complex((vp.max_x - vp.min_x) / float(vp.width), 0.0)
        step_y = complex(0.0, (vp.max_y - vp.min_y) / float(vp.height))
        c0 = complex(vp.min_x, vp.min_y)
        args = {"c0": c0, "step_x": step_x, "step_y": step_y}
        if reference: args.update(reference)
        return args

    def output_semantics(self) -> str:
        return "normalized"
