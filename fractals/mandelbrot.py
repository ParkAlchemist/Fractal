from dataclasses import dataclass
from typing import Dict, Any

from fractals.fractal_base import Fractal, Viewport


@dataclass
class MandelbrotFractal(Fractal):
    name: str = "mandelbrot"

    def parameters(self) -> Dict[str, Any]:
        return {}

    def build_reference(self, viewport: Viewport) -> Dict[str, Any]:
        pass

    def output_semantics(self) -> str:
        return "normalized"
