from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from fractals.fractal_base import Fractal, Viewport, RenderSettings

class Backend(ABC):
    name: str

    @abstractmethod
    def compile(self, fractal: Fractal, settings: RenderSettings) -> None: ...

    @abstractmethod
    def render(self, fractal: Fractal, vp: Viewport, settings: RenderSettings,
               reference: Optional[Dict[str, Any]]) -> np.ndarray: ...
