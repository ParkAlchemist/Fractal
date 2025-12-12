from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from abc import ABC, abstractmethod

@dataclass
class Viewport:
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    width: int
    height: int

@dataclass
class RenderSettings:
    max_iter: int
    samples: int = 1
    precision: np.dtype = np.float64

class Fractal(ABC):
    name: str

    @abstractmethod
    def parameters(self) -> Dict[str, Any]: ...

    @abstractmethod
    def build_reference(self, viewport: Viewport) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def output_semantics(self) -> str: ...
