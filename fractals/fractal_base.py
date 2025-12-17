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
    base_res: str = None
    target_res: str = None

class Fractal(ABC):
    name: str

    @abstractmethod
    def get_backend_params(self, viewport: Viewport,
                           settings: RenderSettings) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    def get_backend_spec(self, settings: RenderSettings,
                         backend_name: str) -> Optional[Dict[str, Any]]:
        ...

    @abstractmethod
    def output_semantics(self) -> str: ...
