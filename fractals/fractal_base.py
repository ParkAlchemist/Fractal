from dataclasses import dataclass
from typing import Optional, Dict, Any
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
    use_perturb: bool = False
    perturb_order: int = 2
    perturb_thresh: float = 1e-6
    hp_dps: int = 160

class Fractal(ABC):
    name: str

    @abstractmethod
    def parameters(self) -> Dict[str, Any]: ...

    @abstractmethod
    def supports_perturbation(self) -> bool: ...

    @abstractmethod
    def build_reference(self, vp: Viewport, settings: RenderSettings) -> Optional[Dict[str, Any]]: ...

    @abstractmethod
    def kernel_args(self, vp: Viewport, settings: RenderSettings,
                    reference: Optional[Dict[str, Any]]) -> Dict[str, Any]: ...

    @abstractmethod
    def output_semantics(self) -> str: ...
