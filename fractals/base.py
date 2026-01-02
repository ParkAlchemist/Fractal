from dataclasses import dataclass
from typing import Dict, Any, Optional
import numpy as np
from abc import ABC, abstractmethod

@dataclass
class Viewport:
    """
    Holds the viewport parameters for rendering a fractal.
    X and Y limits determine the area of the fractal to render.
    Width and Height determine the size of the resulting image in pixels.
    """
    min_x: float
    max_x: float
    min_y: float
    max_y: float
    width: int
    height: int

@dataclass
class RenderSettings:
    """
    Holds the rendering settings for a fractal.
    Max_iter determines the maximum number of iterations for the fractal calculation.
    Samples controls the number of samples per pixel for antialiasing.
    Precision specifies the floating-point precision for calculations.
    Base_res and target_res specify the resolution settings for the fractal.
    """
    max_iter: int
    samples: int = 1
    precision: np.dtype = np.float64
    base_res: str = None
    target_res: str = None

class Fractal(ABC):
    """
    An abstract base class for fractal types.
    """
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
