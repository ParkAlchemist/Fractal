from abc import ABC, abstractmethod
import numpy as np

from utils.enums import ColoringMode

class ColoringStrategy(ABC):
    @abstractmethod
    def apply(self, iter_buf: np.ndarray, mode: ColoringMode,
              exterior_palette: np.ndarray, interior_palette: np.ndarray,
              interior_color=(100, 100, 100)) -> np.ndarray:
        ...
