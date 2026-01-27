from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np
from fractals.base import Fractal, Viewport, RenderSettings


class Backend(ABC):
    """
    A base class for fractal rendering backend.
    """
    name: str

    @abstractmethod
    def compile(self,
                fractal: Fractal,
                settings: RenderSettings
                ) -> None:
        ...

    @abstractmethod
    def render_async(self, fractal: Fractal,
                     vp: Viewport,
                     settings: RenderSettings,
                     reference: Optional[Dict[str, Any]] = None
                     ) -> tuple[np.ndarray, Any]:
        ...

    @abstractmethod
    def render(self,
               fractal: Fractal,
               vp: Viewport,
               settings: RenderSettings,
               reference: Optional[Dict[str, Any]] = None
               ) -> np.ndarray:
        ...

    @abstractmethod
    def close(self) -> None:
        ...

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass
