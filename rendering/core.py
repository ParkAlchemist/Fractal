import numpy as np
from typing import Optional

from fractals.base import Fractal, Viewport, RenderSettings
from backend.manager import BackendManager
from rendering.engines.full_frame import FullFrameEngine
from rendering.engines.base import BaseRenderEngine


class Renderer:
    """
    Main construct for combining the rendering pipeline: (engine + fractal + backend)
    """
    def __init__(self,
                 fractal: Fractal,
                 settings: RenderSettings,
                 engine: Optional[BaseRenderEngine] = None,
                 manager: Optional[BackendManager] = None,
                 default_backend: Optional[str] = None,
                 default_device: Optional[int] = None):
        self.fractal = fractal
        self.settings = settings
        self.engine = engine or FullFrameEngine()
        self.manager = manager or BackendManager()
        self.default_backend = default_backend
        self.default_device = default_device
        self.manager.compile(self.fractal, self.settings)

    def set_engine(self, engine: BaseRenderEngine):
        self.engine = engine

    def render(self, vp: Viewport) -> np.ndarray:
        return self.engine.render(self.fractal,
                                  self.manager,
                                  self.settings,
                                  vp,
                                  backend=self.default_backend,
                                  device=self.default_device)
