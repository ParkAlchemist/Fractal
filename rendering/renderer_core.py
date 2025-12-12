import numpy as np
from typing import Optional

from fractals.fractal_base import Fractal, Viewport, RenderSettings
from backends.backend_base import Backend
from rendering.render_engines import BaseRenderEngine, FullFrameEngine

class Renderer:
    """
    Main construct for combining the rendering pipeline: (engine + fractal + backend)
    """
    def __init__(self, fractal: Fractal, backend: Backend, settings: RenderSettings,
                 engine: Optional[BaseRenderEngine] = None):
        self.fractal = fractal
        self.backend = backend
        self.settings = settings
        self.engine = engine or FullFrameEngine()
        self.backend.compile(self.fractal, self.settings)

    def set_engine(self, engine: BaseRenderEngine):
        self.engine = engine

    def render(self, vp: Viewport) -> np.ndarray:
        return self.engine.render(self.fractal, self.backend, self.settings, vp)
