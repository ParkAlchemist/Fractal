import numpy as np
from typing import Optional

from fractals.fractal_base import Fractal, Viewport, RenderSettings
from backends.backend_base import Backend
from renderers.render_engines import BaseRenderEngine, FullFrameEngine

class Renderer:
    def __init__(self, fractal: Fractal, backend: Backend, settings: RenderSettings,
                 engine: Optional[BaseRenderEngine] = None):
        self.fractal = fractal
        self.backend = backend
        self.settings = settings
        self.engine = engine or FullFrameEngine()

    def set_engine(self, engine: BaseRenderEngine):
        self.engine = engine

    def render(self, vp: Viewport) -> np.ndarray:
        reference = None
        build_ref = self.settings.use_perturb and self.fractal.supports_perturbation()
        if build_ref and getattr(self.engine, "per_tile_reference", False):
            reference = None
        elif build_ref:
            reference = self.fractal.build_reference(vp, self.settings)

        self.backend.compile(self.fractal, self.settings)
        return self.engine.render(self.fractal, self.backend, self.settings, vp, reference)
