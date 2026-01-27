from __future__ import annotations
import numpy as np
from typing import Optional

from fractals.base import Fractal, Viewport, RenderSettings
from rendering.engines.full_frame import FullFrameEngine
from rendering.engines.base import BaseRenderEngine
from rendering.executor import RenderExecutor


class Renderer:

    """
    Facade that binds together:
      - the fractal + render settings,
      - the render engine (strategy),
      - render executor
    """

    def __init__(
        self,
        fractal: Fractal,
        settings: RenderSettings,
        *,
        engine: Optional[BaseRenderEngine] = None,
        executor: Optional[RenderExecutor] = None,
        default_backend: Optional[str] = None,
        default_device: Optional[int] = None,
    ):
        # Core state
        self.fractal = fractal
        self.settings = settings

        # Strategy
        self.engine = engine or FullFrameEngine()

        # Execution & resource ownership
        self.executor = executor or RenderExecutor()

        # Hints
        self.default_backend = default_backend
        self.default_device = default_device

        # Precompile
        self.executor.compile(self.fractal, self.settings)

    # ----------------------------
    # Mutators / helpers
    # ----------------------------

    def set_engine(self, engine: BaseRenderEngine) -> None:
        """Swap rendering strategy."""
        self.engine = engine

    def set_backend_hints(self, backend: Optional[str] = None, device: Optional[int] = None) -> None:
        """Set default backend / device hints for future renders."""
        self.default_backend = backend
        self.default_device = device

    def recompile(self, fractal: Optional[Fractal] = None, settings: Optional[RenderSettings] = None) -> None:
        """Recompile the fractal and/or settings."""
        if fractal is not None:
            self.fractal = fractal
        if settings is not None:
            self.settings = settings
        self.executor.compile(self.fractal, self.settings)

    def close(self) -> None:
        self.executor.close()

    # ----------------------------
    # Render entry point
    # ----------------------------

    def render(self, vp: Viewport) -> np.ndarray:
        """
        Delegate to the engine.
        """
        return self.engine.render(self.fractal,
                                  self.executor,
                                  self.settings,
                                  vp,
                                  backend=self.default_backend,
                                  device=self.default_device)
