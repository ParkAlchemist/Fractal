from typing import Optional, Dict

from rendering.service import RenderService
from utils.enums import EngineMode, BackendType, PrecisionMode


class RenderConfigBuilder:
    """
    Builder for configuring render settings.
    """
    def __init__(self, service: RenderService):
        self.service = service
        self._resolution: Optional[str] = None
        self._max_iter: Optional[int] = None
        self._samples: Optional[int] = None
        self._engine_mode: Optional[EngineMode] = None
        self._tile_w: Optional[int] = None
        self._tile_h: Optional[int] = None
        self._backend: Optional[BackendType] = None
        self._adaptive_opts: Optional[Dict] = None

    def resolution(self, preset: str) -> 'RenderConfigBuilder':
        self._resolution = preset
        return self

    def max_iter(self, value: int) -> 'RenderConfigBuilder':
        self._max_iter = value
        return self

    def samples(self, value: int) -> 'RenderConfigBuilder':
        self._samples = value
        return self

    def engine_mode(self, mode: EngineMode, tile_w: int = None, tile_h: int = None) -> 'RenderConfigBuilder':
        self._engine_mode = mode
        self._tile_w = tile_w
        self._tile_h = tile_h
        return self

    def backend(self, backend: BackendType) -> 'RenderConfigBuilder':
        self._backend = backend
        return self

    def adaptive_opts(self, opts: dict) -> 'RenderConfigBuilder':
        self._adaptive_opts = opts
        return self

    def apply(self):
        # Apply settings to the service
        if self._resolution:
            w, h = self._compute_size(self._resolution)
            self.service.set_image_size(w, h)
        if self._max_iter:
            self.service.set_max_iter(self._max_iter)
        if self._samples:
            self.service.set_samples(self._samples)
        if self._engine_mode:
            self.service.set_engine_mode(self._engine_mode, self._adaptive_opts, self._tile_w, self._tile_h)
        if self._backend:
            #self.renderer.set_kernel(self._backend)
            pass

    @staticmethod
    def _compute_size(preset: str) -> tuple[int, int]:
        mapping = {
            "2160p": 3840,
            "1440p": 2560,
            "1080p": 1920,
            "720p": 1280,
            "480p": 854,
            "360p": 640,
        }
        h = int(preset.replace("p", ""))
        w = mapping.get(preset, 1920)
        return w, h


class RenderAPI:
    """
    Facade for controlling rendering operations and managing callbacks.
    """
    def __init__(self, service: RenderService):
        self.service: RenderService = service

    # ---------- Callbacks --------------------------------
    def on_frame(self, cb): self.service.on_frame = cb
    def on_tile(self, cb): self.service.on_tile = cb
    def on_log(self, cb): self.service.on_log = cb

    # ----------- Facade methods --------------------------
    def set_view(self, center_x: float, center_y: float, zoom: float) -> None:
        """
        Sets the view of the renderer to the specified coordinates and zoom level.

        Args:
            center_x (float): The x-coordinate of the center of the view.
            center_y (float): The y-coordinate of the center of the view.
            zoom (float): The zoom level to set.
        """
        self.service.set_view(center_x, center_y, zoom)

    def set_palettes(self, exterior, interior) -> None:
        """
        Sets the palette for the renderer.

        Args:
            exterior: The exterior palette configuration.
            interior: The interior palette configuration.
        """
        self.service.set_palettes(exterior, interior)

    def set_precision(self, precision: PrecisionMode) -> None:
        """
        Sets the precision mode for the renderer.

        Args:
            precision (PrecisionMode): The precision mode to set.
        """
        self.service.set_precision(precision)

    def set_image_size(self, rw: int, rh: int) -> None:
        """
        Sets the image size for rendering.

        Args:
            rw (int): The width of the render image.
            rh (int): The height of the render image.
        """
        self.service.set_image_size(rw, rh)

    def configure(self) -> RenderConfigBuilder:
        """
        Configures the renderer with a fluent builder pattern.

        Returns:
            RenderConfigBuilder: A builder object for configuring renderer settings.
        """
        return RenderConfigBuilder(self.service)

    def start_async_render(self):
        """
        Initiates asynchronous rendering by delegating to the renderer.
        """
        self.service.start_render()

    def stop_render(self) -> None:
        """
        Stops the ongoing rendering process.
        """
        self.service.stop()
