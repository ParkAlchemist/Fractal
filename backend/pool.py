from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple, Optional, Type

from backend.model.be_base import Backend
from backend.model.be_cpu import CpuBackend
from backend.model.be_cuda import CudaBackend
from backend.model.be_opencl import OpenClBackend
from fractals.base import Fractal, RenderSettings


# Small descriptor of a backend implementation
@dataclass(frozen=True)
class BackendSpec:
    cls: Type[Backend]
    priority: int
    supports_devices: bool


# Default registry
DEFAULT_BACKENDS: Dict[str, BackendSpec] = {
    "CPU":    BackendSpec(cls=CpuBackend,   priority=0,  supports_devices=False),
    "CUDA":   BackendSpec(cls=CudaBackend,  priority=10, supports_devices=True),
    "OPENCL": BackendSpec(cls=OpenClBackend, priority=8, supports_devices=True),
}


class BackendPool:
    """
    Creates, caches and compiles backend instances.
    Keyed by (backend_name, device_id_or_None).
    """

    def __init__(self, registry: Optional[Dict[str, BackendSpec]] = None, telemetry=None) -> None:
        self.registry: Dict[str, BackendSpec] = registry or DEFAULT_BACKENDS
        self._cache: Dict[Tuple[str, Optional[int]], Backend] = {}
        self._compiled_args: Optional[Tuple[Fractal, RenderSettings]] = None  # (fractal, settings)
        self._compiled: bool = False
        self.log = telemetry or (lambda *_: None)

    def get(self, name: str, device: Optional[int] = None) -> Backend:
        key = (name.upper(), device if self.registry[name.upper()].supports_devices else None)
        if key in self._cache:
            return self._cache[key]
        spec = self.registry[name.upper()]
        be = spec.cls(device=device) if spec.supports_devices else spec.cls()
        # If we already compiled for (fractal, settings), apply lazily to new instances
        if self._compiled and self._compiled_args is not None:
            fractal, settings = self._compiled_args
            be.compile(fractal, settings)
        self._cache[key] = be
        return be

    def compile_all(self, fractal, settings) -> None:
        """
        Record the current (fractal, settings), and eager-compile any cached instances.
        Newly created instances will be lazily compiled when first retrieved.
        """
        self._compiled = True
        self._compiled_args = (fractal, settings)
        for be in list(self._cache.values()):
            try:
                be.compile(fractal, settings)
            except Exception as e:
                self.log(f"[BackendPool] compile failed for {getattr(be, 'name', be)}: {e}")

    def close_all(self) -> None:
        for be in list(self._cache.values()):
            try:
                be.close()
            except Exception:
                pass
        self._cache.clear()
        self._compiled = False
        self._compiled_args = None
