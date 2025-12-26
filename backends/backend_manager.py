from typing import NamedTuple, Type

from backend_base import Backend
from backends.cpu_backend import CpuBackend
from backends.cuda_backend import CudaBackend
from backends.opencl_backend import OpenClBackend
from fractals.fractal_base import Fractal, RenderSettings, Viewport


class RenderHandle:
    def __init__(self, result, wait_fn):
        self._result = result
        self._wait_fn = wait_fn
        self._waited = False

    def wait(self):
        if not self._waited:
            self._wait_fn()
            self._waited = True
        return self._result

    @property
    def result(self):
        return self.wait()


class BackendSpec(NamedTuple):
    cls: Type
    priority: int
    supports_devices: bool


BACKENDS = {
    "CPU": BackendSpec(
        cls=CpuBackend,
        priority=0,
        supports_devices=False
    ),
    "CUDA": BackendSpec(
        cls=CudaBackend,
        priority=10,
        supports_devices=True
    ),
    "OPENCL": BackendSpec(
        cls=OpenClBackend,
        priority=8,
        supports_devices=True),
}


class BackendManager:

    def __init__(self,
                 preferred: list[str] = ("CUDA", "OPENCL", "CPU"),
                 allow_fallback: bool = True):
        self.preferred = preferred
        self.allow_fallback = allow_fallback

        # (backend_name, device_id) -> backend instance
        self._backends = {}

        self._compiled = False
        self._compile_args = None

    # --------------- Backend creation ---------------------

    def _get_backend(self, name: str, device: int) -> Backend:
        key = (name, device)
        if key in self._backends:
            return self._backends[key]

        spec = BACKENDS[name]

        if spec.supports_devices:
            backend = spec.cls(device=device)
        else:
            backend = spec.cls()

        if self._compiled:
            fractal, settings = self._compile_args
            backend.compile(fractal, settings)

        self._backends[key] = backend
        return backend

    def _select_backend(self, backend: str | None, device: int | None) -> Backend:

        if backend is not None:
            candidates = [backend]
        else:
            candidates = list(self.preferred)

        last_error = None

        for name in candidates:
            if name not in BACKENDS:
                continue
            spec = BACKENDS[name]
            try:
                dev = device if spec.supports_devices else None
                return self._get_backend(name, dev)

            except Exception as e:
                last_error = e
                if not self.allow_fallback:
                    raise

        raise RuntimeError(f"No suitable backend found. Last error: {last_error}")

    # --------------- Public API ---------------------

    def compile(self, fractal: Fractal, settings: RenderSettings) -> None:
        self._compiled = True
        self._compile_args = (fractal, settings)

        for backend in self._backends.values():
            backend.compile(fractal, settings)

    def render(
            self,
            fractal: Fractal,
            vp: Viewport,
            settings: RenderSettings,
            backend: str = None,
            device: int = 0
    ):
        b = self._select_backend(backend, device)
        return b.render(fractal, vp, settings)

    def render_async(
            self,
            fractal: Fractal,
            vp: Viewport,
            settings: RenderSettings,
            backend: str = None,
            device: int = 0
    ):
        b = self._select_backend(backend, device)

        if hasattr(b, "render_async"):
            # Async rendering
            result, evt = b.render_async(fractal, vp, settings)

            if hasattr(evt, "wait"):
                wait_fn = evt.wait
            elif hasattr(evt, "synchronize"):
                wait_fn = evt.synchronize
            else:
                wait_fn = lambda: None

            return RenderHandle(result, wait_fn)

        # Synchronous rendering
        result = b.render(fractal, vp, settings)
        return RenderHandle(result, lambda: None)

    def close(self):
        for b in self._backends.values():
            try:
                b.close()
            except Exception:
                pass
        self._backends.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
