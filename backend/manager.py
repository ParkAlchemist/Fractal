from __future__ import annotations

import time
from dataclasses import dataclass, field
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import NamedTuple, Type, Optional, Dict, List, Tuple, Callable

import numpy as np

from backend.model.base import Backend
from backend.model.cpu import CpuBackend
from backend.model.cuda import CudaBackend
from backend.model.opencl import OpenClBackend

from fractals.base import Fractal, RenderSettings, Viewport


# --------------------------------------------------------------------------
# Device and Backend metadata
# --------------------------------------------------------------------------

@dataclass(frozen=True)
class DeviceInfo:
    """
    Describes a compute device exposed by a backend.
    """
    backend: str
    device_id: Optional[int]
    name: str
    vendor: Optional[str] = None
    driver: Optional[str] = None
    compute_capability: Optional[str] = None
    memory_total_mb: Optional[int] = None
    memory_free_mb: Optional[int] = None
    score: float = 0.0
    is_available: bool = True
    extra: Dict[str, object] = field(default_factory=dict)

class RenderHandle:
    """
    Wrapper for asynchronous rendering results, providing a wait method to block until the result is available.
    """
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

class MultiRenderHandle:
    """
    Aggregates several RenderHandle objects and an aggregator function.
    """
    def __init__(self, handles: List[RenderHandle], agg_fn: Callable[[List], np.ndarray]):
        self._handles = handles
        self._agg_fn = agg_fn
        self._waited = False
        self._result = None

    def wait(self):
        if not self._waited:
            parts = [h.wait() for h in self._handles]
            self._result = self._agg_fn(parts)
            self._waited = True
        return self._result

    @property
    def result(self):
        return self.wait()


class BackendSpec(NamedTuple):
    """
    Specification for a backend, including its class, priority, and device support.
    """
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
        supports_devices=True
    ),
}

#---------------------------------------------------------------------
# BackendManager: device management + (multi-)render orchestration
# --------------------------------------------------------------------

class BackendManager:
    """
    Central authority for:
        - Device discovery and selection policy
        - Backend caching and compilation
        - Single- and multi-device rendering orchestration
        - Fallbacks and lifecycle
    """
    def __init__(self,
                 preferred: list[str] = BACKENDS.keys(),
                 allow_fallback: bool = True,
                 memory_budget_fraction: float = 0.90,
                 selection_policy: str = "AUTO",    # "AUTO" | "THROUGHPUT" | "LATENCY" | "MEMORY"
                 telemetry: Optional[Callable[[str], None]] = None):
        self.preferred = list(preferred)
        self.allow_fallback = bool(allow_fallback)
        self.memory_budget_fraction = float(memory_budget_fraction)
        self.selection_policy = selection_policy
        self.log = telemetry or (lambda msg: None)

        # Cache: (backend_name, device_id) -> Backend instance
        self._backends: Dict[Tuple[str, Optional[int]], Backend] = {}

        # Global compile state (fractal, settings) to apply on creation or on refresh
        self._compiled = False
        self._compile_args: Optional[Tuple[Fractal, RenderSettings]] = None

        # Device inventory
        self._devices: List[DeviceInfo] = []
        self.refresh_devices()  # Initial probe

    # --------------- Discovery ----------------------------
    def refresh_devices(self) -> None:
        """
        Probes all known backends for devices.
        """
        devices: List[DeviceInfo] = []
        for name in self.preferred:
            if name not in BACKENDS:
                continue
            spec = BACKENDS[name]
            try:
                if spec.supports_devices and hasattr(spec.cls, "enumerate_devices"):
                    raw = spec.cls.enumerate_devices()
                    for r in raw:
                        if isinstance(r, DeviceInfo):
                            di = r
                        else:
                            di = DeviceInfo(backend=name, **r)
                        devices.append(di)
                else:
                    devices.append(DeviceInfo(backend="CPU", device_id=None, name="CPU"))
            except Exception as e:
                self.log(f"[BackendManager] Device probe failed for {name}: {e}")
        self._devices = self._score_devices(devices, self.selection_policy)

    def list_devices(self, backend: Optional[str] = None) -> List[DeviceInfo]:
        if backend:
            return [d for d in self._devices if d.backend.upper() == backend.upper()]
        return list(self._devices)

    # --------------- Selection ----------------------------
    @staticmethod
    def _score_devices(devices: List[DeviceInfo], policy: str) -> List[DeviceInfo]:
        def base_score(d: DeviceInfo) -> float:
            # Priority by backend kind
            prio = BACKENDS.get(d.backend.upper(), BackendSpec(Backend, -1, False)).priority
            mem = (d.memory_total_mb or 0) / 1024.0
            cap = 0.0
            try:
                cap = float(d.compute_capability) if d.compute_capability else 0.0
            except Exception:
                pass
            return prio * 10.0 + mem + cap

        def throughput_score(d: DeviceInfo) -> float:
            return base_score(d)

        def latency_score(d: DeviceInfo) -> float:
            if d.backend.upper() == "CPU":
                return base_score(d) + 1.0
            return base_score(d) - 0.1

        def memory_score(d: DeviceInfo) -> float:
            return d.memory_total_mb or 0

        scored: List[DeviceInfo] = []
        for d in devices:
            if not d.is_available:
                continue
            if policy == "THROUGHPUT":
                s = throughput_score(d)
            elif policy == "LATENCY":
                s = latency_score(d)
            elif policy == "MEMORY":
                s = memory_score(d)
            else:
                s = base_score(d)
            scored.append(DeviceInfo(**{**d.__dict__, "score": s}))
        scored.sort(key=lambda x: x.score, reverse=True)
        return scored

    def choose_device(self,
                      backend: Optional[str] = None,
                      device: Optional[int] = None,
                      policy: Optional[str] = None,
                      require: Optional[Dict[str, object]] = None
                      ) -> DeviceInfo:
        """
        Returns the DeviceInfo to use given hints and constraints.
        """
        policy = policy or self.selection_policy
        if backend is not None:
            devs = [d for d in self._devices if d.backend.upper() == backend.upper()]
        else:
            devs = list(self._devices)

        # Filter by requirements
        require = require or {}
        for key, val in require.items():
            devs = [d for d in devs if getattr(d, key, None) == val]

        if device is not None:
            for d in devs:
                if d.device_id == device:
                    return d
            raise RuntimeError(f"No device {device} for backend {backend or '*'}")

        if not devs:
            raise RuntimeError(f"No devices available for the requested criteria.")

        if policy != self.selection_policy:
            devs = self._score_devices(devs, policy)

        return devs[0]

    # --------------- Backend caching / compilation ---------------------
    def _get_backend(self, name: str, device: Optional[int] = None) -> Backend:
        """
        Retrieves a backend instance for the given name and device.
        """
        key = (name.upper(), device if BACKENDS[name.upper()].supports_devices else None)
        if key in self._backends:
            return self._backends[key]
        spec = BACKENDS[name]
        b = spec.cls(device=device) if spec.supports_devices else spec.cls()
        if self._compiled and self._compile_args:
            fractal, settings = self._compile_args
            b.compile(fractal, settings)
        self._backends[key] = b
        return b

    def compile(self, fractal: Fractal, settings: RenderSettings) -> None:
        """
        Marks the current (fractal, settings) as compiled and applies to cached backend.
        Lazy compiles occur on first use per backend instance.
        """
        self._compiled = True
        self._compile_args = (fractal, settings)

        for be in self._backends.values():
            try:
                be.compile(fractal, settings)
            except Exception as e:
                self.log(f"[BackendManager] Compilation failed for cached backend {be.name}: {e}")

    # -------------------- Single device rendering --------------------
    def _select_backend(self, backend: Optional[str], device: Optional[int]) -> Tuple[str, Optional[int]]:
        """
        Selects and returns a backend instance based on the provided backend and device.
        """
        if backend is not None:
            di = self.choose_device(backend=backend, device=device)
            return di.backend, di.device_id

        last_error = None
        for name in self.preferred:
            try:
                di = self.choose_device(backend=name)
                return di.backend, di.device_id
            except Exception as e:
                last_error = e
                if not self.allow_fallback:
                    raise
        raise RuntimeError(f"No suitable backend found. Last error: {last_error}")

    def render(
            self,
            fractal: Fractal,
            vp: Viewport,
            settings: RenderSettings,
            backend: Optional[str] = None,
            device: Optional[int] = None
    ) -> np.ndarray:
        """
        Renders the fractal with the specified settings using the selected backend and device.
        """
        name, dev = self._select_backend(backend, device)
        b = self._get_backend(name, dev)
        return b.render(fractal, vp, settings)

    def render_async(
            self,
            fractal: Fractal,
            vp: Viewport,
            settings: RenderSettings,
            backend: Optional[str] = None,
            device: Optional[int] = None
    ) -> RenderHandle:
        """
        Renders the fractal asynchronously with the specified settings using the selected backend and device.
        """
        name, dev = self._select_backend(backend, device)
        b = self._get_backend(name, dev)

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

    # ---------------- -- Multi-device rendering ----------------------
    @staticmethod
    def _split_viewport_stripes(vp: Viewport, parts: int, axis: str = "y") -> List[Tuple[Viewport, slice]]:
        """
        Splits a viewport into 'parts' equally (last gets the remainder).
        Returns sub-viewports and target slices for assembling.
        """
        subs: List[Tuple[Viewport, slice]] = []
        if axis.lower().startswith("y"):
            base = vp.height // parts
            off = 0
            for i in range(parts):
                h = base if i < parts - 1 else (vp.height - base * (parts - 1))
                if h <= 0:
                    break
                sy, ey = off / vp.height, (off + h) / vp.height
                sub_min_y = vp.min_y + (vp.max_y - vp.min_y) * sy
                sub_max_y = vp.min_y + (vp.max_y - vp.min_y) * ey
                svp = Viewport(vp.min_x, vp.max_x, sub_min_y, sub_max_y, vp.width, h)
                subs.append((svp, slice(off, off + h)))
                off += h
        else:
            base = vp.width // parts
            off = 0
            for i in range(parts):
                w = base if i < parts - 1 else (vp.width - base * (parts - 1))
                if w <= 0:
                    break
                sx, ex = off / vp.width, (off + w) / vp.width
                sub_min_x = vp.min_x + (vp.max_x - vp.min_x) * sx
                sub_max_x = vp.min_x + (vp.max_x - vp.min_x) * ex
                svp = Viewport(sub_min_x, sub_max_x, vp.min_y, vp.max_y, w, vp.height)
                subs.append((svp, slice(off, off + w)))
                off += w
        return subs

    def render_multi(
            self,
            fractal: Fractal,
            vp: Viewport,
            settings: RenderSettings,
            backend: Optional[str] = None,
            devices: Optional[List[int]] = None,
            split_axis: str = "y"
    ) -> np.ndarray:
        """
        Renders the viewport concurrently across multiple devices of a single backend.
        If the backend is None, this function chooses the best backend (AUTO) and its top-N-devices.
        """
        if backend is None:
            chosen = self.choose_device()
            backend = chosen.backend
        devs = [d for d in self._devices if d.backend.upper() == backend.upper()]

        if not devs:
            raise RuntimeError(f"No devices found for backend {backend}.")

        if devices is not None:
            devs = [d for d in devs if d.device_id in set(devices)]
            if not devs:
                raise RuntimeError(f"No matching device ids {devices} for backend {backend}.")

        # Split viewport into stripes
        parts = len(devs)
        subs = self._split_viewport_stripes(vp, parts, axis=split_axis)

        # Dispatch per-device (prefer backend async if available)
        results: List[np.ndarray] = [] * len(subs)
        t0 = time.perf_counter()

        with ThreadPoolExecutor(max_workers=len(subs)) as ex:
            futs = []
            for i, ((svp, slc), di) in enumerate(zip(subs, devs)):
                name, dev = di.backend, di.device_id
                b = self._get_backend(name, dev)
                if hasattr(b, "render_async"):
                    def submit_async(idx: int, backend_obj: Backend, subvp: Viewport):
                        res, evt = backend_obj.render_async(fractal, subvp, settings)
                        wait_fn = getattr(evt, "wait", getattr(evt, "synchronize", lambda: None))
                        wait_fn()
                        return idx, res
                    futs.append(ex.submit(submit_async, i, b, svp))
                else:
                    def submit_sync(idx: int, backend_obj: Backend, subvp: Viewport):
                        res = backend_obj.render(fractal, subvp, settings)
                        return idx, res
                    futs.append(ex.submit(submit_sync, i, b, svp))

            for fut in as_completed(futs):
                idx, res = fut.result()
                results[idx] = res

        elapsed = (time.perf_counter() - t0) * 1000.0
        self.log(f"[BackendManager] render_multi across {len(subs)} parts on {backend} in {elapsed:.2f} ms")

        # Assemble strips
        canvas = np.zeros((vp.height, vp.width), dtype=settings.precision)
        if split_axis.lower().startswith("y"):
            off = 0
            for part in results:
                h = part.shape[0]
                canvas[off:off + h, :] = part
                off += h
        else:
            off = 0
            for part in results:
                w = part.shape[1]
                canvas[:, off:off + w] = part
                off += w
        return canvas

    def render_multi_async(
            self,
            fractal: Fractal,
            vp: Viewport,
            settings: RenderSettings,
            backend: Optional[str] = None,
            devices: Optional[List[int]] = None,
            split_axis: str = "y",
    ) -> MultiRenderHandle:
        """
        Async multi-device render.
        Returns a MultiRenderHandle; .wait() assembles the final canvas.
        """
        # The async variant issues per-device RenderHandle then assembles.
        if backend is None:
            chosen = self.choose_device()
            backend = chosen.backend
            devs = [d for d in self._devices if d.backend.upper() == backend.upper()]
        else:
            devs = [d for d in self._devices if d.backend.upper() == backend.upper()]

        if not devs:
            raise RuntimeError(f"No devices found for backend {backend}.")

        if devices is not None:
            devs = [d for d in devs if d.device_id in set(devices)]
            if not devs:
                raise RuntimeError(f"No matching device ids {devices} for backend {backend}.")

        parts = len(devs)
        subs = self._split_viewport_stripes(vp, parts, axis=split_axis)

        handles: List[RenderHandle] = []
        for (svp, _), di in zip(subs, devs):
            name, dev = di.backend, di.device_id
            b = self._get_backend(name, dev)
            if hasattr(b, "render_async"):
                res, evt = b.render_async(fractal, svp, settings)
                wait_fn = getattr(evt, "wait", getattr(evt, "synchronize", lambda: None))
                handles.append(RenderHandle(res, wait_fn))
            else:
                # Wrap a sync call as an already-complete handle
                res = b.render(fractal, svp, settings)
                handles.append(RenderHandle(res, lambda: None))

        def assemble(parts_res: List[np.ndarray]) -> np.ndarray:
            canvas = np.zeros((vp.height, vp.width), dtype=settings.precision)
            if split_axis.lower().startswith("y"):
                off = 0
                for part in parts_res:
                    h = part.shape[0]
                    canvas[off:off + h, :] = part
                    off += h
            else:
                off = 0
                for part in parts_res:
                    w = part.shape[1]
                    canvas[:, off:off + w] = part
                    off += w
            return canvas

        return MultiRenderHandle(handles, assemble)

    # ---------------- Lifecycle ------------------------
    def close(self):
        """
        Closes all backend instances and clears the backend cache.
        """
        for b in list(self._backends.values()):
            try:
                b.close()
            except Exception:
                pass
        self._backends.clear()

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.close()
