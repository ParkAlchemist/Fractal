from __future__ import annotations

import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Optional, List, Callable, Tuple

import numpy as np

from fractals.base import Fractal, Viewport, RenderSettings

from devices.manager import DeviceManager
from devices.types import DeviceInfo
from backend.pool import BackendPool


# ---- Async handles (compatible with your existing engines) --------------

class RenderHandle:
    """Wraps an async render (array + wait_fn)."""
    def __init__(self, result, wait_fn: Callable[[], None]):
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
    """Aggregates several RenderHandles and assembles the final canvas on wait()."""
    def __init__(self, handles: List[RenderHandle], assembler: Callable[[List[np.ndarray]], np.ndarray]):
        self._handles = handles
        self._assembler = assembler
        self._waited = False
        self._result: Optional[np.ndarray] = None

    def wait(self):
        if not self._waited:
            parts = [h.wait() for h in self._handles]
            self._result = self._assembler(parts)
            self._waited = True
        return self._result

    @property
    def result(self):
        return self.wait()


class CancelToken:
    def __init__(self) -> None:
        import threading
        self._flag = threading.Event()
    def cancel(self) -> None:
        self._flag.set()
    def is_cancelled(self) -> bool:
        return self._flag.is_set()


# ---- Executor -----------------------------------------------------------

class RenderExecutor:
    """
    Execution facade built on:
      - DeviceManager: discovery/selection/scoring.
      - BackendPool: backend instance lifecycle and compilation.
    """

    def __init__(
        self,
        *,
        devices: Optional[DeviceManager] = None,
        pool: Optional[BackendPool] = None,
        selection_policy: str = "AUTO",
        telemetry: Optional[Callable[[str], None]] = None,
    ) -> None:
        self.devices = devices or DeviceManager()
        self.pool = pool or BackendPool()
        self.selection_policy = selection_policy
        self.log = telemetry or (lambda *_: None)

    # ---- Lifecycle ------------------------------------------------------

    def compile(self, fractal: Fractal, settings: RenderSettings) -> None:
        self.pool.compile_all(fractal, settings)

    def close(self) -> None:
        self.pool.close_all()

    # ---- Device APIs ----------------------------------------------------

    def list_devices(self, backend: Optional[str] = None) -> List[DeviceInfo]:
        return self.devices.list(backend)

    def choose_device(
        self,
        backend: Optional[str] = None,
        device: Optional[int] = None,
        policy: Optional[str] = None,
        require: Optional[dict] = None,
    ) -> DeviceInfo:
        return self.devices.choose(backend=backend, device=device, policy=policy or self.selection_policy, require=require)

    # ---- Single render --------------------------------------------------

    def _choose_backend_and_device(self, backend: Optional[str], device: Optional[int]) -> Tuple[str, Optional[int]]:
        if backend is None and device is None:
            di = self.choose_device(policy=self.selection_policy)
            return di.backend, di.device_id
        if backend is not None and device is None:
            di = self.choose_device(backend=backend, policy=self.selection_policy)
            return di.backend, di.device_id
        return backend, device

    def render(
        self,
        fractal: Fractal,
        vp: Viewport,
        settings: RenderSettings,
        backend: Optional[str] = None,
        device: Optional[int] = None,
    ) -> np.ndarray:
        name, dev = self._choose_backend_and_device(backend, device)
        be = self.pool.get(name, dev)
        return be.render(fractal, vp, settings)

    def render_async(
        self,
        fractal: Fractal,
        vp: Viewport,
        settings: RenderSettings,
        backend: Optional[str] = None,
        device: Optional[int] = None,
    ) -> RenderHandle:
        name, dev = self._choose_backend_and_device(backend, device)
        be = self.pool.get(name, dev)
        if hasattr(be, "render_async"):
            result, evt = be.render_async(fractal, vp, settings)
            # Normalize wait function across CUDA/OpenCL
            if hasattr(evt, "wait"):
                wait_fn = evt.wait
            elif hasattr(evt, "synchronize"):
                wait_fn = evt.synchronize
            else:
                wait_fn = lambda: None
            return RenderHandle(result, wait_fn)
        # Fallback to sync
        result = be.render(fractal, vp, settings)
        return RenderHandle(result, lambda: None)

    # ---- Multi-device (stripe split) ------------------------------------

    @staticmethod
    def _split_viewport_stripes(vp: Viewport, parts: int, axis: str = "y") -> List[Tuple[Viewport, slice]]:
        """
        Stripe splitting
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
        split_axis: str = "y",
    ) -> np.ndarray:
        # Choose backend and the list of devices to use
        if backend is None:
            chosen = self.choose_device()
            backend = chosen.backend
        devs = [d for d in self.devices.list(backend) if d.is_available]
        if not devs:
            raise RuntimeError(f"No devices found for backend {backend}.")
        if devices is not None:
            ids = set(devices)
            devs = [d for d in devs if d.device_id in ids]
            if not devs:
                raise RuntimeError(f"No matching device ids {devices} for backend {backend}.")

        parts = len(devs)
        subs = self._split_viewport_stripes(vp, parts, axis=split_axis)

        results: List[Optional[np.ndarray]] = [None] * len(subs)
        t0 = time.perf_counter()

        # Run in a small thread pool; each worker is a blocking call or waits on its own event
        with ThreadPoolExecutor(max_workers=len(subs)) as ex:
            futs = []
            for i, ((svp, _slc), di) in enumerate(zip(subs, devs)):
                be = self.pool.get(di.backend, di.device_id)
                def run(idx: int, backend_obj, subvp: Viewport):
                    if hasattr(backend_obj, "render_async"):
                        res, evt = backend_obj.render_async(fractal, subvp, settings)
                        wait_fn = getattr(evt, "wait", getattr(evt, "synchronize", lambda: None))
                        wait_fn()
                        return idx, res
                    # sync fallback
                    res = backend_obj.render(fractal, subvp, settings)
                    return idx, res
                futs.append(ex.submit(run, i, be, svp))

            for fut in as_completed(futs):
                idx, res = fut.result()
                results[idx] = res

        if any(r is None for r in results):
            raise RuntimeError("[RenderExecutor] render_multi failed to obtain all parts.")

        elapsed = (time.perf_counter() - t0) * 1000.0
        self.log(f"[RenderExecutor] render_multi across {len(subs)} parts on {backend} in {elapsed:.2f} ms")

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
        if backend is None:
            chosen = self.choose_device()
            backend = chosen.backend
        devs = [d for d in self.devices.list(backend) if d.is_available]
        if not devs:
            raise RuntimeError(f"No devices found for backend {backend}.")
        if devices is not None:
            ids = set(devices)
            devs = [d for d in devs if d.device_id in ids]
            if not devs:
                raise RuntimeError(f"No matching device ids {devices} for backend {backend}.")

        parts = len(devs)
        subs = self._split_viewport_stripes(vp, parts, axis=split_axis)

        handles: List[RenderHandle] = []
        for (svp, _slc), di in zip(subs, devs):
            be = self.pool.get(di.backend, di.device_id)
            if hasattr(be, "render_async"):
                res, evt = be.render_async(fractal, svp, settings)
                wait_fn = getattr(evt, "wait", getattr(evt, "synchronize", lambda: None))
                handles.append(RenderHandle(res, wait_fn))
            else:
                # Wrap sync as completed handle
                res = be.render(fractal, svp, settings)
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
