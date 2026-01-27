from __future__ import annotations
from typing import List, Optional, Dict, Tuple
from devices.types import DeviceInfo
from devices.providers.prov_cpu import CpuDeviceProvider
from devices.providers.prov_cuda import CudaDeviceProvider
from devices.providers.prov_opencl import OpenClDeviceProvider


BACKEND_PRIORITY = {"CPU": 0, "OPENCL": 8, "CUDA": 10}
PROVIDER = CudaDeviceProvider | OpenClDeviceProvider | CpuDeviceProvider

class DeviceManager:
    """
    Enumerates, scores, and selects devices via pluggable providers.
    """
    def __init__(
        self,
        providers: Optional[List[PROVIDER]] = None,
        selection_policy: str = "AUTO",
        perf_cache: Optional[Dict[Tuple[str, Optional[int]], float]] = None,
    ) -> None:
        self.providers = providers or [CudaDeviceProvider(), OpenClDeviceProvider(), CpuDeviceProvider()]
        self.selection_policy = selection_policy
        self.perf_cache = perf_cache or {}
        self._devices: List[DeviceInfo] = []
        self.refresh()

    # ---- Discovery ------------------------------------------------------

    def refresh(self) -> None:
        devices: List[DeviceInfo] = []
        for p in self.providers:
            try:
                for raw in p.enumerate():
                    if isinstance(raw, DeviceInfo):
                        di = raw
                    else:
                        di = DeviceInfo(backend=p.backend, **raw)
                    devices.append(di)
            except Exception:
                # provider failure => skip
                pass
        self._devices = self._score_devices(devices, self.selection_policy)

    def list(self, backend: Optional[str] = None) -> List[DeviceInfo]:
        if backend:
            be = backend.upper()
            return [d for d in self._devices if d.backend.upper() == be]
        return list(self._devices)

    # ---- Selection ------------------------------------------------------

    def choose(
        self,
        backend: Optional[str] = None,
        device: Optional[int] = None,
        policy: Optional[str] = None,
        require: Optional[Dict[str, object]] = None,
    ) -> DeviceInfo:
        policy = policy or self.selection_policy
        cand = self.list(backend) if backend else list(self._devices)

        if require:
            for k, v in require.items():
                cand = [d for d in cand if getattr(d, k, None) == v]

        if device is not None:
            for d in cand:
                if d.device_id == device:
                    return d
            raise RuntimeError(f"No device {device} for backend {backend or '*'}")

        if not cand:
            raise RuntimeError("No devices available for requested criteria")

        if policy != self.selection_policy:
            cand = self._score_devices(cand, policy)

        return cand[0]

    # ---- Scoring --------------------------------------------------------

    @staticmethod
    def _normalize(xs: List[float]) -> List[float]:
        if not xs: return []
        lo, hi = min(xs), max(xs)
        if hi <= lo: return [0.5 for _ in xs]
        r = hi - lo
        return [(x - lo) / r for x in xs]

    @staticmethod
    def _parse_cc(cc: object) -> float:
        """
        Extract a numeric compute indicator: supports '8.6', 'OpenCL 3.0', 'sm_86', etc.
        """
        try:
            s = str(cc)
            num = ""
            for ch in s:
                if ch.isdigit() or ch == ".":
                    num += ch
                elif num:
                    break
            return float(num) if num else 0.0
        except Exception:
            return 0.0

    def _score_devices(self, devices: List[DeviceInfo], policy: str) -> List[DeviceInfo]:
        if not devices: return []

        mems, comps, perfs, prios = [], [], [], []
        for d in devices:
            mems.append(float(d.memory_total_mb or 0))
            comps.append(self._parse_cc(d.compute_capability))
            perfs.append(float(self.perf_cache.get((d.backend, d.device_id), 0.0)))
            prios.append(float(BACKEND_PRIORITY.get(d.backend.upper(), -1)))

        n_mem, n_comp, n_perf, n_prio = map(self._normalize, (mems, comps, perfs, prios))

        presets = {
            "AUTO":        {"prio": 0.35, "memory": 0.20, "compute": 0.30, "perf": 0.15},
            "THROUGHPUT":  {"prio": 0.15, "memory": 0.15, "compute": 0.30, "perf": 0.40},
            "LATENCY":     {"prio": 0.25, "memory": 0.10, "compute": 0.35, "perf": 0.30},
            "MEMORY":      {"prio": 0.10, "memory": 0.70, "compute": 0.10, "perf": 0.10},
        }
        w = presets.get(policy, presets["AUTO"])

        scored: List[DeviceInfo] = []
        for i, d in enumerate(devices):
            if not d.is_available:
                continue
            s = (
                w["prio"]   * n_prio[i] +
                w["memory"] * n_mem[i]  +
                w["compute"]* n_comp[i] +
                w["perf"]   * n_perf[i]
            )
            scored.append(DeviceInfo(**{**d.__dict__, "score": float(s)}))

        scored.sort(key=lambda di: di.score, reverse=True)
        return scored
