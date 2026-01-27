from __future__ import annotations
from typing import List, Dict
import logging
logger = logging.getLogger(__name__)

class CudaDeviceProvider:
    backend = "CUDA"

    @staticmethod
    def enumerate() -> List[Dict]:
        try:
            """
            Probe CUDA devices (via pycuda) and return device dicts for DeviceInfo.
            Falls back to an empty list if pycuda is not available or probe fails.
            """
            devs: List[Dict] = []
            try:
                import pycuda.driver as pycuda
                pycuda.init()
            except Exception:
                logger.exception(
                    "pycuda not available or failed to initialize")
                return devs

            try:
                count = pycuda.Device.count()
            except Exception:
                logger.exception("Failed to query CUDA device count")
                return devs

            for i in range(count):
                try:
                    d = pycuda.Device(i)
                    # device name and memory
                    name = getattr(d, "name", lambda: f"CUDA Device {i}")()
                    try:
                        total_bytes = int(
                            getattr(d, "total_memory",
                                    lambda: d.total_memory)())
                    except Exception:
                        # pycuda may expose total_memory as method or attribute
                        try:
                            total_bytes = int(d.total_memory())
                        except Exception:
                            total_bytes = 0
                    total_mb = int(total_bytes // (1024 ** 2))
                    # compute capability as float if possible
                    compute_capability = None
                    try:
                        cc = d.compute_capability()
                        if isinstance(cc, (tuple, list)) and len(cc) >= 2:
                            compute_capability = float(f"{cc[0]}.{cc[1]}")
                        else:
                            compute_capability = float(cc)
                    except Exception:
                        compute_capability = None
                    # raw attributes (may include multiprocessor count, clocks, etc.)
                    extra = {}
                    try:
                        attrs = d.get_attributes()
                        # keep raw attrs under extra; specific named fields if present
                        extra["raw_attrs"] = dict(attrs)
                        # map common keys if present (names differ across bindings)
                        # multiprocessor count
                        mp = None
                        for k, v in attrs.items():
                            # use attribute name if available
                            try:
                                name_k = getattr(k, "name", str(k)).lower()
                            except Exception:
                                name_k = str(k).lower()
                            if "multiprocessor" in name_k or "mp" in name_k:
                                mp = int(v)
                        if mp is not None:
                            extra["multiprocessors"] = mp
                    except Exception:
                        extra["raw_attrs"] = None

                    # driver/runtime versions if available
                    try:
                        driver = getattr(pycuda, "get_driver_version",
                                         lambda: None)()
                    except Exception:
                        driver = None

                    devs.append({
                        "device_id": i,
                        "name": name,
                        "vendor": "NVIDIA",
                        "driver": driver,
                        "compute_capability": compute_capability,
                        "memory_total_mb": total_mb,
                        "memory_free_mb": None,
                        "is_available": True,
                        "extra": extra,
                    })
                except Exception:
                    logger.exception(
                        "Failed to read CUDA device info for index %d", i)
            return devs
        except Exception as e:
            logger.debug("CUDA enumerate failed: %s", e)
            return []
