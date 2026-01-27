from __future__ import annotations
from typing import List, Dict
import logging
logger = logging.getLogger(__name__)

class OpenClDeviceProvider:
    backend = "OPENCL"

    @staticmethod
    def enumerate() -> List[Dict]:
        try:
            """
            Probe pyopencl platforms/devices and return a list of device dicts
            compatible with DeviceInfo construction in BackendManager.
            """
            devs: List[Dict] = []
            try:
                import pyopencl as cl
            except Exception:
                logger.exception("pyopencl not available")
                return devs

            try:
                plats = cl.get_platforms()
            except Exception:
                logger.exception("Failed to query OpenCL platforms")
                return devs

            ordinal = 0
            for p in plats:
                for d in p.get_devices():
                    try:
                        name = getattr(d, "name", None)
                        vendor = getattr(d, "vendor", None)
                        # driver/version info: prefer device driver then platform version
                        driver = getattr(d, "driver_version", None) or getattr(
                            p,
                            "version",
                            None)
                        # compute capability/version string (try device.version then platform.version)
                        compute_capability = getattr(d, "version",
                                                     None) or getattr(p,
                                                                      "version",
                                                                      None)
                        total_mb = int(
                            getattr(d, "global_mem_size", 0) // (1024 ** 2))
                        extra = {
                            "cores": getattr(d, "max_compute_units", None),
                            "clock_mhz": getattr(d, "max_clock_frequency",
                                                 None),
                            # boolean indicating FP64 support if attribute present
                            "fp64": bool(getattr(d, "double_fp_config", None)),
                        }
                        devs.append({
                            "device_id": ordinal,
                            "name": name or f"OpenCL Device {ordinal}",
                            "vendor": vendor,
                            "driver": driver,
                            "compute_capability": compute_capability,
                            "memory_total_mb": total_mb,
                            "memory_free_mb": None,
                            "is_available": True,
                            "extra": extra,
                        })
                    except Exception:
                        logger.exception(
                            "Failed to read OpenCL device info for platform %s",
                            getattr(p, "name", "<unknown>"))
                    ordinal += 1
            return devs
        except Exception as e:
            logger.debug("OpenCL enumerate failed: %s", e)
            return []
