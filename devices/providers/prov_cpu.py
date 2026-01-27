from __future__ import annotations
from typing import List, Dict

class CpuDeviceProvider:
    backend = "CPU"

    @staticmethod
    def enumerate() -> List[Dict]:
        return [{
            "device_id": None,
            "name": "CPU",
            "vendor": "Generic",
            "driver": None,
            "compute_capability": None,
            "memory_total_mb": None,
            "memory_free_mb": None,
            "is_available": True,
            "extra": {}
        }]
