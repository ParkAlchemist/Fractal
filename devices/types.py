from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict

@dataclass(frozen=True)
class DeviceInfo:
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
