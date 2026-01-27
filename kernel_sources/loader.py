from __future__ import annotations
import importlib
from typing import Dict, Any

from kernel_sources.registry import load_kernel as load_registered


KERNEL_ROOT = "kernel_sources"

def _module_name(backend: str, fractal: str, operation: str) -> str:
    return f"{KERNEL_ROOT}.{backend.lower()}.{fractal.lower()}.{operation.lower()}"

def load_kernel(backend: str, fractal: str, operation: str, precision: str) -> Dict[str, Any]:
    """
    Import module by convention and return kernel metadata via get_kernel(precision).
    """
    try:
        meta = load_registered(backend, fractal, operation, precision)
        _validate_meta(backend, meta, f"registry[{fractal}.{operation}:{backend}/{precision}]")
        return meta
    except KeyError:
        raise

def _validate_meta(backend: str, meta: Dict[str, Any], where: str) -> None:
    if "arg_order" not in meta or not isinstance(meta["arg_order"], (list, tuple)):
        raise KeyError(f"{where}.get_kernel() must return 'arg_order' list")
    if backend.upper() == "OPENCL":
        if "src" not in meta["func"] or "kernel_name" not in meta["func"]:
            raise KeyError(f"{where}.get_kernel() must return 'src' and 'kernel_name' for OpenCL")
    elif backend.upper() in ("CUDA", "CPU"):
        if "func" not in meta:
            raise KeyError(f"{where}.get_kernel() must return 'func' for {backend}")

    for key in ("scalars", "produces", "consumes"):
        if key not in meta or not isinstance(meta[key], (list, tuple)):
            meta.setdefault(key, [])
