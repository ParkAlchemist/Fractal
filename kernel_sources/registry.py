from __future__ import annotations
from typing import Callable, Dict, Tuple, Any, List

# Nested dict: [fractal][op_name][backend][precision] -> meta
_REGISTRY: Dict[str, Dict[str, Dict[str, Dict[str, Dict[str, Any]]]]] = {}

# Static op descriptors (backend-agnostic): defaults, dependencies, etc.
_OP_DESCRIPTORS: Dict[str, Dict[str, Dict[str, Any]]] = {}
# shape: [fractal][op_name] -> {"depends_on": ["iter"], "default_params": {...}, ...}

def register_kernel(fractal: str, op_name: str, backend: str, precision: str, **meta: Any) -> None:
    """
    Register kernel metadata for a given fractal, operation, backend, and precision.
    Example:
        register_kernel("mandelbrot", "iter", "CUDA", "f32", func=my_cuda_func, arg_order=[...])
    """
    _REGISTRY.setdefault(fractal, {}).setdefault(op_name, {}).setdefault(backend.upper(), {})[precision] = meta

def register_op_descriptor(fractal: str, op_name: str, **descriptor: Any) -> None:
    """
    Register static operation descriptor for a given fractal and operation.
    Example:
        register_op_descriptor("mandelbrot", "iter", depends_on=["none"], default_params={"bailout": 4.0})
    """
    _OP_DESCRIPTORS.setdefault(fractal, {})[op_name] = descriptor

def load_kernel(backend: str, fractal: str, op_name: str, precision: str) -> Dict[str, Any]:
    """
    Load kernel metadata from the registry for the given parameters.
    Raises KeyError if not found.
    """
    be = backend.upper()
    try:
        meta = _REGISTRY[fractal][op_name][be][precision]
    except KeyError as e:
        raise KeyError(f"Kernel not found for fractal='{fractal}', op='{op_name}', backend='{be}', precision='{precision}'") from e
    return meta

def list_kernels(fractal: str, backend: str, precision: str) -> List[str]:
    """
    List all registered operation names for the given fractal, backend, and precision.
    """
    be = backend.upper()
    if fractal not in _REGISTRY:
        return []
    ops = []
    for op_name, backends in _REGISTRY[fractal].items():
        if be in backends and precision in backends[be]:
            ops.append(op_name)
    return sorted(ops)

def get_op_descriptor(fractal: str, op_name: str) -> Dict[str, Any]:
    """
    Get the static operation descriptor for the given fractal and operation.
    Raises KeyError if not found.
    """
    try:
        descriptor = _OP_DESCRIPTORS[fractal][op_name]
    except KeyError as e:
        raise KeyError(f"Operation descriptor not found for fractal='{fractal}', op='{op_name}'") from e
    return descriptor

def iter_registry():
    return _REGISTRY
