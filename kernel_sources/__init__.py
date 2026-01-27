# Kernel sources package
from .loader import load_kernel
from .registry import (register_kernel, register_op_descriptor,
                       list_kernels, get_op_descriptor, iter_registry)

__all__ = [
    "load_kernel",
    "register_kernel",
    "register_op_descriptor",
    "list_kernels",
    "get_op_descriptor",
    "iter_registry"
]
__version__ = "0.2.0"

