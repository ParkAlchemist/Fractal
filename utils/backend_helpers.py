import os

import numpy as np
import pyopencl as cl
from mpmath import mp
from numba import cuda

from utils.enums import BackendType


def clear_cache_lock():
    cache_dir = os.path.join(
        os.environ.get("LOCALAPPDATA",
                       os.path.expanduser("~\\AppData\\Local")),
        "pyopencl", "pyopencl", "Cache",
        "pyopencl-compiler-cache-v2-py3.13.9.final.0"
    )
    lock_file = os.path.join(cache_dir, "lock")
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
        except PermissionError:
            pass


def available_backends():
    backs = []
    try:
        if cl.get_platforms():
            backs.append(BackendType.OPENCL.name)
    except Exception as e:
        print("Error in getting OpenCL platforms: ", e)
        pass
    if cuda.is_available():
        backs.append(BackendType.CUDA.name)
    backs.append(BackendType.CPU.name)
    return backs


def make_reference_orbit_hp(c_ref: complex, max_iter: int, mp_dps: int = 160):
    """
    High-precision reference orbit z*_n; returned as float64 array shape (N,2).
    """
    mp.dps = mp_dps
    c = mp.mpc(c_ref.real, c_ref.imag)
    z = mp.mpc(0, 0)
    out = np.empty((max_iter, 2), dtype=np.float64)
    for n in range(max_iter):
        out[n, 0] = float(z.real)
        out[n, 1] = float(z.imag)
        z = z * z + c
    return out
