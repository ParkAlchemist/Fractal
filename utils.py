import os
import numpy as np
from mpmath import mp
from numba import cuda
import pyopencl as cl
from enums import Kernel

def fractal_to_image_coords(fx, fy, center_x, center_y, zoom, image_width, image_height):
    px = int((fx - center_x) * zoom + image_width / 2)
    py = int((fy - center_y) * zoom + image_height / 2)
    return px, py

def image_to_fractal_coords(px, py, center_x, center_y, zoom, image_width, image_height):
    fx = (px - image_width / 2) / zoom + center_x
    fy = (py - image_height / 2) / zoom + center_y
    return fx, fy

def clear_cache_lock():
    cache_dir = os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~\\AppData\\Local")),
        "pyopencl", "pyopencl", "Cache", "pyopencl-compiler-cache-v2-py3.13.9.final.0"
    )
    lock_file = os.path.join(cache_dir, "lock")
    if os.path.exists(lock_file):
        try: os.remove(lock_file)
        except PermissionError: pass

def available_backends():
    backs = []
    try:
        if cl.get_platforms():
            backs.append(Kernel.OPENCL.name)
    except Exception:
        pass
    if cuda.is_available():
        backs.append(Kernel.CUDA.name)
    backs.append(Kernel.CPU.name)
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
        z = z*z + c
    return out
