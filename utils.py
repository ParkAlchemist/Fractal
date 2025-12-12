import os
import numpy as np
from PyQt5.QtGui import QImage
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
    except Exception as e:
        print("Error in getting OpenCL platforms: ", e)
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

def qimage_to_ndarray(img: QImage, require_copy: bool = True) -> np.ndarray:
    """
    Convert a QImage to a NumPy ndarray with shape (h, w, c) where c ∈ {1,3,4},
    handling stride and common QImage formats. Returns a copy by default to avoid
    lifetime issues; set require_copy=False to get a view (unsafe if QImage changes).
    """
    if img is None or img.isNull():
        raise ValueError("qimage_to_ndarray: input QImage is null")

    fmt = img.format()
    w, h = img.width(), img.height()
    bpl = img.bytesPerLine()  # stride in bytes

    # Access raw data
    ptr = img.bits()  # use constBits() in PyQt6
    ptr.setsize(h * bpl)  # expose full buffer to Python

    # Map common formats
    if fmt in (QImage.Format_RGB32, QImage.Format_ARGB32,
               QImage.Format_ARGB32_Premultiplied):
        # Memory layout is 4 bytes per pixel as BGRA on little-endian
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl // 4, 4))
        arr = arr[:, :w, :]  # drop padding at end of row, if any
        # Convert BGRA -> RGBA for conventional use
        arr = arr[..., [2, 1, 0, 3]]
        if fmt == QImage.Format_RGB32:
            # alpha channel is 0xFF; you can drop it to get (h,w,3)
            arr = arr[..., :3]  # RGBA -> RGB
        elif fmt == QImage.Format_ARGB32_Premultiplied:
            # Optional: unpremultiply if needed
            a = arr[..., 3:4].astype(np.float32)
            nz = a != 0
            arr[..., :3] = np.where(nz, (
                        arr[..., :3].astype(np.float32) * 255.0 / a).clip(0,
                                                                          255),
                                    0).astype(np.uint8)

    elif fmt in (QImage.Format_RGBA8888, QImage.Format_RGBA8888_Premultiplied):
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl // 4, 4))[
            :, :w, :]
        if fmt == QImage.Format_RGBA8888_Premultiplied:
            a = arr[..., 3:4].astype(np.float32)
            nz = a != 0
            arr[..., :3] = np.where(nz, (
                        arr[..., :3].astype(np.float32) * 255.0 / a).clip(0,
                                                                          255),
                                    0).astype(np.uint8)

    elif fmt == QImage.Format_RGB888:
        # 3 bytes per pixel, note that bytesPerLine may include padding; shape via stride
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl // 3, 3))[
            :, :w, :]
        # Qt stores RGB in RGB order here; no channel swap needed

    elif fmt in (QImage.Format_Grayscale8, QImage.Format_Alpha8):
        arr = np.frombuffer(ptr, dtype=np.uint8).reshape((h, bpl))[:, :w]
        # produce (h,w,1) for consistency
        arr = arr[..., np.newaxis]

    else:
        # Fallback: convert to RGBA8888 for a dependable path
        converted = img.convertToFormat(QImage.Format_RGBA8888)
        return qimage_to_ndarray(converted, require_copy=require_copy)

    # Return a copy by default (safe)—prevents issues if the underlying QImage goes out of scope
    return arr.copy() if require_copy else arr

def ndarray_to_qimage(arr: np.ndarray) -> QImage:
    """
    Convert a ndarray of shape (h,w), (h,w,1), (h,w,3) [RGB], or (h,w,4) [RGBA]
    into a QImage. Returns a QImage that owns its memory (deep copy).
    """
    if arr.ndim == 2:
        h, w = arr.shape
        qimg = QImage(arr.data.tobytes(), w, h, w, QImage.Format_Grayscale8)
        return qimg.copy()

    if arr.ndim == 3 and arr.shape[2] in (1, 3, 4):
        h, w, c = arr.shape
        if c == 1:
            qimg = QImage(arr.data.tobytes(), w, h, w, QImage.Format_Grayscale8)
            return qimg.copy()
        elif c == 3:
            # Qt expects RGB888
            bytes_per_line = 3 * w
            qimg = QImage(arr.data.tobytes(), w, h, bytes_per_line, QImage.Format_RGB888)
            return qimg.copy()
        elif c == 4:
            # Qt RGBA8888 is a safe, explicit format
            bytes_per_line = 4 * w
            qimg = QImage(arr.data.tobytes(), w, h, bytes_per_line,
                          QImage.Format_RGBA8888)
            return qimg.copy()

    raise ValueError(f"Unsupported ndarray shape {arr.shape}")


def qimage_to_np_gray(img: QImage) -> np.ndarray:
    """Convert QImage (RGB32/ARGB32) to normalized luminance [0,1]."""
    w, h = img.width(), img.height()
    ptr = img.bits()
    ptr.setsize(img.byteCount())
    arr = np.frombuffer(ptr, dtype=np.uint8).reshape(h, img.bytesPerLine() // 4, 4)
    # BGRA or ARGB; take channels robustly
    r = arr[..., 2].astype(np.float32)
    g = arr[..., 1].astype(np.float32)
    b = arr[..., 0].astype(np.float32)
    # simple luminance
    gray = (0.2126 * r + 0.7152 * g + 0.0722 * b) / 255.0
    return gray

def gradient_mag(gray: np.ndarray) -> np.ndarray:
    """Cheap gradient magnitude using 1st differences."""
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = (gray[:, 2:] - gray[:, :-2]) * 0.5
    gy[1:-1, :] = (gray[2:, :] - gray[:-2, :]) * 0.5
    return np.sqrt(gx*gx + gy*gy)
