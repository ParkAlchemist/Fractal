import os


def fractal_to_image_coords(fx, fy, center_x, center_y, zoom, image_width, image_height):
    """
    Convert fractal coordinates to pixel coordinates in the rendered image.

    Parameters:
        fx, fy (float): Coordinates in fractal space (complex plane).
        center_x, center_y (float): Fractal coordinates at the center of the rendered image.
        zoom (float): Pixels per unit in fractal space.
        image_width, image_height (int): Dimensions of the rendered image (e.g., 2× viewport size).

    Returns:
        (int, int): Pixel coordinates (px, py) in the rendered image.
    """
    px = int((fx - center_x) * zoom + image_width / 2)
    py = int((fy - center_y) * zoom + image_height / 2)
    return px, py


def image_to_fractal_coords(px, py, center_x, center_y, zoom, image_width, image_height):
    """
    Convert pixel coordinates in the rendered image to fractal coordinates.

    Parameters:
        px, py (int): Pixel coordinates in the rendered image.
        center_x, center_y (float): Fractal coordinates at the center of the rendered image.
        zoom (float): Pixels per unit in fractal space.
        image_width, image_height (int): Dimensions of the rendered image.

    Returns:
        (float, float): Fractal coordinates (fx, fy).
    """
    fx = (px - image_width / 2) / zoom + center_x
    fy = (py - image_height / 2) / zoom + center_y
    return fx, fy


def viewport_to_image_coords(vx, vy, image_width, image_height, viewport_width, viewport_height):
    """
    Convert viewport pixel coordinates (e.g., mouse position) to image coordinates.

    Parameters:
        vx, vy (int): Pixel coordinates in the viewport (e.g., QLabel).
        image_width, image_height (int): Dimensions of the rendered image (2× viewport size).
        viewport_width, viewport_height (int): Dimensions of the viewport.

    Returns:
        (int, int): Corresponding pixel coordinates in the rendered image.
    """
    offset_x = (image_width - viewport_width) // 2
    offset_y = (image_height - viewport_height) // 2
    return vx + offset_x, vy + offset_y


def image_to_viewport_coords(ix, iy, image_width, image_height, viewport_width, viewport_height):
    """
    Convert image pixel coordinates to viewport-relative coordinates.

    Parameters:
        ix, iy (int): Pixel coordinates in the rendered image.
        image_width, image_height (int): Dimensions of the rendered image.
        viewport_width, viewport_height (int): Dimensions of the viewport.

    Returns:
        (int, int): Pixel coordinates relative to the viewport.
    """
    offset_x = (image_width - viewport_width) // 2
    offset_y = (image_height - viewport_height) // 2
    return ix - offset_x, iy - offset_y


def viewport_to_fractal_coords(vx, vy, center_x, center_y, zoom, viewport_width, viewport_height):
    """
    Convert viewport pixel coordinates to fractal coordinates.

    Parameters:
        vx, vy (int): Pixel coordinates in the viewport (e.g., mouse position).
        center_x, center_y (float): Fractal coordinates at the center of the viewport.
        zoom (float): Pixels per unit in fractal space.
        viewport_width, viewport_height (int): Dimensions of the viewport.

    Returns:
        (float, float): Fractal coordinates (fx, fy).
    """
    fx = (vx - viewport_width / 2) / zoom + center_x
    fy = (vy - viewport_height / 2) / zoom + center_y
    return fx, fy


def fractal_to_viewport_coords(fx, fy, center_x, center_y, zoom, viewport_width, viewport_height):
    """
    Convert fractal coordinates to viewport pixel coordinates.

    Parameters:
        fx, fy (float): Coordinates in fractal space.
        center_x, center_y (float): Fractal coordinates at the center of the viewport.
        zoom (float): Pixels per unit in fractal space.
        viewport_width, viewport_height (int): Dimensions of the viewport.

    Returns:
        (int, int): Pixel coordinates (vx, vy) in the viewport.
    """
    vx = int((fx - center_x) * zoom + viewport_width / 2)
    vy = int((fy - center_y) * zoom + viewport_height / 2)
    return vx, vy

def clear_cache_lock():
    # --- Automatically remove stale PyOpenCL cache lock ---
    cache_dir = os.path.join(
        os.environ.get("LOCALAPPDATA", os.path.expanduser("~\\AppData\\Local")),
        "pyopencl", "pyopencl", "Cache",
        "pyopencl-compiler-cache-v2-py3.13.9.final.0"
    )
    lock_file = os.path.join(cache_dir, "lock")
    if os.path.exists(lock_file):
        try:
            os.remove(lock_file)
            print(f"Removed stale PyOpenCL cache lock: {lock_file}")
        except PermissionError:
            print(f"Could not remove PyOpenCL cache lock: {lock_file}, check permissions.")