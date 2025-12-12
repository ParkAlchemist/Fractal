"""
Benchmark the modular fractal renderer (CPU / CUDA / OpenCL).
Supports full-frame and tile engines, perturbation, and f32/f64 precision.

Usage examples:
  python benchmark.py --backends cpu,cuda,opencl-h --res 800x600,1280x720 \
      --precision f64 --max-iter 1000 --samples 2 --runs 5 --tile yes --tile-size 256x256

  python benchmark.py --backends opencl-h --opencl-prefer-cpu yes --tile yes

Author: refactored for the new architecture
"""

import os
import csv
import time
import argparse
import platform
from typing import List, Tuple, Optional

import numpy as np

# --- Architecture imports ----------------------------------------------------
from fractals.fractal_base import Viewport, RenderSettings
from fractals.mandelbrot import MandelbrotFractal
from rendering.renderer_core import Renderer
from rendering.render_engines import FullFrameEngine, TileEngine

from backends.backend_base import Backend
from backends.cpu_backend import CpuBackend

# CUDA (optional)
try:
    from backends.cuda_backend import CudaBackend
    import numba.cuda as cuda
    CUDA_AVAILABLE = cuda.is_available()
except Exception as e:
    print(f"Cuda not available: {e}")
    CUDA_AVAILABLE = False

# OpenCL hardened backend (robust on Windows drivers)
try:
    from backends.opencl_backend import OpenClBackend as OclBackend
    OCL_AVAILABLE = True
except Exception as e:
    print(f"OpenCL not available: {e}")
    OCL_AVAILABLE = False

# --- Helpers -----------------------------------------------------------------

def parse_resolution_list(res_str: str) -> List[Tuple[int, int]]:
    """
    Parse resolutions like "800x600,1280x720".
    """
    if not res_str:
        return [(800, 600), (1280, 720), (1920, 1080)]
    out: List[Tuple[int, int]] = []
    for token in res_str.split(','):
        token = token.strip().lower()
        if not token:
            continue
        w, h = token.split('x')
        out.append((int(w), int(h)))
    return out

def parse_tile_size(token: str) -> Tuple[int, int]:
    """
    Parse tile size like "256x256".
    """
    token = token.strip().lower().replace(' ', '')
    w, h = token.split('x')
    return int(w), int(h)

def precision_to_dtype(tag: str):
    """
    Map CLI precision tag -> numpy dtype.
    """
    tag = tag.lower()
    if tag in ("f32", "float32"):
        return np.float32
    if tag in ("f64", "float64", "double"):
        return np.float64
    raise ValueError(f"Unsupported precision: {tag}")

def human_backend_name(b: Backend) -> str:
    try:
        return b.name
    except Exception:
        return b.__class__.__name__

def device_summary(cuda_enabled: bool, ocl_enabled: bool, ocl_prefer_cpu: bool) -> Tuple[str, str]:
    """
    Return (CPU summary, Accelerator summary string).
    """
    cpu_info = platform.processor() or platform.machine()
    accel = []
    if cuda_enabled:
        try:
            dev = cuda.get_current_device()
            accel.append(f"CUDA: {dev.name.decode('utf-8') if hasattr(dev.name,'decode') else str(dev.name)}")
        except Exception:
            accel.append("CUDA: not available")
    if ocl_enabled:
        prefer = "CPU" if ocl_prefer_cpu else "GPU"
        accel.append(f"OpenCL ({prefer} preferred)")
    return cpu_info or "Unknown CPU", "; ".join(accel) if accel else "No accelerator"

# --- Benchmark core ----------------------------------------------------------

def make_backend(tag: str, ocl_prefer_cpu: bool) -> Optional[Backend]:
    tag = tag.lower().strip()
    if tag == "cpu":
        return CpuBackend()
    if tag == "cuda":
        if not CUDA_AVAILABLE:
            return None
        return CudaBackend()
    if tag in ("opencl", "opencl-h"):
        if not OCL_AVAILABLE:
            return None
        # Hardened backend accepts prefer_cpu flag; non-hardened ignores it.
        try:
            return OclBackend(prefer_cpu=ocl_prefer_cpu)  # type: ignore[arg-type]
        except TypeError:
            return OclBackend()  # fallback signature
    raise ValueError(f"Unknown backend tag: {tag}")

def build_renderer(backend: Backend,
                   settings: RenderSettings,
                   engine_mode: str,
                   tile_size: Tuple[int, int]) -> Renderer:
    """
    Build a Renderer(fractal, backend, settings, engine).
    """
    fractal = MandelbrotFractal()
    if engine_mode == "tile":
        engine = TileEngine(tile_w=tile_size[0], tile_h=tile_size[1], per_tile_reference=settings.use_perturb)
    else:
        engine = FullFrameEngine()
    return Renderer(fractal, backend, settings, engine=engine)

def do_render(renderer: Renderer, width: int, height: int) -> np.ndarray:
    """
    Perform a single render of the canonical viewport.
    """
    min_x, max_x = -2.0, 1.0
    min_y, max_y = -1.5, 1.5
    vp = Viewport(min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, width=width, height=height)
    return renderer.render(vp)

def benchmark_combo(backend: Backend,
                    precision: np.dtype,
                    max_iter: int,
                    samples: int,
                    perturb: bool,
                    engine_mode: str,
                    tile_size: Tuple[int, int],
                    width: int,
                    height: int,
                    runs: int,
                    warmup: int = 1) -> Tuple[float, float]:
    """
    Runs warmups (not timed), then 'runs' timed renders.
    Returns (avg_time_seconds, fps).
    """
    settings = RenderSettings(max_iter=max_iter,
                              samples=samples,
                              precision=precision,
                              use_perturb=perturb,
                              perturb_order=2,
                              perturb_thresh=1e-6,
                              hp_dps=160)
    renderer = build_renderer(backend, settings, engine_mode, tile_size)

    # Compile/build JIT once before timing
    renderer.backend.compile(renderer.fractal, renderer.settings)

    # Warm-ups
    for _ in range(max(0, warmup)):
        _ = do_render(renderer, width, height)

    # Timed runs
    times = []
    for _ in range(runs):
        t0 = time.perf_counter()
        _ = do_render(renderer, width, height)
        t1 = time.perf_counter()
        times.append(t1 - t0)

    avg = sum(times) / len(times)
    fps = 1.0 / avg if avg > 0 else 0.0
    return avg, fps

# --- CSV writer --------------------------------------------------------------

def write_csv_row(writer: csv.writer,
                  resolution: Tuple[int, int],
                  rows_by_backend: List[Tuple[str, Optional[Tuple[float, float]]]]):
    """
    rows_by_backend: list of (backend_label, (avg, fps)) where the tuple may be None if backend not available
    """
    base = [f"{resolution[0]}x{resolution[1]}"]
    for _, result in rows_by_backend:
        if result is None:
            base.extend(["n/a", "n/a"])
        else:
            avg, fps = result
            base.extend([f"{avg:.4f}", f"{fps:.2f}"])
    writer.writerow(base)

# --- CLI ---------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser(description="Benchmark modular Mandelbrot renderer.")
    p.add_argument("--backends", type=str, default="cpu,cuda,opencl-h",
                   help="Comma separated list: cpu,cuda,opencl or opencl-h (hardened)")
    p.add_argument("--res", type=str, default="800x600,1280x720,1920x1080",
                   help="Comma separated WxH list")
    p.add_argument("--precision", type=str, default="f32", choices=["f32", "f64", "float32", "float64", "double"])
    p.add_argument("--max-iter", type=int, default=500)
    p.add_argument("--samples", type=int, default=1)
    p.add_argument("--runs", type=int, default=3)
    p.add_argument("--warmup", type=int, default=1)
    p.add_argument("--perturb", type=str, default="no", choices=["yes", "no"])
    p.add_argument("--tile", type=str, default="no", choices=["yes", "no"])
    p.add_argument("--tile-size", type=str, default="256x256")
    p.add_argument("--opencl-prefer-cpu", type=str, default="no", choices=["yes", "no"])
    p.add_argument("--csv", type=str, default="benchmark_results.csv")
    args = p.parse_args()

    backends_tags = [t.strip().lower() for t in args.backends.split(",") if t.strip()]
    resolutions = parse_resolution_list(args.res)
    precision = precision_to_dtype(args.precision)
    tile_mode = (args.tile.lower() == "yes")
    tile_size = parse_tile_size(args.tile_size)
    perturb = (args.perturb.lower() == "yes")
    ocl_prefer_cpu = (args.opencl_prefer_cpu.lower() == "yes")

    # Hardware summary
    cpu_info, accel_info = device_summary(CUDA_AVAILABLE, OCL_AVAILABLE, ocl_prefer_cpu)
    print("=== Hardware Summary ===")
    print("CPU:", cpu_info)
    print("Accel:", accel_info)
    print()

    # Build backends
    backends: List[Tuple[str, Optional[Backend]]] = []
    for tag in backends_tags:
        try:
            b = make_backend(tag, ocl_prefer_cpu)
            if b is None:
                print(f"[SKIP] Backend '{tag}' is not available on this machine.")
            else:
                name = human_backend_name(b)
                print(f"[OK] Using backend: {name}")
                backends.append((name, b))
        except Exception as e:
            print(f"[ERR] Backend '{tag}' init failed: {e}")
            backends.append((tag.upper(), None))
    print()

    # Prepare CSV
    if os.path.exists(args.csv):
        os.remove(args.csv)
    with open(args.csv, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["Hardware Summary"])
        writer.writerow(["CPU", cpu_info])
        writer.writerow(["Accelerators", accel_info])
        writer.writerow([])

        # Header row
        header = ["Resolution"]
        for name, _ in backends:
            header.extend([f"{name} Time (s)", f"{name} FPS"])
        writer.writerow(header)

        # Benchmark loops
        engine_mode = "tile" if tile_mode else "full"
        print(f"Engine mode: {engine_mode}, tile={tile_size if tile_mode else '-'}")
        print(f"Settings: precision={precision}, max_iter={args.max_iter}, samples={args.samples}, perturb={perturb}")
        print()

        for (w, h) in resolutions:
            print(f"=== {w}x{h} ===")
            row_results: List[Tuple[str, Optional[Tuple[float, float]]]] = []
            for name, backend in backends:
                if backend is None:
                    row_results.append((name, None))
                    continue
                try:
                    avg, fps = benchmark_combo(
                        backend=backend,
                        precision=precision,
                        max_iter=args.max_iter,
                        samples=args.samples,
                        perturb=perturb,
                        engine_mode=engine_mode,
                        tile_size=tile_size,
                        width=w,
                        height=h,
                        runs=args.runs,
                        warmup=args.warmup,
                    )
                    print(f"{name:>12}  avg={avg:.4f}s  fps={fps:.2f}")
                    row_results.append((name, (avg, fps)))
                except Exception as e:
                    print(f"{name:>12}  FAIL: {e}")
                    row_results.append((name, None))
            write_csv_row(writer, (w, h), row_results)
            print()

    print(f"Benchmark results saved to {args.csv}")

if __name__ == "__main__":
    main()
