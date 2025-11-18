import time
import os
import numpy as np
import csv
import platform
from numba import cuda
import pyopencl as cl

from fractal import Mandelbrot


# Detect CPU and GPU info
cpu_info = platform.processor() or platform.machine()
try:
    cuda.detect()
    gpu_info = cuda.get_current_device().name.decode('utf-8')
except Exception:
    gpu_info = "No CUDA GPU detected"


def benchmark_renderer(renderer, name, runs=3):
    print(f'Benchmarking {name} ({runs} runs)...')
    test_min_x, test_max_x = -2.0, 1.0
    test_min_y, test_max_y = -1.5, 1.5
    renderer.render(test_min_x, test_max_x, test_min_y, test_max_y)
    times = []
    for _ in range(runs):
        start = time.time()
        renderer.render(test_min_x, test_max_x, test_min_y, test_max_y)
        times.append(time.time() - start)
    avg_time = sum(times) / runs
    fps = 1.0 / avg_time if avg_time > 0 else 0
    print(f'{name} average render time: {avg_time:.3f}s | FPS: {fps:.2f}')
    return avg_time, fps

def main():
    resolutions = [(800, 600), (1280, 720), (1920, 1080)]
    max_iter = 500
    palette = np.random.randint(0, 255, size=(256 * 3), dtype=np.uint8)
    width, height = 100, 100
    cpu_renderer = Mandelbrot(palette=palette, img_width=width, img_height=height, max_iter=max_iter, kernel='cpu')
    cuda_renderer = None
    opencl_renderer = None

    cuda_available = cuda.is_available()
    cl_available = True if cl.get_platforms() else False

    if cuda_available:
        cuda_renderer = Mandelbrot(palette=palette, img_width=width, img_height=height, max_iter=max_iter, kernel='cuda')
    if cl_available:
        opencl_renderer = Mandelbrot(palette=palette, img_width=width, img_height=height, max_iter=max_iter, kernel='opencl')

    csv_file = 'benchmark_results.csv'

    if os.path.exists(csv_file):
        os.remove(csv_file)

    with open(csv_file, 'w', newline='') as f:
        writer = csv.writer(f)
        # Hardware summary header
        writer.writerow(['Hardware Summary'])
        writer.writerow(['CPU', cpu_info])
        writer.writerow(['GPU', gpu_info])
        writer.writerow([])
        # Column headers
        headers = ['Resolution', 'CPU Time (s)', 'CPU FPS']
        if cuda_available: headers += ['CUDA Time (s)', 'CUDA FPS']
        if cl_available: headers += ['OpenCL Time (s)', 'OpenCL FPS']
        writer.writerow(headers)
        for res in resolutions:
            width, height = res
            print(f'=== Benchmarking resolution: {width}x{height} ===')
            cpu_renderer.change_image_size(width, height)
            if cuda_renderer: cuda_renderer.change_image_size(width, height)
            if opencl_renderer: opencl_renderer.change_image_size(width, height)
            cpu_time, cpu_fps = benchmark_renderer(cpu_renderer, 'CPU')
            row = [f'{width}x{height}', f'{cpu_time:.3f}', f'{cpu_fps:.2f}']
            if cuda_renderer:
                cuda_time, cuda_fps = benchmark_renderer(cuda_renderer, 'CUDA')
                row += [f'{cuda_time:.3f}', f'{cuda_fps:.2f}']
            if opencl_renderer:
                opencl_time, opencl_fps = benchmark_renderer(opencl_renderer,
                                                             'OpenCL')
                row += [f'{opencl_time:.3f}', f'{opencl_fps:.2f}']
            writer.writerow(row)
        print(f'Benchmark results saved to {csv_file}')

if __name__ == '__main__':
    main()
