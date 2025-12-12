# Fractal

A small Python program to visualize the Mandelbrot set with an interactive PySide6 GUI.

Repository: https://github.com/ParkAlchemist/Fractal

Table of Contents
- [Overview](#overview)
- [Roadmap](#roadmap)
- [Requirements](#requirements)
- [Installation](#installation)
- [Running the GUI](#running-the-gui)
- [Usage](#usage)
- [Exporting Images](#exporting-images)
- [Development](#development)
- [Contributing](#contributing)
- [Troubleshooting](#troubleshooting)
- [License](#license)
- [Contact](#contact)

## Overview
Fractal is a lightweight viewer for exploring the Mandelbrot set. It provides an interactive GUI built with PySide6 to pan, zoom, and tweak rendering parameters in real time. This README describes the project as it currently stands: a Python-only application with a single GUI entrypoint (`main.py`).

## Roadmap
This project is actively planned to evolve beyond the current Mandelbrot-only GUI. High-level goals and milestones:

Short-term (next sprints)
- GPU kernels for tile-based rendering
  - Implement improved GPU kernels to render tiles asynchronously so the UI remains responsive during high-resolution or deep-zoom renders.
  - Prototype with a Python GPU option (e.g., Numba + CUDA, PyOpenCL) or via a thin C/C++ extension if necessary.
- Tile-based renderer plumbing
  - Add tile scheduling and progressive refinement so coarse tiles render quickly and refine over time.
- CLI / headless renderer
  - Provide a command-line mode to render images and export frames without the GUI (useful for automated rendering and CI).

Mid-term
- Additional escape-time fractals
  - Add Julia set rendering (parameterized Julia seeds and presets).
  - Add Burning Ship fractal and other variations (Newton, Multibrot).
- More coloring modes & palettes
  - Histogram normalization, smooth coloring, distance estimation coloring, user palettes, and palette import/export.
- Export enhancements
  - High-bit-depth image support (TIFF), multi-threaded export, and a frames-to-video workflow.

Long-term
- Multiple GPU backends and auto-detection (CUDA, OpenCL, possibly Vulkan)
- Plugin/scripting API for custom formulas and post-processing
- Animation tools and non-linear camera paths
- Cross-platform packaging and release artifacts (PyPI package / prebuilt binaries)

How you can help
- File design/implementation issues describing specific roadmap tasks and tag them (e.g., `help wanted`, `good first issue`).
- Prototype or test GPU kernels and submit performance comparisons.
- Contribute palettes, presets, and example scenes.
- Help with documentation and packaging.

If you'd like to prioritize any specific item above (for example, Julia set first or a particular GPU backend), open an issue and label it `roadmap` so it can be scheduled.

## Requirements
- Python 3.10+ (should work on 3.9 in many environments, but 3.10+ is recommended)
- PySide6

Optional (if present/used in the repo):
- numpy — numeric helpers and speedups
- Pillow (PIL) — saving/exporting images

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ParkAlchemist/Fractal.git
cd Fractal
```

2. (Recommended) Create and activate a virtual environment:
```bash
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows (PowerShell)
.venv\Scripts\Activate.ps1
```

3. Install dependencies:
- If the repo contains a requirements file:
```bash
pip install -r requirements.txt
```
- If no requirements file is provided, install the minimum:
```bash
pip install PySide6
# optional helpers
pip install numpy Pillow
```

## Running the GUI

The application's entrypoint is `main.py`. From the project root run:
```bash
python main.py
```

This launches the PySide6 GUI. If you installed the package differently or run from an IDE, run the same `main.py` module.

If you prefer to run module-style (if the package exposes a module), you can also try:
```bash
python -m main
```
(Only use this if `main.py` is structured to be runnable as a module.)

## Usage

Once the GUI opens, typical interactions are:

- Pan the view by clicking and dragging.
- Zoom in/out using the mouse wheel or zoom controls in the UI.
- Adjust parameters such as maximum iterations, coloring/palette, and other available sliders or fields.
- Use any "Reset" or "Home" control to return to the default framed view.
- Use "Save" or "Export" (if provided in the UI) to write the current view to an image file.

UI controls and exact names may vary depending on current code; check the widgets visible in the window for available options.

## Exporting Images

Use the GUI's save/export button (if present) to export the current view as PNG (recommended). If the UI does not expose an export button, the codebase may include a helper for capturing the rendered widget — search the repo for functions named like `save_image`, `export`, or similar.

Suggested external workflow for animations (not currently included):
1. Render a series of frames (one per zoom-step or parameter-step).
2. Assemble frames with ffmpeg:
```bash
ffmpeg -r 30 -i frames/frame_%04d.png -c:v libx264 -pix_fmt yuv420p out/animation.mp4
```

## Development

Suggested local workflow:
1. Create a branch for your work:
```bash
git checkout -b feat/<short-description>
```
2. Make changes, test the GUI by running `python main.py`.
3. Commit and open a pull request with screenshots for GUI changes.

If you add features that require new dependencies, add them to `requirements.txt` (if present) or document them in this README.

Testing:
- The project currently focuses on GUI behavior. If you add non-GUI logic (rendering helpers, math utilities), consider adding unit tests for those modules.

## Contributing

Contributions and bug reports are welcome. To contribute:
- Open an issue describing the problem or feature request (include screenshots for UI issues).
- Fork the repository and submit a pull request with a clear description of changes.
- Keep commits focused and include tests where applicable.

For roadmap-related contributions, please:
- Reference the specific roadmap item in your issue or PR (e.g., "Roadmap: Tile-based GPU rendering").
- Provide short performance notes or screenshots where applicable.

## Troubleshooting

- PySide6 import errors:
  - Ensure you installed PySide6 into the active Python environment:
    ```bash
    pip install PySide6
    ```
  - On Linux/Wayland, if GUI scaling or rendering behaves oddly, try running under Xorg or adjust QT environment variables (e.g., QT_QPA_PLATFORM).

- Performance:
  - For very deep zooms or large windows, rendering may be CPU-intensive. Close other CPU-heavy apps or reduce resolution / iterations.

- If `python main.py` fails, check for:
  - Missing dependencies (look at the top of the traceback to see which import failed).
  - File location — ensure you are running the command from the repository root or provide a correct path to `main.py`.

## License

Add your preferred license here. Example (MIT):
```
MIT License
Copyright (c) 2025 ParkAlchemist
```

## Contact

- Repo: https://github.com/ParkAlchemist/Fractal
- Issues: https://github.com/ParkAlchemist/Fractal/issues
- Author: ParkAlchemist

---
