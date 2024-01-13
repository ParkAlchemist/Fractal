import numpy as np
from scipy.interpolate import CubicSpline
from PIL import Image

def create_smooth_gradient():
    positions = np.array([0.0, 0.16, 0.42, 0.6425, 0.8575])
    colors = np.array([
        (0, 7, 100),
        (32, 107, 203),
        (237, 255, 255),
        (255, 170, 0),
        (0, 2, 0)
    ])

    # Ensure positions are in the range [0, 1)
    positions = np.clip(positions, 0, 1)

    # Use monotone cubic interpolation
    cubic_interpolation = CubicSpline(positions, colors, bc_type='clamped', extrapolate=True)

    # Generate the smooth gradient with high resolution
    smooth_gradient = cubic_interpolation(np.linspace(0, 1, 2048))

    return smooth_gradient

def compute_color(i, re, im, scale, gradient):
    smoothed = np.log2(np.log2(re**2 + im**2) / 2)
    color_i = int(np.sqrt(i + 10 - smoothed) * scale) % len(gradient)
    color = gradient[color_i]

    # Check for NaN and replace with a default color
    if np.isnan(color).any():
        default_color = (0, 0, 0)  # Replace with your default color
        color = default_color

    return tuple(map(int, color))

if __name__ == '__main__':
    width, height = 512, 512
    scale = 256

    # Generate the smooth gradient
    smooth_gradient = create_smooth_gradient()

    # Example usage:
    i = 1000  # Replace with your actual iteration number
    re = 0.5  # Replace with your actual diverged coordinates
    im = 0.5
    color = compute_color(i, re, im, scale, smooth_gradient)
    print(f"Computed Color: {color}")

    # You can use the color in the rendering of Mandelbrot or Julia sets
    # ...

    # Create an image using the smooth gradient (for visualization)
    gradient_image = Image.fromarray(np.uint8(smooth_gradient))
    gradient_image.show()
