from dataclasses import dataclass
import numpy as np

@dataclass
class JuliaSet:
    max_iterations: int
    escape_radius: float = 2.0
    julia_constant: complex = -0.7 + 0.27015j  # Adjust this constant based on your Julia Set

    def __contains__(self, c: np.ndarray, tolerance=1e-6) -> bool:
        return np.all(np.isclose(self.stability(c), 1, rtol=tolerance))

    def stability(self, c_values: np.ndarray, smooth=False,
                  clamp=True) -> np.ndarray:
        values = self.escape_count(c_values, smooth) / self.max_iterations
        return np.clip(values, 0.0, 1.0) if clamp else values

    def escape_count(self, c_values: np.ndarray, smooth=False) -> np.ndarray:
        z_values = np.zeros_like(c_values, dtype=np.complex128)
        iterations = np.zeros_like(c_values, dtype=np.float32)

        for iteration in range(self.max_iterations):
            mask = np.abs(z_values) <= self.escape_radius
            z_values[mask] = z_values[mask] ** 2 + self.julia_constant
            iterations[mask] += 1

        if smooth:
            mask = np.abs(z_values) > self.escape_radius
            iterations[mask] += 1 - np.log(np.log(np.abs(z_values[mask])) / np.log(2))

        return iterations.astype(float)


if __name__ == '__main__':
    # Example usage
    width = 800
    height = 600
    pixel_count = width * height
    pixel_start = 0
    scale = 0.01

    julia_set = JuliaSet(max_iterations=100, escape_radius=2.0)

    # Create an array of k values from 0 to pixel_count - 1
    k_values = np.arange(pixel_start, pixel_start + pixel_count)

    # Calculate x, y, re, im for all k values in one go
    x = np.mod(pixel_start + k_values, width)
    y = (pixel_start + k_values) // width
    re = scale * (x - width / 2)
    im = scale * (height / 2 - y)
    c_values = re + 1j * im

    # Check if the points are in the Julia set
    result = c_values[np.array([c in julia_set for c in c_values])]
    print(result)
