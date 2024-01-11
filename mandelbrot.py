from dataclasses import dataclass
import numpy as np


@dataclass
class MandelbrotSet:
    max_iterations: int
    escape_radius: float = 2.0

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
            z_values[mask] = z_values[mask] ** 2 + c_values[mask]
            iterations[mask] += 1

        if smooth:
            mask = np.abs(z_values) > self.escape_radius
            iterations[mask] += 1 - np.log(np.log(np.abs(z_values[mask])) / np.log(2))

        return iterations.astype(float)
