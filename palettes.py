import numpy as np
from scipy.interpolate import interp1d

def denormalize(palette):
    return [tuple(int(channel * 255) for channel in color) for color in palette]

def edge():
    exterior = [(1, 1, 1)] * 50
    interior = [(1, 1, 1)] * 5
    gray_area = [(1 - i / 44,) * 3 for i in range(45)]
    return denormalize(exterior + gray_area + interior)

def create_smooth_gradient(palette, resolution=1000, interpolation='cubic'):
    """
    Generates a smooth gradient from a list of RGB tuples using interpolation.
    Returns a list of RGB tuples.
    """
    palette = np.array(palette)
    indices = np.linspace(0, len(palette) - 1, len(palette))
    interp_func = interp1d(indices, palette, kind=interpolation, axis=0, fill_value="extrapolate")
    smooth_gradient = interp_func(np.linspace(0, len(palette) - 1, resolution))
    return [tuple(map(int, np.clip(color, 0, 255))) for color in smooth_gradient]

# Define base palettes
base_palettes = {
    "Classic": [(0, 0, 0), (66, 30, 15), (25, 7, 26), (9, 1, 47),
                (4, 4, 73), (0, 7, 100), (12, 44, 138),
                (24, 82, 177), (57, 125, 209), (134, 181, 229),
                (211, 236, 248), (241, 233, 191), (248, 201, 95),
                (255, 170, 0), (204, 128, 0), (153, 87, 0), (106, 52, 3)],
    "Fire": [(0, 0, 0), (255, 0, 0), (255, 85, 0), (255, 170, 0),
             (255, 255, 0), (255, 255, 85), (255, 255, 170)],
    "Ocean": [(0, 0, 0), (0, 32, 64), (0, 64, 128), (0, 96, 192),
              (0, 128, 255), (64, 160, 255), (128, 192, 255)],
    "Smooth": create_smooth_gradient([(0, 0, 0), (66, 30, 15), (25, 7, 26), (9, 1, 47),
                                      (4, 4, 73), (0, 7, 100), (12, 44, 138),
                                      (24, 82, 177), (57, 125, 209), (134, 181, 229),
                                      (211, 236, 248), (241, 233, 191), (248, 201, 95),
                                      (255, 170, 0), (204, 128, 0), (153, 87, 0), (106, 52, 3)]),
    "Edge": edge()
}

# Export palettes dictionary
palettes = base_palettes
