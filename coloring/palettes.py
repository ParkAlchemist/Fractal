import numpy as np
from scipy.interpolate import interp1d


def apply_gamma_correction(palette, gamma=0.8):
    """
    Applies gamma correction to a palette to increase contrast.

    Parameters:
        palette (list of tuple): List of RGB tuples (0–255).
        gamma (float): Gamma value (<1 brightens, >1 darkens).

    Returns:
        list of tuple: Gamma-corrected palette.
    """
    corrected = []
    for r, g, b in palette:
        r_corr = int(255 * ((r / 255) ** gamma))
        g_corr = int(255 * ((g / 255) ** gamma))
        b_corr = int(255 * ((b / 255) ** gamma))
        corrected.append((r_corr, g_corr, b_corr))
    return corrected

def stretch_contrast(palette):
    """
    Linearly stretches the RGB values to span the full 0–255 range.

    Parameters:
        palette (list of tuple): List of RGB tuples (0–255).

    Returns:
        list of tuple: Contrast-stretched palette.
    """
    arr = np.array(palette, dtype=np.float32)
    min_vals = arr.min(axis=0)
    max_vals = arr.max(axis=0)
    stretched = (arr - min_vals) / (max_vals - min_vals + 1e-5) * 255
    return [tuple(map(int, np.clip(color, 0, 255))) for color in stretched]

def create_smooth_gradient(palette, resolution=64, interpolation='cubic'):
    """
    Generates a smooth gradient from a list of RGB tuples using interpolation.

    Parameters:
        palette (list of tuple): A list of RGB tuples (each value 0–255) defining the base colors.
        resolution (int): Number of colors in the output gradient. Default is 256.
        interpolation (str): Interpolation method ('linear', 'quadratic', 'cubic', etc.).

    Returns:
        list of tuple: A list of RGB tuples (0–255) forming a smooth gradient.
    """
    if len(palette) < 2:
        raise ValueError("Palette must contain at least two colors for interpolation.")

    palette = np.array(palette, dtype=np.float32)
    indices = np.linspace(0, len(palette) - 1, num=len(palette))
    interp_func = interp1d(indices, palette, kind=interpolation, axis=0, fill_value="extrapolate")
    smooth_indices = np.linspace(0, len(palette) - 1, num=resolution)
    smooth_gradient = interp_func(smooth_indices)
    smooth = [tuple(map(int, np.clip(color, 0, 255))) for color in smooth_gradient]
    smooth = apply_gamma_correction(smooth)
    smooth = stretch_contrast(smooth)
    return smooth


# Define base palettes
base_palettes = {
    "Fire": create_smooth_gradient([
        (0, 0, 0), (255, 0, 0), (255, 85, 0), (255, 170, 0),
        (255, 255, 0), (255, 255, 85), (255, 255, 170)]),

    "Ocean": create_smooth_gradient([
        (0, 0, 0), (0, 32, 64), (0, 64, 128), (0, 96, 192),
        (0, 128, 255), (64, 160, 255), (128, 192, 255)]),

    "Classic": create_smooth_gradient([
        (0, 0, 0), (66, 30, 15), (25, 7, 26), (9, 1, 47), (4, 4, 73),
        (0, 7, 100), (12, 44, 138), (24, 82, 177), (57, 125, 209),
        (134, 181, 229), (211, 236, 248), (241, 233, 191), (248, 201, 95),
        (255, 170, 0), (204, 128, 0), (153, 87, 0), (106, 52, 3)]),

    "NeonGlow": create_smooth_gradient([
        (0, 0, 0),
        (0, 255, 255),
        (255, 0, 255),
        (255, 255, 0),
        (255, 255, 255)]),

    "IceFire": create_smooth_gradient([
        (0, 0, 0),
        (0, 0, 128),
        (0, 128, 255),
        (255, 255, 255),
        (255, 128, 0),
        (255, 0, 0),
        (0, 0, 0)]),


    "Viridis": create_smooth_gradient([
        (68, 1, 84),
        (59, 82, 139),
        (33, 145, 140),
        (94, 201, 98),
        (253, 231, 37)]),


    "Sunset": create_smooth_gradient([
        (0, 0, 0),
        (44, 0, 44),
        (128, 0, 64),
        (255, 94, 77),
        (255, 195, 113),
        (255, 255, 204)]),


    "Aurora": create_smooth_gradient([
        (0, 0, 0),
        (10, 10, 50),
        (0, 255, 128),
        (0, 128, 255),
        (255, 0, 255),
        (255, 255, 255)]),


    "Grayscale": create_smooth_gradient([
        (0, 0, 0),
        (32, 32, 32),
        (64, 64, 64),
        (96, 96, 96),
        (128, 128, 128),
        (160, 160, 160),
        (192, 192, 192),
        (224, 224, 224),
        (255, 255, 255)]),


    "InvertedGrayscale": create_smooth_gradient([
        (255, 255, 255),
        (224, 224, 224),
        (192, 192, 192),
        (160, 160, 160),
        (128, 128, 128),
        (96, 96, 96),
        (64, 64, 64),
        (32, 32, 32),
        (0, 0, 0)])

}

# Export palettes dictionary
keys = list(base_palettes.keys())
keys.sort()
palettes = {i: base_palettes[i] for i in keys}
