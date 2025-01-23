from scipy.interpolate import interp1d
import numpy as np


def denormalize(palette):
    return [
        tuple(int(channel*255) for channel in color)
        for color in palette
    ]


def make_gradient(colors, interpolation="linear"):
    X = [i / (len(colors) - 1) for i in range(len(colors))]
    Y = [[color[i] for color in colors] for i in range(3)]
    channels = [interp1d(X, y, kind=interpolation) for y in Y]
    return lambda x: [np.clip(channel(x), 0, 1) for channel in channels]


def create_smooth_gradient(palette, resolution=1000):
    indices = np.linspace(0, len(palette) - 1, len(palette))
    interp_func = interp1d(indices, palette, kind='cubic', axis=0, fill_value="extrapolate")
    smooth_gradient = interp_func(np.linspace(0, len(palette) - 1, resolution))
    return smooth_gradient.astype(int)


def edge():
    exterior = [(1, 1, 1)] * 50
    interior = [(1, 1, 1)] * 5
    gray_area = [(1 - i / 44,) * 3 for i in range(45)]
    return denormalize(exterior + gray_area + interior)


def gen_gradient(colors):
    grad = []
    for i in range(1, len(colors)):
        j = i - 1
        temp = get_color_gradient(list(colors[j]), list(colors[i]), 16)
        for val in temp:
            grad.append(hex_to_RGB(val))
    return grad


def hex_to_RGB(hex_str):
    """ #FFFFFF -> [255,255,255]"""
    #Pass 16 to the integer function for change of base
    return [int(hex_str[i:i+2], 16) for i in range(1,6,2)]


def rgb_to_hex(rgb):
    return '#{:02x}{:02x}{:02x}'.format(rgb[0], rgb[1], rgb[2])


def get_color_gradient(c1, c2, n):
    """
    Given two hex colors, returns a color gradient
    with n colors.
    """
    assert n > 1
    c1_rgb = np.array(c1)/255
    c2_rgb = np.array(c2)/255
    mix_pcts = [x/(n-1) for x in range(n)]
    rgb_colors = [((1-mix)*c1_rgb + (mix*c2_rgb)) for mix in mix_pcts]
    return ["#" + "".join([format(int(round(val*255)), "02x") for val in item]) for item in rgb_colors]
