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


def edge():
    exterior = [(1, 1, 1)] * 50
    interior = [(1, 1, 1)] * 5
    gray_area = [(1 - i / 44,) * 3 for i in range(45)]
    return denormalize(exterior + gray_area + interior)


def gen_gradient(colors, interpolation="linear", num_colors=256):
    gradient = make_gradient(colors, interpolation=interpolation)
    return denormalize([gradient(i / num_colors) for i in range(num_colors)])
