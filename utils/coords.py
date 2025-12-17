def fractal_to_image_coords(fx, fy, center_x, center_y, zoom, image_width,
                            image_height):
    px = int((fx - center_x) * zoom + image_width / 2)
    py = int((fy - center_y) * zoom + image_height / 2)
    return px, py


def image_to_fractal_coords(px, py, center_x, center_y, zoom, image_width,
                            image_height):
    fx = (px - image_width / 2) / zoom + center_x
    fy = (py - image_height / 2) / zoom + center_y
    return fx, fy

