import numpy as np
from enums import ColoringMode
from coloring.base import ColoringStrategy

class SmoothEscapeColoring(ColoringStrategy):
    def apply(self, iter_buf: np.ndarray, mode: ColoringMode,
              exterior_palette: np.ndarray, interior_palette: np.ndarray,
              interior_color=(100, 100, 100)) -> np.ndarray:
        h, w = iter_buf.shape
        rgb = np.zeros((h, w, 3), dtype=np.uint8)
        ex_size = len(exterior_palette)

        interior_mask = iter_buf >= 0.999
        exterior_mask = ~interior_mask

        if mode in (ColoringMode.EXTERIOR, ColoringMode.HYBRID):
            vals = iter_buf[exterior_mask]
            idx_f = vals * (ex_size - 1)
            idx = np.clip(idx_f.astype(np.int32), 0, ex_size - 1)
            t = idx_f - idx
            idx_next = np.clip(idx + 1, 0, ex_size - 1)
            c0 = exterior_palette[idx]
            c1 = exterior_palette[idx_next]
            blended = (((1 - t)[:, None] * c0) + (t[:, None] * c1)).astype(np.uint8)
            rgb[exterior_mask] = blended

        if mode in (ColoringMode.INTERIOR, ColoringMode.HYBRID):
            if np.any(interior_mask):
                y, x = np.where(interior_mask)
                cx, cy = w / 2.0, h / 2.0
                d = np.sqrt((x - cx) ** 2 + (y - cy) ** 2)
                dn = d / d.max() if d.max() > 0 else d
                grad = (np.array(interior_color)[None, :] * (1 - dn[:, None])).astype(np.uint8)
                rgb[interior_mask] = grad
        return rgb
