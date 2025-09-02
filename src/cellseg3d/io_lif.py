from pathlib import Path
from typing import Tuple

import numpy as np
from readlif.reader import LifFile



def iter_series_images(lif_path: Path):
    lif = LifFile(str(lif_path))
    for img in lif.get_iter_image():
        yield img


def spacing_um_from_img(img) -> Tuple[float | None, float | None, float | None]:
    """Return (dz, dy, dx) in µm, or None if missing."""
    # img.scale: (scale_x_px_per_um, scale_y_px_per_um, scale_z_px_per_um, scale_t_images_per_sec_or_None)
    sx_px_per_um, sy_px_per_um, sz_px_per_um, _ = img.scale

    def inv_or_none(v):
        try:
            return 1.0 / float(v) if v and float(v) > 0 else None
        except Exception:
            return None

    dx_um = inv_or_none(sx_px_per_um)
    dy_um = inv_or_none(sy_px_per_um)
    dz_um = inv_or_none(sz_px_per_um)
    return dz_um, dy_um, dx_um


def load_stack_zyx(img, t: int, c: int) -> np.ndarray:
    """Return (Z, Y, X) 3D stack for given time & channel."""
    t = int(np.clip(t, 0, int(img.dims.t) - 1))
    c = int(np.clip(c, 0, int(img.channels) - 1))
    planes = [np.array(pil_im) for pil_im in img.get_iter_z(t=t, c=c, m=0)]
    if not planes:
        raise ValueError("No z-planes found for this series/channel/timepoint.")
    return np.stack(planes, axis=0)
