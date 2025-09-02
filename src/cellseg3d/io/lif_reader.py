from pathlib import Path
from typing import Tuple

import numpy as np
from readlif.reader import LifFile


def iter_series_images(lif_path: Path):
    lif = LifFile(str(lif_path))
    yield from lif.get_iter_image()


def spacing_um_from_img(img) -> Tuple[float | None, float | None, float | None]:
    sx, sy, sz, _ = img.scale

    def inv(v):
        try:
            return 1.0 / float(v) if v and float(v) > 0 else None
        except Exception:
            return None

    return inv(sz), inv(sy), inv(sx)


def load_stack_zyx(img, t: int, c: int) -> np.ndarray:
    t = int(np.clip(t, 0, int(img.dims.t) - 1))
    c = int(np.clip(c, 0, int(img.channels) - 1))
    z = int(img.dims.z)
    it = img.get_iter_z(t=t, c=c, m=0)
    first = np.asarray(next(it))
    vol = np.empty((z, *first.shape), dtype=first.dtype)
    vol[0] = first
    for zi, plane in enumerate(it, start=1):
        if zi >= z:
            break
        vol[zi] = np.asarray(plane)
    return vol
