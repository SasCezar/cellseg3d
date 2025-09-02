from typing import Tuple

import numpy as np
import pandas as pd
from skimage.measure import regionprops


def centroids_from_labels(labels: np.ndarray, spacing_um: Tuple[float, float, float]) -> pd.DataFrame:
    dz, dy, dx = spacing_um
    rows = []
    for p in regionprops(labels):
        zc, yc, xc = p.centroid
        rows.append(
            {
                "label": int(p.label),
                "z_vox": float(zc),
                "y_vox": float(yc),
                "x_vox": float(xc),
                "z_um": float(zc * dz),
                "y_um": float(yc * dy),
                "x_um": float(xc * dx),
                "volume_vox": int(p.area),
            }
        )
    return pd.DataFrame(rows)
