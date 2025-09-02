from typing import List

import napari
import numpy as np

from ..core.types import LayerSpec


def _default_img_kwargs(arr: np.ndarray, render_3d: bool, volume_mode: str) -> dict:
    v1, v99 = np.percentile(arr, (1, 99))
    kw = {"blending": "translucent", "contrast_limits": (float(v1), float(v99))}
    if render_3d:
        kw["rendering"] = volume_mode
        kw["depiction"] = "volume"
    else:
        kw["rendering"] = "mip"
    return kw


def visualize_layers(layers: List[LayerSpec], title: str, *, render_3d: bool, volume_mode: str, points_size: int):
    v = napari.Viewer(title=title, ndisplay=3 if render_3d else 2)
    for spec in layers:
        k = spec.kind
        data = spec.data
        kwargs = dict(spec.kwargs or {})
        if k == "image":
            base = _default_img_kwargs(data, render_3d, volume_mode)
            for kk, vv in base.items():
                kwargs.setdefault(kk, vv)
            v.add_image(data, name=spec.name, **kwargs)
        elif k == "labels":
            kwargs.setdefault("blending", "translucent")
            v.add_labels(data, name=spec.name, **kwargs)
        elif k == "points":
            kwargs.setdefault("size", points_size)
            kwargs.setdefault("face_color", "yellow")
            v.add_points(data, name=spec.name, **kwargs)
    napari.run()
