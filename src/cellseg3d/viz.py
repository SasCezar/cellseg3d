# viz.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Dict, List, Literal

import numpy as np
import napari

from .settings import VisualizationCfg

LayerKind = Literal["image", "labels", "points"]


@dataclass
class LayerSpec:
    name: str
    kind: LayerKind
    data: np.ndarray
    kwargs: Dict[str, Any] | None = None  # visual kwargs per layer


# ---- Legend (name -> human description) ----
LAYER_LEGEND: dict[str, str] = {
    # base channels
    "cells (ch0)": "Raw image channel 0 (objects of interest).",
    "tissue (ch1)": "Raw image channel 1 (tissue/background) used as watershed prior.",
    # volumes (debug)
    "denoised (ch0)": "Channel 0 after configured denoising (tv/nlm/bilateral/...).",
    # progressive masks
    "bw_raw": "Binary mask right after thresholding the (denoised) channel 0.",
    "bw_morph": "Mask after morphological open/close.",
    "bw_post": "Mask after small-object removal (and post open/close).",
    # maps
    "distance": "Anisotropic distance transform (µm) of bw_post.",
    "elevation_pre_prior": "Elevation = -distance (before tissue prior).",
    "elevation": "Elevation used by watershed = -distance (+ tissue prior).",
    # seeds / results
    "labels": "Final segmentation labels (post-watershed & cleanup).",
    "centroids": "One point per label centroid (z,y,x in voxel units).",
}


def _default_img_kwargs(arr: np.ndarray, render_3d: bool, volume_mode: str) -> dict:
    v1, v99 = np.percentile(arr, (1, 99))
    kw = {"blending": "translucent", "contrast_limits": (float(v1), float(v99))}
    if render_3d:
        kw["rendering"] = volume_mode
        kw["depiction"] = "volume"
    else:
        kw["rendering"] = "mip"
    return kw


def _ensure_metadata(kwargs: dict, layer_name: str) -> dict:
    """Attach legend text into metadata['description'] without clobbering existing metadata."""
    desc = LAYER_LEGEND.get(layer_name)
    if desc:
        md = dict(kwargs.get("metadata") or {})
        md.setdefault("description", desc)
        kwargs["metadata"] = md
    return kwargs


def visualize_layers(
    layers: List[LayerSpec],
    title: str,
    viz_cfg: VisualizationCfg,
):
    if napari is None:
        print("napari not available; skipping visualization.")
        return

    vol_mode = viz_cfg.volume_mode.value if hasattr(viz_cfg.volume_mode, "value") else viz_cfg.volume_mode
    viewer = napari.Viewer(title=title, ndisplay=3 if viz_cfg.render_3d else 2)

    for spec in layers:
        kind = spec.kind
        data = spec.data
        kwargs = dict(spec.kwargs or {})

        if kind == "image":
            base = _default_img_kwargs(data, viz_cfg.render_3d, vol_mode)
            for k, v in base.items():
                kwargs.setdefault(k, v)
            kwargs = _ensure_metadata(kwargs, spec.name)
            viewer.add_image(data, name=spec.name, **kwargs)

        elif kind == "labels":
            base = {"blending": "translucent"}
            for k, v in base.items():
                kwargs.setdefault(k, v)
            kwargs = _ensure_metadata(kwargs, spec.name)
            viewer.add_labels(data, name=spec.name, **kwargs)

        elif kind == "points":
            base = {"size": viz_cfg.points_size, "face_color": "yellow"}
            for k, v in base.items():
                kwargs.setdefault(k, v)
            kwargs = _ensure_metadata(kwargs, spec.name)
            viewer.add_points(data, name=spec.name, **kwargs)

    napari.run()
