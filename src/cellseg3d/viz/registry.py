from typing import Callable, Dict, Optional

from ..core.types import LayerSpec, SegmentResult, Spacing

LayerBuilder = Callable[[SegmentResult, Spacing], Optional[LayerSpec]]
REGISTRY: Dict[str, LayerBuilder] = {}


def register(key: str):
    def _wrap(fn: LayerBuilder) -> LayerBuilder:
        REGISTRY[key] = fn
        return fn

    return _wrap


@register("ch0")
def _layer_ch0(res: SegmentResult, spacing: Spacing):
    arr = res.debug.get("vol_denoised")
    if arr is None:
        return None
    return LayerSpec("cells (ch0)", "image", arr, {"colormap": "gray", "opacity": 0.7})


@register("ch1")
def _layer_ch1(res: SegmentResult, spacing: Spacing):
    arr = res.debug.get("tissue")  # stored by pipeline if available
    if arr is None:
        return None
    return LayerSpec("tissue (ch1)", "image", arr, {"blending": "additive", "opacity": 0.5, "colormap": "magenta"})


@register("labels")
def _layer_labels(res: SegmentResult, spacing: Spacing):
    return LayerSpec("labels", "labels", res.labels, {"blending": "translucent"})


@register("centroids")
def _layer_points(res: SegmentResult, spacing: Spacing):
    pts = res.artifacts.get("centroids_vox")
    if pts is None:
        return None
    return LayerSpec("centroids", "points", pts, {"face_color": "yellow"})


@register("density_prob")
def _layer_density(res: SegmentResult, spacing: Spacing):
    arr = res.debug.get("density_prob")
    if arr is None:
        return None
    return LayerSpec("density_prob", "image", arr, {"colormap": "magma", "opacity": 0.6})
