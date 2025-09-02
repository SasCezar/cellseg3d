from __future__ import annotations
import numpy as np
import napari

from .settings import VisualizationCfg


def visualize(
    volume: np.ndarray,
    labels: np.ndarray | None,
    points_zyx: np.ndarray,
    title: str,
    viz_cfg: VisualizationCfg,
    *,
    volume_tissue: np.ndarray | None = None,  # <-- NEW
):
    if napari is None:
        print("napari not available; skipping visualization.")
        return

    viewer = napari.Viewer(title=title, ndisplay=3 if viz_cfg.render_3d else 2)

    # --- main image layer ---
    img_kwargs = dict(
        name="cells (ch0)",
        blending="translucent",
        contrast_limits=(np.percentile(volume, 1), np.percentile(volume, 99)),
    )
    if viz_cfg.render_3d:
        img_kwargs["rendering"] = (
            viz_cfg.volume_mode.value if hasattr(viz_cfg.volume_mode, "value") else viz_cfg.volume_mode
        )
        img_kwargs["depiction"] = "volume"
    else:
        img_kwargs["rendering"] = "mip"

    viewer.add_image(volume, **img_kwargs)

    # --- optional tissue overlay (ch1) ---
    if volume_tissue is not None:
        t_kwargs = dict(
            name="tissue (ch1)",
            blending="additive",  # additive looks great for overlays
            opacity=0.6,
            colormap="magenta",
            contrast_limits=(np.percentile(volume_tissue, 1), np.percentile(volume_tissue, 99)),
        )
        if viz_cfg.render_3d:
            t_kwargs["rendering"] = img_kwargs["rendering"]
            t_kwargs["depiction"] = "volume"
        else:
            t_kwargs["rendering"] = "mip"

        viewer.add_image(volume_tissue, **t_kwargs)

    # --- labels ---
    if viz_cfg.show_labels and labels is not None:
        viewer.add_labels(labels, name="labels", blending="translucent")

    # --- points ---
    if points_zyx.size:
        viewer.add_points(points_zyx, name="centroids", size=viz_cfg.points_size, face_color="yellow")

    napari.run()
