import numpy as np
import napari

from .settings import VisualizationCfg


def visualize(
    volume: np.ndarray,
    labels: np.ndarray | None,
    points_zyx: np.ndarray,
    title: str,
    viz_cfg: VisualizationCfg,
):
    if napari is None:
        print("napari not available; skipping visualization.")
        return

    viewer = napari.Viewer(title=title, ndisplay=3 if viz_cfg.render_3d else 2)

    img_kwargs = dict(
        name="volume",
        blending="translucent",
        contrast_limits=(np.percentile(volume, 1), np.percentile(volume, 99)),
    )
    if viz_cfg.render_3d:
        img_kwargs["rendering"] = viz_cfg.volume_mode.value
        img_kwargs["depiction"] = "volume"
    else:
        img_kwargs["rendering"] = "mip"

    viewer.add_image(volume, **img_kwargs)

    if viz_cfg.show_labels and labels is not None:
        viewer.add_labels(labels, name="labels", blending="translucent")

    if points_zyx.size:
        viewer.add_points(points_zyx, name="centroids", size=viz_cfg.points_size, face_color="yellow")

    if viz_cfg.render_3d and viz_cfg.surface.make and labels is not None:
        try:
            from skimage.filters import gaussian
            from skimage.measure import marching_cubes

            surf_src = labels.astype(float)
            if viz_cfg.surface.smoothing and viz_cfg.surface.smoothing > 0:
                surf_src = gaussian(surf_src, sigma=viz_cfg.surface.smoothing, preserve_range=True)

            if viz_cfg.surface.downsample and viz_cfg.surface.downsample > 1:
                s = int(viz_cfg.surface.downsample)
                surf_src = surf_src[::s, ::s, ::s]
                voxel_scale = np.array([s, s, s], dtype=float)
            else:
                voxel_scale = np.array([1.0, 1.0, 1.0], dtype=float)

            verts, faces, normals, values = marching_cubes(
                surf_src, level=viz_cfg.surface.level, allow_degenerate=False
            )
            verts = verts * voxel_scale
            viewer.add_surface(
                (verts, faces, values), name="labels_surface", shading="smooth", colormap="turbo", opacity=0.5
            )
        except Exception as e:
            print(f"Surface extraction skipped: {e}")

    napari.run()
