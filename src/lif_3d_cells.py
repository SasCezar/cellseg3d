from pathlib import Path
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel, Field, ValidationError, PositiveInt, field_validator

from readlif.reader import LifFile
from skimage.filters import threshold_otsu
from skimage.morphology import ball, opening, closing, remove_small_objects
from skimage.measure import label, regionprops


from scipy import ndimage as ndi
from skimage.morphology import h_maxima
from skimage.feature import peak_local_max
from skimage.segmentation import watershed, relabel_sequential
from skimage.measure import label as cc_label

import napari


# Settings
class Spacing(BaseModel):
    dz: float = Field(..., gt=0)
    dy: float = Field(..., gt=0)
    dx: float = Field(..., gt=0)


class Settings(BaseModel):
    lif_path: Path
    valid_idx: list[int]
    preferred_channel: int = 0
    preferred_timepoint: int = 0
    min_voxels: PositiveInt = 50
    # morph_radius: PositiveInt = 1
    default_spacing_um: Spacing
    output_dir: Optional[Path] = None

    open_radius: int = 1
    close_radius: int = 0

    watershed_enabled: bool = True
    watershed_method: str = "hmax"  # "hmax" | "peaks"
    watershed_h: float = 1.0
    watershed_footprint: int = 3
    watershed_compactness: float = 0.0

    visualize: bool = True
    show_labels: bool = True
    points_size: int = Field(6, gt=0)

    render_3d: bool = True
    volume_mode: str = "attenuated_mip"  # 'translucent' | 'mip' | 'attenuated_mip' | 'additive'
    make_surface: bool = True
    surface_smoothing: float = 1.0
    surface_level: float = 0.5
    downsample_for_surface: int = 1

    @field_validator("lif_path")
    @classmethod
    def lif_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"LIF file not found: {v}")
        return v

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)


# Helper functions
def ensure_output_dir(cfg: Settings, lif_path: Path) -> Path:
    if cfg.output_dir:
        out = cfg.output_dir
    else:
        out = lif_path.with_suffix("")  # folder named after lif
    out.mkdir(exist_ok=True, parents=True)
    return out


def get_spacing_um(img) -> Tuple[float | None, float | None, float | None]:
    """
    Your reader exposes `image.scale` as px/µm for (x, y, z) and images/sec for t.
    We invert to get µm/px for spatial axes. If any missing, fall back to defaults.
    """
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
    """
    Build a 3D stack (Z,Y,X) using your LifImage API.
    Uses get_iter_z(t=?, c=?, m=0) to collect PIL frames, converts to numpy.
    """
    # guard indices
    t = int(np.clip(t, 0, int(img.dims.t) - 1))
    c = int(np.clip(c, 0, int(img.channels) - 1))

    # collect z-planes
    planes = []
    for pil_im in img.get_iter_z(t=t, c=c, m=0):
        planes.append(np.array(pil_im))

    if not planes:
        raise ValueError("No z-planes found for this series/channel/timepoint.")

    volume = np.stack(planes, axis=0)  # (Z, Y, X)
    return volume


def segment_3d(
    volume: np.ndarray,
    min_voxels: int,
    open_radius: int,
    close_radius: int,
    *,
    watershed_enabled: bool,
    watershed_method: str,
    watershed_h: float,
    watershed_footprint: int,
    watershed_compactness: float,
    spacing_um: tuple[float, float, float],  # (dz, dy, dx)
) -> tuple[np.ndarray, np.ndarray]:
    """
    Otsu -> (optional) opening/closing -> remove small -> (optional) watershed split.
    Returns: (binary_mask, labels)
    """
    vol = volume.astype(np.float16)
    th = threshold_otsu(vol)
    bw = vol > th

    if open_radius:
        bw = opening(bw, ball(open_radius))
    if close_radius:
        bw = closing(bw, ball(close_radius))

    # remove tiny specks before watershed
    bw = remove_small_objects(bw, min_voxels)

    # Default: simple connected components (if watershed disabled)
    labels = cc_label(bw, connectivity=3)

    if not watershed_enabled or not np.any(bw):
        return bw, labels

    # Use anisotropic distance (in µm) so splitting respects voxel spacing
    dz, dy, dx = spacing_um
    distance = ndi.distance_transform_edt(bw, sampling=(dz, dy, dx))

    # Seed markers
    if watershed_method.lower() == "hmax":
        # h in *distance units*; we accept voxels in YAML, convert to µm
        # Convert "voxels" to µm using geometric mean of spacings (rough, but practical)
        # or just use dz/dy/dx ≈ 1 if nearly isotropic.
        # Here we keep it simple: treat h as voxels of an *isotropic* grid:
        # approximate µm-per-voxel as mean spacing.
        mu_per_vox = float(dz + dy + dx) / 3.0
        seeds = h_maxima(distance, h=float(watershed_h) * mu_per_vox)
        markers, _ = ndi.label(seeds)
    elif watershed_method.lower() == "peaks":
        fp = int(max(1, watershed_footprint))
        coords = peak_local_max(
            distance,
            labels=bw,
            footprint=np.ones((fp, fp, fp), dtype=bool),
            exclude_border=False,
        )
        seed_mask = np.zeros_like(bw, dtype=bool)
        if coords.size:
            seed_mask[tuple(coords.T)] = True
        markers, _ = ndi.label(seed_mask)
    else:
        raise ValueError("watershed_method must be 'hmax' or 'peaks'.")

    # Watershed on inverted distance so peaks become basins
    labels = watershed(
        -distance,
        markers=markers,
        mask=bw,
        compactness=float(watershed_compactness),
    )

    # Remove tiny basins post-split and relabel consecutively
    if min_voxels > 1:
        sizes = np.bincount(labels.ravel())
        small = np.where(sizes < min_voxels)[0]
        small = small[small != 0]
        if small.size:
            labels[np.isin(labels, small)] = 0
        labels, _, _ = relabel_sequential(labels)

    return bw, labels


def centroids_from_mask(bw: np.ndarray, spacing_um: Tuple[float, float, float]) -> tuple[pd.DataFrame, np.ndarray]:
    dz, dy, dx = spacing_um
    lab = label(bw, connectivity=3)
    props = regionprops(lab)

    rows = []
    for p in props:
        zc, yc, xc = p.centroid  # voxel coords
        rows.append(
            dict(
                label=int(p.label),
                z_vox=float(zc),
                y_vox=float(yc),
                x_vox=float(xc),
                z_um=float(zc * dz),
                y_um=float(yc * dy),
                x_um=float(xc * dx),
                volume_vox=int(p.area),
            )
        )
    return pd.DataFrame(rows), lab


def visualize(
    volume: np.ndarray,
    labels: Optional[np.ndarray],
    points_zyx: np.ndarray,
    title: str,
    cfg: Settings,
):
    if napari is None:
        print("napari not available; skipping visualization.")
        return

    viewer = napari.Viewer(title=title, ndisplay=3 if cfg.render_3d else 2)

    #  Image layer (handle 2D vs 3D cleanly)
    img_kwargs = dict(
        name="volume",
        blending="translucent",
        contrast_limits=(np.percentile(volume, 1), np.percentile(volume, 99)),
    )
    if cfg.render_3d:
        img_kwargs["rendering"] = cfg.volume_mode  # 'mip'|'translucent'|'attenuated_mip'|'additive'
        img_kwargs["depiction"] = "volume"  # 3D
    else:
        img_kwargs["rendering"] = "mip"  # harmless in 2D
        # do NOT pass 'depiction' in 2D (or set to 'plane' if you prefer)
        # img_kwargs["depiction"] = "plane"

    vol_layer = viewer.add_image(volume, **img_kwargs)

    #  Labels
    lbl_layer = None
    if cfg.show_labels and labels is not None:
        lbl_layer = viewer.add_labels(labels, name="labels", blending="translucent")

    #  Points
    if points_zyx.size:
        viewer.add_points(
            points_zyx,  # (z, y, x)
            name="centroids",
            size=cfg.points_size,
            face_color="yellow",
            # no 'ndim' needed; napari infers from data (3D points work fine in 2D view with a z slider)
        )

    # Optional surface mesh (3D only)
    if cfg.render_3d and cfg.make_surface and labels is not None:
        try:
            from skimage.filters import gaussian
            from skimage.measure import marching_cubes

            surf_src = labels.astype(float)
            if cfg.surface_smoothing and cfg.surface_smoothing > 0:
                surf_src = gaussian(surf_src, sigma=cfg.surface_smoothing, preserve_range=True)

            if cfg.downsample_for_surface and cfg.downsample_for_surface > 1:
                s = int(cfg.downsample_for_surface)
                surf_src = surf_src[::s, ::s, ::s]
                voxel_scale = np.array([s, s, s], dtype=float)
            else:
                voxel_scale = np.array([1.0, 1.0, 1.0], dtype=float)

            verts, faces, normals, values = marching_cubes(surf_src, level=cfg.surface_level, allow_degenerate=False)
            verts = verts * voxel_scale

            viewer.add_surface(
                (verts, faces, values),
                name="labels_surface",
                shading="smooth",
                colormap="turbo",
                opacity=0.5,
            )
        except Exception as e:
            print(f"Surface extraction skipped: {e}")

    if cfg.render_3d:
        try:
            viewer.camera.angles = (30, 30, 0)
            viewer.camera.zoom = 0.8
        except Exception:
            pass

    napari.run()


def centroids_from_labels(labels: np.ndarray, spacing_um: tuple[float, float, float]) -> pd.DataFrame:
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


# Pipeline
def process(cfg: Settings):
    lif = LifFile(str(cfg.lif_path))
    out_dir = ensure_output_dir(cfg, cfg.lif_path)

    # iterate series
    for idx, img in enumerate(lif.get_iter_image(), start=1):
        if idx not in cfg.valid_idx:
            continue
        # dims are (x, y, z, t, m)
        z_dim = int(img.dims.z)
        t_dim = int(img.dims.t)
        c_dim = int(img.channels)

        if z_dim < 1:
            print(f"[{idx}] {img.name}: no Z dimension; skipping.")
            continue

        t = int(np.clip(cfg.preferred_timepoint, 0, max(0, t_dim - 1)))
        c = int(np.clip(cfg.preferred_channel, 0, max(0, c_dim - 1)))

        # 3D stack (Z,Y,X)
        volume = load_stack_zyx(img, t=t, c=c)

        # spacing in µm (fallbacks if missing)
        dz, dy, dx = get_spacing_um(img)
        dz = dz or cfg.default_spacing_um.dz
        dy = dy or cfg.default_spacing_um.dy
        dx = dx or cfg.default_spacing_um.dx
        spacing = (dz, dy, dx)

        # segment & centroids
        bw, lab = segment_3d(
            volume,
            cfg.min_voxels,
            cfg.open_radius,
            cfg.close_radius,
            watershed_enabled=cfg.watershed_enabled,
            watershed_method=cfg.watershed_method,
            watershed_h=cfg.watershed_h,
            watershed_footprint=cfg.watershed_footprint,
            watershed_compactness=cfg.watershed_compactness,
            spacing_um=spacing,  # (dz, dy, dx)
        )
        df = centroids_from_labels(lab, spacing)

        # save CSV
        series_name = getattr(img, "name", f"series_{idx}")
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(series_name))
        csv_path = out_dir / f"{idx:03d}_{safe}_centroids.csv"
        df.to_csv(csv_path, index=False)

        print(f"[{idx}] {series_name}: ZYX {volume.shape} | found {len(df)} cells | saved -> {csv_path}")

        # visualize
        if cfg.visualize:
            pts = df[["z_vox", "y_vox", "x_vox"]].to_numpy(dtype=np.float32) if len(df) else np.empty((0, 3))
            visualize(
                volume,
                lab if cfg.show_labels else None,
                pts,
                f"{idx:03d} - {series_name}",
                cfg,
            )


# Entry
if __name__ == "__main__":
    cfg_path = Path(__file__).with_name("config.yaml")
    try:
        settings = Settings.from_yaml(cfg_path)
    except FileNotFoundError:
        raise SystemExit(f"Config file not found: {cfg_path}")
    except ValidationError as e:
        raise SystemExit(f"Config validation error:\n{e}")

    process(settings)
