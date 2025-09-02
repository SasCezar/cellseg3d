from typing import Tuple, Optional

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import (
    threshold_otsu,
    threshold_yen,
    threshold_li,
    threshold_triangle,
    gaussian,
)
from skimage.measure import label as cc_label
from skimage.morphology import (
    ball,
    closing,
    opening,
    remove_small_objects,
    binary_opening,
    binary_closing,
    h_maxima,
)
from skimage.segmentation import watershed, relabel_sequential

# NOTE: we keep type imports loose to avoid tight coupling to enums.
# If you already have enums in settings.py, this still works.
from .settings import SegmentationCfg  # your existing Pydantic config model

# -----------------------------
# Single place to control dtype
# -----------------------------
IMAGE_DTYPE = np.float16  # <--- flip here if you want float32 again
DIST_DTYPE = np.float32  # keep distance math stable

Spacing = Tuple[float, float, float]  # (dz, dy, dx)


# -----------------------------
# Utilities
# -----------------------------
def _robust_normalize(x: np.ndarray, p_lo=1.0, p_hi=99.0) -> np.ndarray:
    lo, hi = np.percentile(x, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(x, dtype=IMAGE_DTYPE)
    y = np.clip((x - lo) / (hi - lo), 0, 1, out=None)
    return y.astype(IMAGE_DTYPE, copy=False)


def _maybe_denoise(vol: np.ndarray, sigma: float, median_size: int) -> np.ndarray:
    y = vol
    if sigma and sigma > 0:
        y = gaussian(y, sigma=float(sigma), preserve_range=True)
        y = y.astype(IMAGE_DTYPE, copy=False)
    if median_size and median_size > 1:
        y = ndi.median_filter(y, size=int(median_size))
        y = y.astype(IMAGE_DTYPE, copy=False)
    return y


def _threshold_scalar(vol: np.ndarray, method: str, percentile: float) -> float:
    m = method.lower()
    if m == "otsu":
        return float(threshold_otsu(vol))
    if m == "yen":
        return float(threshold_yen(vol))
    if m == "li":
        return float(threshold_li(vol))
    if m == "triangle":
        return float(threshold_triangle(vol))
    if m == "percentile":
        pc = float(np.clip(percentile, 0, 100))
        return float(np.percentile(vol, pc))
    # fallback
    return float(threshold_otsu(vol))


def _postprocess_binary(
    bw: np.ndarray,
    min_voxels: int,
    post_open_radius: int,
    post_close_radius: int,
) -> np.ndarray:
    if post_open_radius and post_open_radius > 0:
        bw = binary_opening(bw, ball(int(post_open_radius)))
    if post_close_radius and post_close_radius > 0:
        bw = binary_closing(bw, ball(int(post_close_radius)))
    if min_voxels and min_voxels > 1:
        bw = remove_small_objects(bw, min_size=int(min_voxels))
    return bw


def _anisotropic_distance(bw: np.ndarray, spacing_um: Spacing) -> np.ndarray:
    dz, dy, dx = spacing_um
    dist = ndi.distance_transform_edt(bw, sampling=(dz, dy, dx))
    return dist.astype(DIST_DTYPE, copy=False)


def _seed_markers(
    bw: npt.NDArray[np.bool_],
    distance: npt.NDArray[np.float32],
    cfg: SegmentationCfg,
    spacing_um: Spacing,
) -> np.ndarray:
    method = getattr(cfg.watershed, "method", "hmax")
    m = (method.value if hasattr(method, "value") else str(method)).lower()

    if m == "hmax":
        dz, dy, dx = spacing_um
        mu_per_vox = float(dz + dy + dx) / 3.0
        h_val = float(getattr(cfg.watershed, "h", 1.0)) * mu_per_vox
        seeds = h_maxima(distance, h=h_val)
        markers, _ = ndi.label(seeds)
        return markers

    # peaks
    fp = int(max(1, getattr(cfg.watershed, "footprint", 3)))
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
    return markers


def _apply_tissue_prior(
    elevation: np.ndarray,  # elevation for watershed (low = basins)
    tissue: Optional[np.ndarray],  # channel-1 volume
    *,
    mode: str,  # "normalize" | "threshold"
    weight: float,
    norm_clip: tuple[float, float],
    thr_method: str,
    thr_percentile: float,
) -> np.ndarray:
    if tissue is None or weight <= 0:
        return elevation

    if mode == "normalize":
        t_norm = _robust_normalize(
            tissue.astype(IMAGE_DTYPE, copy=False),
            p_lo=float(norm_clip[0]),
            p_hi=float(norm_clip[1]),
        )
        return (elevation.astype(DIST_DTYPE, copy=False) + weight * t_norm.astype(DIST_DTYPE)).astype(
            DIST_DTYPE, copy=False
        )

    if mode == "threshold":
        # Build a 'hard' ridge: add weight where tissue > threshold
        t = tissue.astype(IMAGE_DTYPE, copy=False)
        t_thr = _threshold_scalar(t, thr_method, thr_percentile)
        ridge = (t > t_thr).astype(DIST_DTYPE, copy=False)
        return (elevation.astype(DIST_DTYPE, copy=False) + weight * ridge).astype(DIST_DTYPE, copy=False)

    # unknown mode -> no-op
    return elevation


# -----------------------------
# Public API (single orchestrator)
# -----------------------------
def segment_3d(
    volume_main: npt.NDArray[np.generic],
    spacing_um: Spacing,
    cfg: SegmentationCfg,
    *,
    volume_tissue: Optional[np.ndarray] = None,  # pass channel-1 here if available
) -> tuple[np.ndarray, np.ndarray]:
    """
    High-level segmentation:
      1) preprocess (denoise)
      2) threshold (configurable)
      3) morphology + small object removal
      4) optional watershed with anisotropic distance
         (optionally steered by a tissue channel prior in 'normalize' or 'threshold' mode)
    Returns: (binary_mask, labels)
    """

    # --- 1) Preprocess ---
    vol = volume_main.astype(IMAGE_DTYPE, copy=False)
    denoise_sigma = float(getattr(cfg, "denoise_sigma", 0.0))
    median_size = int(getattr(cfg, "median_size", 0))
    vol = _maybe_denoise(vol, sigma=denoise_sigma, median_size=median_size)

    # --- 2) Threshold ---
    thr_cfg = getattr(cfg, "threshold", None)
    thr_method = "otsu"
    thr_percentile = 99.0
    if thr_cfg is not None:
        thr_method = str(getattr(thr_cfg, "method", "otsu"))
        thr_percentile = float(getattr(thr_cfg, "percentile", 99.0))

    th = _threshold_scalar(vol, method=thr_method, percentile=thr_percentile)
    bw = vol > th

    # --- 3) Morphology (pre-watershed clean up) ---
    # primary open/close
    if getattr(cfg.morphology, "open_radius", 0):
        bw = opening(bw, ball(int(cfg.morphology.open_radius)))
    if getattr(cfg.morphology, "close_radius", 0):
        bw = closing(bw, ball(int(cfg.morphology.close_radius)))

    # speckle cleanup + optional extra open/close
    post_open_radius = int(getattr(cfg, "post_open_radius", 0))
    post_close_radius = int(getattr(cfg, "post_close_radius", 0))
    bw = _postprocess_binary(
        bw,
        min_voxels=int(getattr(cfg, "min_voxels", 50)),
        post_open_radius=post_open_radius,
        post_close_radius=post_close_radius,
    )

    # Early exit if watershed disabled or nothing to do
    labels = cc_label(bw, connectivity=3)
    ws_enabled = bool(getattr(cfg.watershed, "enabled", True))
    if (not ws_enabled) or (not np.any(bw)):
        return bw, labels

    # --- 4) Watershed splitting ---
    distance = _anisotropic_distance(bw, spacing_um)
    elevation = -distance  # basins at object centers

    # Optional tissue prior
    tp_cfg = getattr(cfg, "tissue_prior", None)
    if tp_cfg is not None and bool(getattr(tp_cfg, "enabled", False)):
        mode = str(getattr(tp_cfg, "mode", "normalize")).lower()
        weight = float(getattr(tp_cfg, "weight", 0.5))
        norm_clip = tuple(getattr(tp_cfg, "norm_clip", (1.0, 99.0)))
        t_thr_method = str(getattr(tp_cfg, "threshold_method", "yen"))
        t_thr_percentile = float(getattr(tp_cfg, "threshold_percentile", 95.0))

        elevation = _apply_tissue_prior(
            elevation=elevation,
            tissue=volume_tissue,
            mode=mode,
            weight=weight,
            norm_clip=norm_clip,
            thr_method=t_thr_method,
            thr_percentile=t_thr_percentile,
        )

    markers = _seed_markers(bw, distance, cfg, spacing_um)

    labels = watershed(
        elevation,  # negative distance (+ optional prior)
        markers=markers,
        mask=bw,
        compactness=float(getattr(cfg.watershed, "compactness", 0.0)),
    )

    # Final small basin cleanup & relabel
    min_vox = int(getattr(cfg, "min_voxels", 50))
    if min_vox > 1:
        sizes = np.bincount(labels.ravel())
        small = np.where(sizes < min_vox)[0]
        small = small[small != 0]
        if small.size:
            labels[np.isin(labels, small)] = 0
        labels, _, _ = relabel_sequential(labels)

    return bw, labels
