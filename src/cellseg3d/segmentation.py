
from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu, threshold_yen, threshold_li, threshold_triangle
from skimage.measure import label as cc_label
from skimage.morphology import ball, closing, opening, remove_small_objects, binary_opening, binary_closing, h_maxima
from skimage.segmentation import watershed, relabel_sequential
from enum import Enum
from skimage.restoration import denoise_tv_chambolle, denoise_nl_means, estimate_sigma, denoise_bilateral


from .settings import SegmentationCfg

# ---- central numeric control ----
IMAGE_DTYPE = np.float32
DIST_DTYPE = np.float32

Spacing = Tuple[float, float, float]  # (dz, dy, dx)


class DenoiseMethod(str, Enum):
    none = "none"
    gaussian = "gaussian"
    tv = "tv"
    nlm = "nlm"
    bilateral = "bilateral"
    wavelet = "wavelet"


@dataclass
class SegmentResult:
    bw: np.ndarray
    labels: np.ndarray
    debug: Dict[str, np.ndarray | float]  # intermediates and scalars


# ---------- helpers (unchanged from earlier refactor, trimmed for brevity) ----------
def _robust_normalize(x: np.ndarray, p_lo=1.0, p_hi=99.0) -> np.ndarray:
    lo, hi = np.percentile(x, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(x, dtype=IMAGE_DTYPE)
    y = np.clip((x - lo) / (hi - lo), 0, 1, out=None)
    return y.astype(IMAGE_DTYPE, copy=False)


def _maybe_denoise(vol, method: str, params: dict, spacing_um):
    m = (method or "none").lower()
    if m == "none":
        return vol
    if m == "gaussian":
        from scipy.ndimage import gaussian_filter

        dy, dx = spacing_um[1], spacing_um[2]
        dz = spacing_um[0]
        tgt = float(params.get("target_um", 0.6))
        sig = (tgt / dz, tgt / dy, tgt / dx)  # physical 0.6 µm
        vol32 = vol.astype(np.float32, copy=False)  # <--- cast up
        den = gaussian_filter(vol32, sigma=sig)
        return den.astype(vol.dtype, copy=False)  # <--- cast back to float16
    if m == "tv":
        w = float(params.get("weight", 0.05))
        return denoise_tv_chambolle(vol, weight=w, channel_axis=None).astype(vol.dtype, copy=False)
    if m == "nlm":
        sigma_est = estimate_sigma(vol, channel_axis=None)
        h = float(params.get("h", 0.8 * sigma_est))
        ps = int(params.get("patch_size", 3))
        pd = int(params.get("patch_distance", 5))
        return denoise_nl_means(
            vol, h=h, patch_size=ps, patch_distance=pd, fast_mode=True, channel_axis=None, preserve_range=True
        ).astype(vol.dtype, copy=False)
    if m == "bilateral":
        sc = float(params.get("sigma_color", 0.1))
        ss = float(params.get("sigma_spatial", 1.5))
        return denoise_bilateral(vol, sigma_color=sc, sigma_spatial=ss, channel_axis=None).astype(vol.dtype, copy=False)
    if m == "wavelet":
        from skimage.restoration import denoise_wavelet

        return denoise_wavelet(
            vol,
            method=params.get("method", "BayesShrink"),
            mode=params.get("mode", "soft"),
            rescale_sigma=True,
            channel_axis=None,
        ).astype(vol.dtype, copy=False)
    return vol


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
        return float(np.percentile(vol, float(np.clip(percentile, 0, 100))))
    return float(threshold_otsu(vol))


def _postprocess_binary(bw: np.ndarray, min_voxels: int, post_open_radius: int, post_close_radius: int) -> np.ndarray:
    if post_open_radius and post_open_radius > 0:
        bw = binary_opening(bw, ball(int(post_open_radius)))
    if post_close_radius and post_close_radius > 0:
        bw = binary_closing(bw, ball(int(post_close_radius)))
    if min_voxels and min_voxels > 1:
        bw = remove_small_objects(bw, min_size=int(min_voxels))
    return bw


def _anisotropic_distance(bw: np.ndarray, spacing_um: Spacing) -> np.ndarray:
    dz, dy, dx = spacing_um
    return ndi.distance_transform_edt(bw, sampling=(dz, dy, dx)).astype(DIST_DTYPE, copy=False)


def _seed_markers(
    bw: npt.NDArray[np.bool_], distance: npt.NDArray[np.float32], cfg: SegmentationCfg, spacing_um: Spacing
) -> np.ndarray:
    method = getattr(cfg.watershed, "method", "hmax")
    m = (method.value if hasattr(method, "value") else str(method)).lower()
    if m == "hmax":
        dz, dy, dx = spacing_um
        mu_per_vox = (dz + dy + dx) / 3.0
        seeds = h_maxima(distance, h=float(getattr(cfg.watershed, "h", 1.0)) * float(mu_per_vox))
        markers, _ = ndi.label(seeds)
        return markers

    fp = int(max(1, getattr(cfg.watershed, "footprint", 3)))
    coords = peak_local_max(distance, labels=bw, footprint=np.ones((fp, fp, fp), dtype=bool), exclude_border=False)
    seed_mask = np.zeros_like(bw, dtype=bool)
    if coords.size:
        seed_mask[tuple(coords.T)] = True
    markers, _ = ndi.label(seed_mask)

    return markers


def _apply_tissue_prior(
    elevation: np.ndarray,
    tissue: Optional[np.ndarray],
    *,
    mode: str,
    weight: float,
    norm_clip: tuple[float, float],
    thr_method: str,
    thr_percentile: float,
) -> np.ndarray:
    if tissue is None or weight <= 0:
        return elevation
    if mode == "normalize":
        t_norm = _robust_normalize(
            tissue.astype(IMAGE_DTYPE, copy=False), p_lo=float(norm_clip[0]), p_hi=float(norm_clip[1])
        )
        return (elevation.astype(DIST_DTYPE) + weight * t_norm.astype(DIST_DTYPE)).astype(DIST_DTYPE)
    if mode == "threshold":
        t = tissue.astype(IMAGE_DTYPE, copy=False)
        t_thr = _threshold_scalar(t, thr_method, thr_percentile)
        ridge = (t > t_thr).astype(DIST_DTYPE, copy=False)
        return (elevation.astype(DIST_DTYPE) + weight * ridge).astype(DIST_DTYPE)
    return elevation


# ----------------- public API: return SegmentResult with debug dict -----------------
def segment_3d(
    volume_main: npt.NDArray[np.generic],
    spacing_um: Spacing,
    cfg: SegmentationCfg,
    *,
    volume_tissue: Optional[np.ndarray] = None,
) -> SegmentResult:
    dbg: Dict[str, np.ndarray | float] = {}

    # 1) preprocess
    vol = volume_main.astype(IMAGE_DTYPE, copy=False)
    vol = _maybe_denoise(
        vol,
        method=getattr(cfg, "denoise_method", "none"),
        params=getattr(cfg, "denoise_params", {}),
        spacing_um=spacing_um,
    )
    dbg["vol_denoised"] = vol

    # 2) threshold
    thr_cfg = getattr(cfg, "threshold", None)
    t_method = "otsu" if thr_cfg is None else str(getattr(thr_cfg, "method", "otsu"))
    t_pc = 99.0 if thr_cfg is None else float(getattr(thr_cfg, "percentile", 99.0))
    th = _threshold_scalar(vol, t_method, t_pc)
    dbg["th_value"] = float(th)
    bw = vol > th
    dbg["bw_raw"] = bw.astype(np.uint8, copy=False)

    # 3) morphology pre-watershed
    if getattr(cfg.morphology, "open_radius", 0):
        bw = opening(bw, ball(int(cfg.morphology.open_radius)))
    if getattr(cfg.morphology, "close_radius", 0):
        bw = closing(bw, ball(int(cfg.morphology.close_radius)))
    dbg["bw_morph"] = bw.astype(np.uint8, copy=False)

    # post cleanup
    bw = _postprocess_binary(
        bw,
        min_voxels=int(getattr(cfg, "min_voxels", 50)),
        post_open_radius=int(getattr(cfg, "post_open_radius", 0)),
        post_close_radius=int(getattr(cfg, "post_close_radius", 0)),
    )
    dbg["bw_post"] = bw.astype(np.uint8, copy=False)

    # early return if watershed disabled
    labels = cc_label(bw, connectivity=3)
    if not bool(getattr(cfg.watershed, "enabled", True)) or not np.any(bw):
        return SegmentResult(bw=bw, labels=labels, debug=dbg)

    # 4) watershed
    distance = _anisotropic_distance(bw, spacing_um)
    elevation = -distance
    dbg["distance"] = distance
    dbg["elevation_pre_prior"] = elevation

    tp = getattr(cfg, "tissue_prior", None)
    if tp is not None and bool(getattr(tp, "enabled", False)):
        elevation = _apply_tissue_prior(
            elevation=elevation,
            tissue=volume_tissue,
            mode=str(getattr(tp, "mode", "normalize")).lower(),
            weight=float(getattr(tp, "weight", 0.5)),
            norm_clip=tuple(getattr(tp, "norm_clip", (1.0, 99.0))),
            thr_method=str(getattr(tp, "threshold_method", "yen")),
            thr_percentile=float(getattr(tp, "threshold_percentile", 95.0)),
        )
    dbg["elevation"] = elevation

    markers = _seed_markers(bw, distance, cfg, spacing_um)
    dbg["markers"] = markers.astype(np.int32, copy=False)

    labels = watershed(
        elevation, markers=markers, mask=bw, compactness=float(getattr(cfg.watershed, "compactness", 0.0))
    )

    # final cleanup
    min_vox = int(getattr(cfg, "min_voxels", 50))
    if min_vox > 1:
        sizes = np.bincount(labels.ravel())
        small = np.where(sizes < min_vox)[0]
        small = small[small != 0]
        if small.size:
            labels[np.isin(labels, small)] = 0
        labels, _, _ = relabel_sequential(labels)

    dbg["labels"] = labels.astype(np.int32, copy=False)
    return SegmentResult(bw=bw, labels=labels, debug=dbg)
