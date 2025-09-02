import logging
from typing import Optional

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_li, threshold_otsu, threshold_triangle, threshold_yen
from skimage.morphology import ball, binary_closing, binary_opening, remove_small_objects
from skimage.restoration import denoise_bilateral, denoise_nl_means, denoise_tv_chambolle, estimate_sigma
from skimage.segmentation import relabel_sequential, watershed

from ..core.debug import DebugStore
from ..core.types import SegmentResult, Spacing
from ..settings import SegmentationCfg

log = logging.getLogger("cellseg3d")

IMAGE_DTYPE = np.float16
DIST_DTYPE = np.float32


def _robust_normalize(x: np.ndarray, p_lo=1.0, p_hi=99.0) -> np.ndarray:
    lo, hi = np.percentile(x, [p_lo, p_hi])
    if hi <= lo:
        return np.zeros_like(x, dtype=IMAGE_DTYPE)
    y = np.clip((x - lo) / (hi - lo), 0, 1)
    return y.astype(IMAGE_DTYPE, copy=False)


def _denoise(vol: np.ndarray, cfg: SegmentationCfg, spacing: Spacing) -> np.ndarray:
    m = (cfg.denoise_method or "none").lower()
    if m == "none":
        return vol
    if m == "gaussian":
        from scipy.ndimage import gaussian_filter

        dz, dy, dx = spacing
        tgt = float(cfg.denoise_params.get("target_um", 0.6))
        sig = (tgt / dz, tgt / dy, tgt / dx)
        v32 = vol.astype(np.float32, copy=False)
        out = gaussian_filter(v32, sigma=sig)
        return out.astype(vol.dtype, copy=False)
    if m == "tv":
        w = float(cfg.denoise_params.get("weight", 0.05))
        return denoise_tv_chambolle(vol, weight=w, channel_axis=None).astype(vol.dtype, copy=False)
    if m == "nlm":
        sigma_est = estimate_sigma(vol, channel_axis=None)
        h = float(cfg.denoise_params.get("h", 0.8 * sigma_est))
        ps = int(cfg.denoise_params.get("patch_size", 3))
        pd = int(cfg.denoise_params.get("patch_distance", 5))
        return denoise_nl_means(
            vol, h=h, patch_size=ps, patch_distance=pd, fast_mode=True, channel_axis=None, preserve_range=True
        ).astype(vol.dtype, copy=False)
    if m == "bilateral":
        sc = float(cfg.denoise_params.get("sigma_color", 0.1))
        ss = float(cfg.denoise_params.get("sigma_spatial", 1.5))
        return denoise_bilateral(vol, sigma_color=sc, sigma_spatial=ss, channel_axis=None).astype(vol.dtype, copy=False)
    return vol


def _threshold_scalar(vol: np.ndarray, method: str, percentile: float) -> float:
    m = method.lower()
    if m == "yen":
        return float(threshold_yen(vol))
    if m == "li":
        return float(threshold_li(vol))
    if m == "triangle":
        return float(threshold_triangle(vol))
    if m == "percentile":
        return float(np.percentile(vol, float(np.clip(percentile, 0, 100))))
    return float(threshold_otsu(vol))


def _density_from_mask(bw: np.ndarray, spacing: Spacing, sigma_um: float, core_alpha: float) -> np.ndarray:
    dz, dy, dx = spacing
    d_in = ndi.distance_transform_edt(bw, sampling=(dz, dy, dx)).astype(np.float32)
    mass = (d_in**core_alpha).astype(np.float32) if core_alpha > 0 else bw.astype(np.float32)
    sig = (sigma_um / dz, sigma_um / dy, sigma_um / dx)
    density = ndi.gaussian_filter(mass, sigma=sig).astype(np.float32)
    norm = ndi.gaussian_filter(np.ones_like(mass, dtype=np.float32), sigma=sig)
    eps = np.finfo(np.float32).eps
    density /= norm + eps
    m = float(density.max())
    if m > 0:
        density /= m
    return density


def segment_3d(
    volume_main: npt.NDArray[np.generic],
    spacing_um: Spacing,
    cfg: SegmentationCfg,
    *,
    volume_tissue: Optional[np.ndarray],
    dbg: DebugStore,
) -> SegmentResult:
    # 1) preprocess
    vol = volume_main.astype(IMAGE_DTYPE, copy=False)
    vol = _denoise(vol, cfg, spacing_um)
    dbg.add("vol_denoised", vol)

    # 2) threshold & morphology
    th = _threshold_scalar(vol, cfg.threshold.method, cfg.threshold.percentile)
    dbg.add("th_value", float(th))
    bw = vol > th
    dbg.add("bw_raw", bw.astype(np.uint8, copy=False))

    if cfg.morphology.open_radius:
        bw = binary_opening(bw, ball(int(cfg.morphology.open_radius)))
    if cfg.morphology.close_radius:
        bw = binary_closing(bw, ball(int(cfg.morphology.close_radius)))
    dbg.add("bw_morph", bw.astype(np.uint8, copy=False))

    if cfg.min_voxels:
        bw = remove_small_objects(bw, min_size=int(cfg.min_voxels))
    if cfg.post_open_radius:
        bw = binary_opening(bw, ball(int(cfg.post_open_radius)))
    if cfg.post_close_radius:
        bw = binary_closing(bw, ball(int(cfg.post_close_radius)))
    dbg.add("bw_post", bw.astype(np.uint8, copy=False))

    if cfg.compute_density:
        density = _density_from_mask(bw, spacing_um, cfg.density_sigma_um, cfg.density_core_alpha)
        dbg.add("density_prob", density)

    if (not cfg.watershed.enabled) or (not np.any(bw)):
        labels = (bw > 0).astype(np.int32)
        return SegmentResult(bw=bw, labels=labels, artifacts={"th": float(th)}, debug=dbg)

    # 3) watershed
    distance = ndi.distance_transform_edt(bw, sampling=spacing_um).astype(DIST_DTYPE, copy=False)
    elevation = -distance
    dbg.add("distance", distance)
    dbg.add("elevation_pre_prior", elevation)

    if cfg.tissue_prior and cfg.tissue_prior.enabled and (volume_tissue is not None):
        t = volume_tissue.astype(IMAGE_DTYPE, copy=False)
        t_norm = _robust_normalize(t, *cfg.tissue_prior.norm_clip)
        elevation = (elevation + cfg.tissue_prior.weight * t_norm.astype(DIST_DTYPE)).astype(DIST_DTYPE)
    dbg.add("elevation", elevation)

    coords = peak_local_max(distance, labels=bw, footprint=np.ones((3, 3, 3), bool), exclude_border=False)
    seeds = np.zeros_like(bw, dtype=bool)
    if coords.size:
        seeds[tuple(coords.T)] = True
    markers, _ = ndi.label(seeds)
    dbg.add("markers", markers.astype(np.int32, copy=False))

    labels = watershed(elevation, markers=markers, mask=bw, compactness=float(cfg.watershed.compactness))

    if cfg.min_voxels and cfg.min_voxels > 1:
        sizes = np.bincount(labels.ravel())
        small = np.where(sizes < int(cfg.min_voxels))[0]
        small = small[small != 0]
        if small.size:
            labels[np.isin(labels, small)] = 0
        labels, _, _ = relabel_sequential(labels)

    return SegmentResult(bw=bw, labels=labels, artifacts={"th": float(th)}, debug=dbg)
