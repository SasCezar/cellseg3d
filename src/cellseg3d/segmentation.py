from typing import Tuple

import numpy as np
import numpy.typing as npt
from scipy import ndimage as ndi
from skimage.feature import peak_local_max
from skimage.filters import threshold_otsu
from skimage.measure import label as cc_label
from skimage.morphology import ball, closing, h_maxima, opening, remove_small_objects
from skimage.segmentation import watershed
from skimage.segmentation import relabel_sequential

from settings import SegmentationCfg, WatershedMethod

Spacing = Tuple[float, float, float]  # (dz, dy, dx)


def _seed_markers(
    bw: npt.NDArray[np.bool_],
    distance: npt.NDArray[np.float32],
    cfg: SegmentationCfg,
    spacing_um: Spacing,
) -> np.ndarray:
    if cfg.watershed.method is WatershedMethod.hmax:
        dz, dy, dx = spacing_um
        mu_per_vox = float(dz + dy + dx) / 3.0
        seeds = h_maxima(distance, h=float(cfg.watershed.h) * mu_per_vox)
        markers, _ = ndi.label(seeds)
        return markers

    # peaks
    fp = int(max(1, cfg.watershed.footprint))
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


def segment_3d(
    volume: npt.NDArray[np.generic],
    spacing_um: Spacing,
    cfg: SegmentationCfg,
) -> tuple[np.ndarray, np.ndarray]:
    vol = volume.astype(np.float32)
    th = threshold_otsu(vol)
    bw = vol > th

    # Morphology
    if cfg.morphology.open_radius:
        bw = opening(bw, ball(cfg.morphology.open_radius))
    if cfg.morphology.close_radius:
        bw = closing(bw, ball(cfg.morphology.close_radius))

    # Remove small objects before watershed
    bw = remove_small_objects(bw, cfg.min_voxels)
    labels = cc_label(bw, connectivity=3)

    if (not cfg.watershed.enabled) or (not np.any(bw)):
        return bw, labels

    # Anisotropic distance (µm) so splitting respects voxel spacing
    dz, dy, dx = spacing_um
    distance = ndi.distance_transform_edt(bw, sampling=(dz, dy, dx)).astype(np.float32)

    markers = _seed_markers(bw, distance, cfg, spacing_um)

    labels = watershed(
        -distance,
        markers=markers,
        mask=bw,
        compactness=float(cfg.watershed.compactness),
    )

    # Clean small basins and relabel
    if cfg.min_voxels > 1:
        sizes = np.bincount(labels.ravel())
        small = np.where(sizes < cfg.min_voxels)[0]
        small = small[small != 0]
        if small.size:
            labels[np.isin(labels, small)] = 0
        labels, _, _ = relabel_sequential(labels)

    return bw, labels
