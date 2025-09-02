from __future__ import annotations
from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, PositiveInt, field_validator


# -----------------------------------
# Enums for clarity & typo protection
# -----------------------------------
class VolumeMode(str, Enum):
    translucent = "translucent"
    mip = "mip"
    attenuated_mip = "attenuated_mip"
    additive = "additive"


class WatershedMethod(str, Enum):
    hmax = "hmax"
    peaks = "peaks"


# ---------------------
# Acquisition settings
# ---------------------
class Spacing(BaseModel):
    dz: float = Field(..., gt=0)
    dy: float = Field(..., gt=0)
    dx: float = Field(..., gt=0)


class DataCfg(BaseModel):
    lif_path: Path
    valid_idx: List[int]
    output_dir: Optional[Path] = None

    @field_validator("lif_path")
    @classmethod
    def lif_exists(cls, v: Path) -> Path:
        if not v.exists():
            raise ValueError(f"LIF file not found: {v}")
        return v


class AcquisitionCfg(BaseModel):
    preferred_channel: int = 0
    preferred_timepoint: int = 0
    default_spacing_um: Spacing


# ---------------------
# Segmentation settings
# ---------------------
class ThresholdCfg(BaseModel):
    method: str = "otsu"  # 'otsu'|'yen'|'li'|'triangle'|'percentile'
    percentile: float = 99.0  # used if method == 'percentile'


class TissuePriorCfg(BaseModel):
    enabled: bool = False
    mode: str = "normalize"  # 'normalize' | 'threshold'
    weight: float = 0.5
    norm_clip: Tuple[float, float] = (1.0, 99.0)  # only used in 'normalize'
    threshold_method: str = "yen"  # used if mode == 'threshold'
    threshold_percentile: float = 95.0


class MorphologyCfg(BaseModel):
    open_radius: int = 1
    close_radius: int = 0


class WatershedCfg(BaseModel):
    enabled: bool = True
    method: WatershedMethod = WatershedMethod.hmax
    h: float = 1.0
    footprint: int = 3
    compactness: float = 0.0


class SegmentationCfg(BaseModel):
    # --- denoising ---
    denoise_method: str = "none"
    denoise_params: dict = {}
    # --- thresholding ---
    threshold: ThresholdCfg = ThresholdCfg()

    # --- morphology ---
    min_voxels: PositiveInt = 50
    morphology: MorphologyCfg = MorphologyCfg()

    # --- watershed ---
    watershed: WatershedCfg = WatershedCfg()

    # --- tissue prior ---
    tissue_prior: Optional[TissuePriorCfg] = TissuePriorCfg()

    # --- post cleanup ---
    post_open_radius: int = 0
    post_close_radius: int = 0


# ---------------------
# Visualization settings
# ---------------------
class SurfaceCfg(BaseModel):
    make: bool = False
    smoothing: float = 1.0
    level: float = 0.5
    downsample: int = 1


class VisualizationCfg(BaseModel):
    enabled: bool = True
    show_labels: bool = True
    points_size: int = Field(6, gt=0)
    render_3d: bool = False
    volume_mode: VolumeMode = VolumeMode.translucent
    surface: SurfaceCfg = SurfaceCfg()

    # optional: add fields for tissue overlay styling
    tissue_opacity: float = 0.6
    tissue_colormap: str = "magenta"

    show_debug_layers: bool = True  # show intermediates in napari
    save_debug_npz: bool = False  # save intermediates to disk


# ---------------------
# Runtime settings
# ---------------------
class RuntimeCfg(BaseModel):
    log_level: str = "INFO"


# ---------------------
# Root settings object
# ---------------------
class Settings(BaseModel):
    data: DataCfg
    acquisition: AcquisitionCfg
    segmentation: SegmentationCfg
    visualization: VisualizationCfg
    runtime: RuntimeCfg = RuntimeCfg()

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
