from __future__ import annotations

from enum import Enum
from pathlib import Path
from typing import List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field, PositiveInt, field_validator


class VolumeMode(str, Enum):
    translucent = "translucent"
    mip = "mip"
    attenuated_mip = "attenuated_mip"
    additive = "additive"


class WatershedMethod(str, Enum):
    hmax = "hmax"
    peaks = "peaks"


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
        # Soft-validate: allow missing during packaging; warn later in runtime
        return v


class AcquisitionCfg(BaseModel):
    default_spacing_um: Spacing
    preferred_timepoint: int = 0
    preferred_channel: int = 0


class ThresholdCfg(BaseModel):
    method: str = "otsu"
    percentile: float = 99.0


class TissuePriorCfg(BaseModel):
    enabled: bool = False
    mode: str = "normalize"
    weight: float = 0.5
    norm_clip: Tuple[float, float] = (1.0, 99.0)
    threshold_method: str = "yen"
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
    denoise_method: str = "none"
    denoise_params: dict = {}
    threshold: ThresholdCfg = ThresholdCfg()

    min_voxels: PositiveInt = 50
    morphology: MorphologyCfg = MorphologyCfg()
    watershed: WatershedCfg = WatershedCfg()
    tissue_prior: Optional[TissuePriorCfg] = TissuePriorCfg()
    post_open_radius: int = 0
    post_close_radius: int = 0

    # extras
    compute_density: bool = False
    density_sigma_um: float = 2.5
    density_core_alpha: float = 1.5


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

    tissue_opacity: float = 0.6
    tissue_colormap: str = "magenta"

    layers_to_show: List[str] = ["ch0", "labels", "centroids"]
    preview_downscale: int = 1


class MemoryCfg(BaseModel):
    image_dtype: str = "float16"
    dist_dtype: str = "float32"
    persist_dir: Path = Path("data/out/_cache")
    persist_large_arrays: bool = True
    large_threshold_mb: int = 64
    use_zarr: bool = False


class RuntimeCfg(BaseModel):
    log_level: str = "INFO"


class Settings(BaseModel):
    data: DataCfg
    acquisition: AcquisitionCfg
    segmentation: SegmentationCfg
    visualization: VisualizationCfg
    memory: MemoryCfg
    runtime: RuntimeCfg = RuntimeCfg()

    @classmethod
    def from_yaml(cls, path: Path) -> "Settings":
        with open(path, "r") as f:
            data = yaml.safe_load(f)
        return cls(**data)
