from pathlib import Path

import numpy as np

from ..core.debug import DebugStore, MemoryPolicy
from ..core.types import SegmentResult, Spacing
from ..feat.centroids import centroids_from_labels
from ..io.lif_reader import load_stack_zyx, spacing_um_from_img
from ..seg.segmenter import segment_3d
from ..settings import Settings
from ..viz.registry import REGISTRY
from ..viz.viewer import visualize_layers


def process_series(img, idx: int, cfg: Settings, out_dir: Path):
    name = getattr(img, "name", f"series_{idx}")
    z_dim = int(img.dims.z)
    if z_dim < 1:
        return None

    t = int(np.clip(cfg.acquisition.preferred_timepoint, 0, int(img.dims.t) - 1))
    c0 = int(np.clip(cfg.acquisition.preferred_channel, 0, int(img.channels) - 1))

    vol0 = load_stack_zyx(img, t=t, c=c0)
    vol1 = load_stack_zyx(img, t=t, c=1) if int(img.channels) > 1 else None

    dz, dy, dx = spacing_um_from_img(img)
    dz = dz or cfg.acquisition.default_spacing_um.dz
    dy = dy or cfg.acquisition.default_spacing_um.dy
    dx = dx or cfg.acquisition.default_spacing_um.dx
    spacing: Spacing = (dz, dy, dx)

    dbg = DebugStore(
        MemoryPolicy(
            persist_dir=Path(cfg.memory.persist_dir),
            persist_large_arrays=cfg.memory.persist_large_arrays,
            large_threshold_mb=cfg.memory.large_threshold_mb,
            use_zarr=cfg.memory.use_zarr,
        )
    )
    if vol1 is not None:
        dbg.add("tissue", vol1)  # available in registry

    res: SegmentResult = segment_3d(vol0, spacing, cfg.segmentation, volume_tissue=vol1, dbg=dbg)

    # features
    df = centroids_from_labels(res.labels, spacing)
    pts = df[["z_vox", "y_vox", "x_vox"]].to_numpy(dtype=np.float32) if len(df) else np.empty((0, 3), np.float32)
    # update artifacts
    res = SegmentResult(
        bw=res.bw, labels=res.labels, artifacts={**res.artifacts, "centroids_vox": pts}, debug=res.debug
    )

    # save centroids
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
    (out_dir / "metrics").mkdir(parents=True, exist_ok=True)
    df.to_csv(out_dir / "metrics" / f"{idx:03d}_{safe}_centroids.csv", index=False)

    # viz
    if cfg.visualization.enabled:
        layer_keys = cfg.visualization.layers_to_show
        layers = []
        for key in layer_keys:
            builder = REGISTRY.get(key)
            if not builder:
                continue
            spec = builder(res, spacing)
            if spec is not None:
                layers.append(spec)
        visualize_layers(
            layers,
            f"{idx:03d} - {name}",
            render_3d=cfg.visualization.render_3d,
            volume_mode=cfg.visualization.volume_mode.value
            if hasattr(cfg.visualization.volume_mode, "value")
            else cfg.visualization.volume_mode,
            points_size=cfg.visualization.points_size,
        )
    return res
