import logging
from pathlib import Path

import numpy as np
import typer

from .io_lif import iter_series_images, load_stack_zyx, spacing_um_from_img
from .segmentation import segment_3d, SegmentResult
from .features import centroids_from_labels
from .settings import Settings
from .viz import visualize_layers, LayerSpec

app = typer.Typer(add_completion=False)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
    )


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def _save_debug(out_dir: Path, idx: int, name: str, dbg: dict):
    # Write all arrays/scalars to an .npz file per series
    safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
    npz_path = out_dir / f"{idx:03d}_{safe}_debug.npz"
    savez_dict = {}
    for k, v in dbg.items():
        if isinstance(v, np.ndarray):
            savez_dict[k] = v
        else:
            # scalars
            savez_dict[k] = np.array(v)
    np.savez_compressed(npz_path, **savez_dict)
    return npz_path


def _build_layers(
    vol0: np.ndarray,
    vol1: np.ndarray | None,
    seg: SegmentResult,
    show_all_debug: bool,
    viz_cfg,
):
    layers: list[LayerSpec] = []

    # --- base channels (with physical scale & linear interp) ---
    layers.append(
        LayerSpec(
            name="cells (ch0)",
            kind="image",
            data=vol0,
            kwargs={"interpolation2d": "linear", "colormap": "gray", "opacity": 0.7},
        )
    )
    if vol1 is not None:
        layers.append(
            LayerSpec(
                name="tissue (ch1)",
                kind="image",
                data=vol1,
                kwargs={
                    "blending": "additive",
                    "opacity": viz_cfg.tissue_opacity,
                    "colormap": viz_cfg.tissue_colormap,
                },
            )
        )

    # --- primary result ---
    layers.append(
        LayerSpec(
            name="labels",
            kind="labels",
            data=seg.labels,
            kwargs={"blending": "translucent"},
        )
    )

    # --- debug/intermediates (optional) ---
    if show_all_debug:
        # rawdenoised volumes
        if "vol_denoised" in seg.debug:
            layers.append(
                LayerSpec(
                    name="denoised (ch0)",
                    kind="image",
                    data=seg.debug["vol_denoised"],
                    kwargs={"colormap": "gray", "opacity": 0.7, "interpolation2d": "linear"},
                )
            )

        # progressive masks
        for key in ("bw_raw", "bw_morph", "bw_post"):
            if key in seg.debug:
                layers.append(
                    LayerSpec(
                        name=key,
                        kind="image",
                        data=seg.debug[key].astype(np.uint8),
                        kwargs={"colormap": "gray", "opacity": 0.6},
                    )
                )

        # distance & elevations
        if "distance" in seg.debug:
            layers.append(
                LayerSpec(
                    name="distance",
                    kind="image",
                    data=seg.debug["distance"],
                    kwargs={"colormap": "magma", "opacity": 0.7},
                )
            )
        if "elevation_pre_prior" in seg.debug:
            layers.append(
                LayerSpec(
                    name="elevation_pre_prior",
                    kind="image",
                    data=seg.debug["elevation_pre_prior"],
                    kwargs={"colormap": "viridis", "opacity": 0.6},
                )
            )
        if "elevation" in seg.debug:
            layers.append(
                LayerSpec(
                    name="elevation",
                    kind="image",
                    data=seg.debug["elevation"],
                    kwargs={"colormap": "viridis", "opacity": 0.7},
                )
            )
        if "density_prob" in seg.debug:
            layers.append(
                LayerSpec(
                    name="density_prob",
                    kind="image",
                    data=seg.debug["density_prob"],
                    kwargs={"colormap": "magma", "opacity": 0.7},
                )
            )

    return layers


@app.command()
def run(
    config: Path = typer.Option(..., "--config", "-c", help="Path to config.yaml"),
):
    cfg = Settings.from_yaml(config)
    setup_logging(cfg.runtime.log_level)
    log = logging.getLogger("cellseg3d")

    out_dir = ensure_output_dir(cfg.data.output_dir or cfg.data.lif_path.with_suffix(""))

    valid = set(cfg.data.valid_idx)
    for idx, img in enumerate(iter_series_images(cfg.data.lif_path), start=1):
        if idx not in valid:
            continue

        name = getattr(img, "name", f"series_{idx}")
        z_dim = int(img.dims.z)
        t_dim = int(img.dims.t)
        c_dim = int(img.channels)
        if z_dim < 1:
            log.warning("[%03d] %s: no Z dimension; skipping.", idx, name)
            continue

        t = int(np.clip(cfg.acquisition.preferred_timepoint, 0, max(0, t_dim - 1)))
        _ = int(np.clip(cfg.acquisition.preferred_channel, 0, max(0, c_dim - 1)))

        vol0 = load_stack_zyx(img, t=t, c=0)
        vol1 = load_stack_zyx(img, t=t, c=1) if c_dim > 1 else None

        dz, dy, dx = spacing_um_from_img(img)
        dz = dz or cfg.acquisition.default_spacing_um.dz
        dy = dy or cfg.acquisition.default_spacing_um.dy
        dx = dx or cfg.acquisition.default_spacing_um.dx
        spacing = (dz, dy, dx)

        seg = segment_3d(volume_main=vol0, spacing_um=spacing, cfg=cfg.segmentation, volume_tissue=vol1)
        df = centroids_from_labels(seg.labels, spacing)
        if len(df):
            pts = df[["z_vox", "y_vox", "x_vox"]].to_numpy(dtype=np.float32)
        else:
            pts = np.empty((0, 3), dtype=np.float32)

        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
        csv_path = out_dir / f"{idx:03d}_{safe}_centroids.csv"
        df.to_csv(csv_path, index=False)

        if cfg.visualization.save_debug_npz:
            npz = _save_debug(out_dir, idx, name, seg.debug)
            log.info("Saved debug intermediates -> %s", npz)

        log.info("[%03d] %s: ZYX %s | found %d cells | saved -> %s", idx, name, tuple(vol0.shape), len(df), csv_path)

        if cfg.visualization.enabled:
            try:
                layers = _build_layers(vol0, vol1, seg, cfg.visualization.show_debug_layers, cfg.visualization)
                layers.append(
                    LayerSpec(
                        name="centroids",
                        kind="points",
                        data=pts,
                        kwargs={"size": cfg.visualization.points_size, "face_color": "yellow"},
                    )
                )
                visualize_layers(layers, f"{idx:03d} - {name}", cfg.visualization)
            except Exception as e:
                log.warning("Visualization failed: %s", e, exc_info=True)


if __name__ == "__main__":
    app()
