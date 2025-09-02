import logging
from pathlib import Path

import numpy as np
import typer

from .io_lif import iter_series_images, load_stack_zyx, spacing_um_from_img
from .segmentation import segment_3d
from .features import centroids_from_labels
from .settings import Settings

app = typer.Typer(add_completion=False)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
    )


def ensure_output_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


@app.command()
def run(config: Path = typer.Option(..., "--config", "-c", help="Path to config.yaml")):
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
        c = int(np.clip(cfg.acquisition.preferred_channel, 0, max(0, c_dim - 1)))

        vol0 = load_stack_zyx(img, t=t, c=0)
        vol1 = load_stack_zyx(img, t=t, c=1)

        dz, dy, dx = spacing_um_from_img(img)
        dz = dz or cfg.acquisition.default_spacing_um.dz
        dy = dy or cfg.acquisition.default_spacing_um.dy
        dx = dx or cfg.acquisition.default_spacing_um.dx
        spacing = (dz, dy, dx)

        bw, labels = segment_3d(
            volume_main=vol0,
            spacing_um=spacing,
            cfg=cfg.segmentation,
            volume_tissue=vol1,  # <-- new
        )
        df = centroids_from_labels(labels, spacing)

        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in str(name))
        csv_path = out_dir / f"{idx:03d}_{safe}_centroids.csv"
        df.to_csv(csv_path, index=False)

        log.info("[%03d] %s: ZYX %s | found %d cells | saved -> %s", idx, name, tuple(vol0.shape), len(df), csv_path)

        if cfg.visualization.enabled:
            try:
                from .viz import visualize

                pts = df[["z_vox", "y_vox", "x_vox"]].to_numpy(dtype=np.float32) if len(df) else np.empty((0, 3))
                visualize(
                    vol0,
                    labels if cfg.visualization.show_labels else None,
                    pts,
                    f"{idx:03d} - {name}",
                    cfg.visualization,
                    volume_tissue=vol1,
                )
            except Exception as e:
                log.warning("Visualization failed: %s", e, exc_info=True)


if __name__ == "__main__":
    app()
