import logging
from pathlib import Path

import typer

from .io.lif_reader import iter_series_images
from .pipeline.run import process_series
from .settings import Settings

app = typer.Typer(add_completion=False)


def setup_logging(level: str) -> None:
    logging.basicConfig(
        level=getattr(logging, level.upper(), logging.INFO),
        format="%(asctime)s | %(levelname)-8s | %(name)s | %(filename)s:%(lineno)d | %(funcName)s() | %(message)s",
    )


@app.command()
def run(config: Path = typer.Option(..., "--config", "-c", help="Path to config.yaml")):
    cfg = Settings.from_yaml(config)
    setup_logging(cfg.runtime.log_level)
    log = logging.getLogger("cellseg3d")
    out_dir = cfg.data.output_dir or cfg.data.lif_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)

    valid = set(cfg.data.valid_idx)
    for idx, img in enumerate(iter_series_images(cfg.data.lif_path), start=1):
        if idx not in valid:
            continue
        try:
            process_series(img, idx, cfg, out_dir)
        except Exception as e:
            log.exception("Processing failed for series %03d: %s", idx, e)


if __name__ == "__main__":
    app()
