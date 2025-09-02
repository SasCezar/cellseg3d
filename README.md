# cellseg3d

![Linux](https://img.shields.io/badge/os-linux-green)
![macOS](https://img.shields.io/badge/os-macOS-lightgrey)
![Windows](https://img.shields.io/badge/os-windows-blue)
![License: GPL v3](https://img.shields.io/badge/license-GPLv3-blue)

**cellseg3d** is a memory-optimized 3D segmentation pipeline for **confocal / electron microscopy** LIF datasets (Leica `.lif`), built with modern Python tools.

It provides reproducible segmentation of nuclei or cells, handles large 3D stacks efficiently, and integrates tightly with **Napari** for visualization.
The second channel can optionally serve as a **tissue/edge prior** to refine watershed splitting of touching cells.

---

## ✨ Features

-   Load `.lif` microscopy series with channel/time selection.
-   Apply **denoising** (Gaussian, TV, NLM, Bilateral).
-   Thresholding (Otsu, Yen, Li, Triangle, Percentile).
-   Morphological cleanup and size filtering.
-   Optional **tissue prior** from channel 1 (normalize/threshold to bias watershed ridges).
-   Watershed-based splitting of touching cells.
-   Export **centroids** to CSV with voxel & µm coordinates.
-   Optional **density maps** (KDE-like probability fields).
-   Memory-aware **DebugStore** (large intermediates stored as NPZ/Zarr).
-   Flexible visualization with **Napari**, controlled entirely by YAML config.

---

## 🚀 Quick start

```bash
# Install dependencies (requires Python >=3.11)
uv sync    # or: pip install -e .

# Run segmentation on the dataset specified in config.yaml
cellseg3d run -c config.yaml
```

---

## 🔬 Pipeline

1. **I/O**: Load series from Leica `.lif` file using `readlif`.
2. **Preprocessing**: Denoising (optional) + intensity normalization.
3. **Thresholding**: Adaptive (Otsu, etc.) to get binary masks.
4. **Morphology**: Open/close + small object removal.
5. **Watershed**: Distance transform → markers → watershed split.
    - _Optional tissue prior_: channel 1 enhances ridges for separating touching cells.
6. **Features**: Extract centroids, volumes, etc. (export CSV).
7. **Debug & Viz**: DebugStore stores intermediates; Napari shows user-selected layers.

---

## 🛠️ Tech stack

-   **Python** (3.13)
-   **NumPy / SciPy** (numerics, morphology, distance transforms)
-   **scikit-image** (segmentation, denoising, regionprops)
-   **Pandas** (feature tables)
-   **typer** (CLI)
-   **pydantic** (config models)
-   **napari** (interactive 3D visualization)
-   **readlif** (Leica `.lif` reader)

---

## 📂 Project structure

```
src/cellseg3d/
  core/      # types, debug store
  io/        # LIF reader
  seg/       # segmentation
  feat/      # feature extraction (centroids)
  viz/       # registry + napari viewer
  pipeline/  # orchestrator
```

---

## ⚙️ Configuration

All parameters are controlled via a single YAML file (example: `config.yaml`).

### Data

```yaml
data:
    lif_path: data/2025.07.24_right_after_seeding.lif
    valid_idx: [8] # series indices to process
    output_dir: data/out
```

### Acquisition

```yaml
acquisition:
    default_spacing_um: { dz: 1.0, dy: 0.2, dx: 0.2 }
    preferred_timepoint: 0
    preferred_channel: 0
```

### Segmentation

```yaml
segmentation:
    denoise_method: "none" # none|gaussian|tv|nlm|bilateral
    threshold: { method: "otsu" } # add percentile: 99 for percentile method.
    morphology: { open_radius: 1, close_radius: 0 }
    min_voxels: 100
    watershed: { enabled: true, method: "hmax", h: 1.0 }
    tissue_prior: { enabled: false, mode: "normalize", weight: 0.5 }
    compute_density: false
```

### Visualization

```yaml
visualization:
    enabled: true
    layers_to_show: ["ch0", "labels", "centroids"]
    render_3d: false
    points_size: 6
    tissue_opacity: 0.6
    tissue_colormap: "magenta"
```

### Memory

```yaml
memory:
    image_dtype: "float16" # working precision
    dist_dtype: "float32"
    persist_large_arrays: true # spill heavy arrays to disk
    large_threshold_mb: 64
    persist_dir: "data/out/_cache"
```

---

## 📜 License

This project is licensed under the **GNU GPL v3**.
See the [LICENSE](LICENSE) file for details.

---
