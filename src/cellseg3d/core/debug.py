from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict

import numpy as np


@dataclass
class MemoryPolicy:
    persist_dir: Path
    persist_large_arrays: bool = True
    large_threshold_mb: int = 64
    use_zarr: bool = False  # placeholder for future use


class DebugStore:
    """Stores heavy arrays on disk (compressed) and keeps small scalars in memory."""

    def __init__(self, policy: MemoryPolicy):
        self._policy = policy
        self._index: Dict[str, Dict[str, Any]] = {}
        self._scalars: Dict[str, Any] = {}
        self._policy.persist_dir.mkdir(parents=True, exist_ok=True)

    def add(self, name: str, value: Any) -> None:
        if isinstance(value, np.ndarray):
            nbytes = int(value.nbytes)
            if self._policy.persist_large_arrays and (nbytes >= self._policy.large_threshold_mb * 1024**2):
                path = self._save_array(name, value)
                self._index[name] = {
                    "kind": "ndarray",
                    "path": str(path),
                    "shape": value.shape,
                    "dtype": str(value.dtype),
                }
            else:
                self._index[name] = {"kind": "ndarray_inline", "array": value}
        else:
            self._scalars[name] = value

    def get(self, name: str):
        meta = self._index.get(name)
        if meta is None:
            return self._scalars.get(name)
        kind = meta["kind"]
        if kind == "ndarray_inline":
            return meta["array"]
        if kind == "ndarray":
            path = Path(meta["path"])
            if path.suffix == ".npz":
                with np.load(path) as f:
                    return f["arr_0"]
            elif path.suffix == ".npy":
                return np.load(path, mmap_mode="r")
            else:
                raise ValueError(f"Unknown artifact file: {path}")
        raise KeyError(name)

    def meta(self) -> Dict[str, Any]:
        return {**self._scalars, **self._index}

    def _save_array(self, name: str, arr: np.ndarray) -> Path:
        safe = "".join(ch if ch.isalnum() or ch in "-_." else "_" for ch in name)
        path = self._policy.persist_dir / f"{safe}.npz"
        np.savez_compressed(path, arr)
        return path
