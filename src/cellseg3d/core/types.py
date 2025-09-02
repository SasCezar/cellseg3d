from dataclasses import dataclass
from typing import Any, Mapping, Optional, Tuple

import numpy as np

from ..core.debug import DebugStore

Spacing = Tuple[float, float, float]  # (dz, dy, dx)


@dataclass
class SegmentResult:
    bw: np.ndarray
    labels: np.ndarray
    artifacts: Mapping[str, Any]
    debug: DebugStore


@dataclass
class LayerSpec:
    name: str
    kind: str  # 'image' | 'labels' | 'points'
    data: Any
    kwargs: Optional[dict] = None
