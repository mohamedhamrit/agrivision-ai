"""Data structures shared across AgriVision AI modules."""

from __future__ import annotations

from dataclasses import dataclass, field, asdict
from enum import Enum
from typing import Any, Dict, List, Optional


class Severity(str, Enum):
    HEALTHY = "healthy"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"


class Disease(str, Enum):
    HEALTHY = "healthy"
    BLIGHT = "blight"
    RUST = "rust"
    LEAF_SPOT = "leaf_spot"
    UNKNOWN = "unknown"
    NO_LEAF_FOUND = "no_leaf_found"


_HEALTHY_SEVERITY = Severity.HEALTHY.value


@dataclass
class DetectionResult:
    """Result of running the OpenCV 5 pipeline on a single frame or video."""

    frame_index: int = 0
    disease: str = Disease.UNKNOWN.value
    confidence: float = 0.0
    severity: str = _HEALTHY_SEVERITY
    lesion_area_ratio: float = 0.0
    leaf_area_ratio: float = 0.0
    lesions: int = 0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class VideoAnalysisResult:
    """Aggregated result over multiple sampled frames of a video."""

    frames_analyzed: int = 0
    detections: List[DetectionResult] = field(default_factory=list)
    dominant_disease: str = Disease.UNKNOWN.value
    mean_severity_score: float = 0.0
    worst_severity: str = "unknown"
    avg_lesion_area_ratio: float = 0.0
    processing_time_ms: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)
