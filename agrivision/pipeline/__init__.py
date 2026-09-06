"""Core OpenCV 5 crop-disease detection pipeline.

Implements the vision stages described in the AgriVision AI proposal:
  1. Video frame sampling
  2. Color-space transformation (BGR -> HSV)
  3. Morphological operations
  4. Contour detection & lesion segmentation
  5. Feature/severity scoring
"""

from __future__ import annotations

import os
import time
from typing import List, Optional, Sequence

import cv2
import numpy as np

from ..models import DetectionResult, Severity, VideoAnalysisResult, Disease


# Tunable parameters
class PipelineConfig:
    """Tunable settings for the vision pipeline."""

    def __init__(
        self,
        frame_sample_rate: int = 1,
        max_frames: int = 20,
        hsv_leaf_lower: tuple = (25, 25, 25),
        hsv_leaf_upper: tuple = (100, 255, 255),
        hsv_lesion_lower: tuple = (10, 60, 60),
        hsv_lesion_upper: tuple = (30, 255, 255),
        morph_kernel: int = 5,
        min_contour_area: int = 150,
        severity_thresholds: tuple = (0.05, 0.30, 0.60),
    ):
        self.frame_sample_rate = frame_sample_rate
        self.max_frames = max_frames
        self.hsv_leaf_lower = np.array(hsv_leaf_lower, dtype=np.uint8)
        self.hsv_leaf_upper = np.array(hsv_leaf_upper, dtype=np.uint8)
        self.hsv_lesion_lower = np.array(hsv_lesion_lower, dtype=np.uint8)
        self.hsv_lesion_upper = np.array(hsv_lesion_upper, dtype=np.uint8)
        self.morph_kernel = morph_kernel
        self.min_contour_area = min_contour_area
        self.severity_thresholds = severity_thresholds


class DiseaseClassifier:
    """Disease classification for a segmented frame.

    Backed by the CNN classifier (agrivision.classifier.CNNClassifier) when a
    trained model is available; otherwise falls back to a lesion-ratio heuristic
    so the pipeline stays runnable without weights. OpenCV 5 owns the
    segmentation/feature extraction; this decides the disease label.
    """

    def __init__(self, model_path: Optional[str] = None) -> None:
        from ..classifier import CNNClassifier

        if model_path is None:
            # Default to the trained model shipped alongside the package.
            model_path = os.path.join(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                "models", "leaf_cnn.h5",
            )
        self._cnn = CNNClassifier(model_path=model_path)

    def predict(self, image_bgr: np.ndarray, lesion_ratio: float) -> tuple:
        return self._cnn.predict(image_bgr, lesion_ratio)


class OpenCVPipeline:
    """Orchestrates frame extraction, segmentation, and severity scoring."""

    def __init__(self, config: Optional[PipelineConfig] = None,
                 classifier: Optional[DiseaseClassifier] = None) -> None:
        self.config = config or PipelineConfig()
        self.classifier = classifier or DiseaseClassifier()

    # ------------------------------------------------------------------
    # Frame sampling
    # ------------------------------------------------------------------
    def sample_frames(self, frames: Sequence[np.ndarray]) -> List[np.ndarray]:
        """Uniformly sample at most max_frames from a frame sequence."""
        if not frames:
            return []
        step = max(1, self.config.frame_sample_rate)
        sampled = list(frames[::step])[: self.config.max_frames]
        return sampled

    def extract_frames(self, video_path: str) -> List[np.ndarray]:
        """Extract sampled frames from a video file using OpenCV 5."""
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        frames: List[np.ndarray] = []
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        indices = set(range(0, total, max(1, self.config.frame_sample_rate)))
        idx = 0
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            if idx in indices and len(frames) < self.config.max_frames:
                frames.append(frame)
            idx += 1
        cap.release()
        return frames

    # ------------------------------------------------------------------
    # Leaf segmentation
    # ------------------------------------------------------------------
    def segment_leaf_mask(self, image: np.ndarray) -> np.ndarray:
        """Return a binary mask of leaf/plant regions using HSV thresholding."""
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        mask = cv2.inRange(hsv, self.config.hsv_leaf_lower,
                           self.config.hsv_leaf_upper)
        kernel = np.ones((self.config.morph_kernel, self.config.morph_kernel),
                         np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        return mask

    # ------------------------------------------------------------------
    # Lesion segmentation
    # ------------------------------------------------------------------
    def find_lesion_contours(self, lesion_mask: np.ndarray) -> List[np.ndarray]:
        """Return lesion contours above the minimum area threshold."""
        contours, _ = cv2.findContours(
            lesion_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
        )
        return [c for c in contours
                if cv2.contourArea(c) >= self.config.min_contour_area]

    def _plant_region_around_leaf(self, image: np.ndarray, core_leaf: np.ndarray) -> np.ndarray:
        """Return the dominant plant cluster as (plant_region, lesion_mask).

        The plant cluster is the union of green leaf and lesion-colored pixels,
        both of which make up the plant body. Background (neutral / low
        saturation) is excluded. Returns the filled plant region and the lesion
        mask restricted to that region.
        """
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        lesion_hue = cv2.inRange(hsv, self.config.hsv_lesion_lower,
                                 self.config.hsv_lesion_upper)
        plant = cv2.bitwise_or(core_leaf, lesion_hue)

        # Keep only the largest connected plant component (drops stray noise).
        n, labels, stats, _ = cv2.connectedComponentsWithStats(plant, 8)
        if n <= 1:
            return core_leaf, np.zeros_like(core_leaf)
        largest = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        region = np.where(labels == largest, np.uint8(255), np.uint8(0))
        region = np.uint8(region)

        lesion = cv2.bitwise_and(lesion_hue, region)
        kernel = np.ones((3, 3), np.uint8)
        lesion = cv2.morphologyEx(lesion, cv2.MORPH_CLOSE, kernel)
        return region, lesion

    def segment_lesion_mask(self, image: np.ndarray, leaf_mask: np.ndarray) -> np.ndarray:
        """Return a binary mask of disease lesions.

        Convenience wrapper used when only an explicit leaf mask is available.
        """
        region, lesion = self._plant_region_around_leaf(image, leaf_mask)
        return lesion

    # ------------------------------------------------------------------
    # Severity scoring
    # ------------------------------------------------------------------
    def score_severity(self, lesion_ratio: float) -> str:
        """Map lesion area ratio to a severity level."""
        mild, moderate, severe = self.config.severity_thresholds
        if lesion_ratio < mild:
            return Severity.HEALTHY.value
        if lesion_ratio < moderate:
            return Severity.MILD.value
        if lesion_ratio < severe:
            return Severity.MODERATE.value
        return Severity.SEVERE.value

    # ------------------------------------------------------------------
    # Single frame analysis
    # ------------------------------------------------------------------
    def analyze_frame(self, image: np.ndarray, frame_index: int = 0) -> DetectionResult:
        core_leaf = self.segment_leaf_mask(image)
        # Plant region = largest cluster of green + lesion pixels; lesions are
        # grouped with the plant body while the background is excluded.
        plant_region, lesion_mask = self._plant_region_around_leaf(image, core_leaf)
        contours = self.find_lesion_contours(lesion_mask)

        leaf_area = int(np.count_nonzero(plant_region))
        lesion_area = int(np.count_nonzero(lesion_mask))
        lesion_ratio = lesion_area / leaf_area if leaf_area > 0 else 0.0

        disease, confidence = self.classifier.predict(image, lesion_ratio)
        # OpenCV 5 geometry is the source of truth for severity: the pixel
        # lesion load drives grading, not the (learned) disease label. If the
        # classifier says "healthy" while lesions are clearly present, flag the
        # label as uncertain rather than masking real damage.
        severity = self.score_severity(lesion_ratio)
        if disease == Disease.HEALTHY.value and severity != Severity.HEALTHY.value:
            disease = Disease.UNKNOWN.value
            confidence = min(confidence, 0.5)

        return DetectionResult(
            frame_index=frame_index,
            disease=disease,
            confidence=round(float(confidence), 4),
            severity=severity,
            lesion_area_ratio=round(float(lesion_ratio), 4),
            leaf_area_ratio=round(float(leaf_area) / float(image.shape[0] * image.shape[1]), 4),
            lesions=len(contours),
            metadata={
                "leaf_mask_px": leaf_area,
                "lesion_mask_px": lesion_area,
                "image_h": int(image.shape[0]),
                "image_w": int(image.shape[1]),
            },
        )

    # ------------------------------------------------------------------
    # Video analysis
    # ------------------------------------------------------------------
    def analyze_video(self, video_path: str) -> VideoAnalysisResult:
        """Run the pipeline over sampled frames of a video."""
        start = time.perf_counter()
        frames = self.extract_frames(video_path)
        detections = [
            self.analyze_frame(frame, idx)
            for idx, frame in enumerate(self.sample_frames(frames))
        ]

        if not detections:
            return VideoAnalysisResult(
                frames_analyzed=0,
                mean_severity_score=0.0,
                processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
            )

        ratios = [d.lesion_area_ratio for d in detections]
        avg_ratio = float(np.mean(ratios))
        worst = max(detections, key=lambda d: self._severity_rank(d.severity))

        return VideoAnalysisResult(
            frames_analyzed=len(detections),
            detections=[d.to_dict() for d in detections],
            dominant_disease=detections[0].disease,
            mean_severity_score=round(avg_ratio, 4),
            worst_severity=worst.severity,
            avg_lesion_area_ratio=round(avg_ratio, 4),
            processing_time_ms=round((time.perf_counter() - start) * 1000, 2),
            metadata={"sampled_frames": len(frames)},
        )

    @staticmethod
    def _severity_rank(severity: str) -> int:
        order = {Severity.HEALTHY.value: 0, Severity.MILD.value: 1,
                 Severity.MODERATE.value: 2, Severity.SEVERE.value: 3}
        return order.get(severity, -1)
