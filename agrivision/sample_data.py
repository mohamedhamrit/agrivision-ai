"""Synthetic data generator for local development, testing & CNN training.

Creates OpenCV-compatible images/videos with controllable disease loads so the
pipeline and classifier can be exercised without the (large) PlantVillage
dataset.

Disease types (each with a distinct visual signature):
    healthy   : clean green leaf, no lesions
    blight    : brown/orange necrotic spots
    rust      : orange-red pustules with chlorotic halo
    leaf_spot : dark circular spots with yellow ring

Each diseased class also varies in severity (lesion coverage fraction of the
leaf). The generator draws lesions until a target fraction of green leaf is
covered so the detected ratio is directly controllable.
"""

from __future__ import annotations

import os
from typing import Dict, Tuple

import cv2
import numpy as np

DISEASES = ("healthy", "blight", "rust", "leaf_spot")

# BGR colors for each disease's lesions
_LESION_COLORS: Dict[str, Tuple[int, int, int]] = {
    "blight": (25, 70, 150),     # brown/orange
    "rust": (15, 80, 200),       # orange-red
    "leaf_spot": (50, 60, 80),   # dark brown/black
}


def make_leaf_scene(lesion_ratio: float = 0.0,
                    disease: str = "blight",
                    size: Tuple[int, int] = (224, 224),
                    seed: int = 0) -> np.ndarray:
    """Build a synthetic image of a green leaf with optional disease lesions.

    Args:
        lesion_ratio: target fraction (0..1) of green leaf covered by lesions.
        disease: one of DISEASES.
        size: output (width, height).
        seed: RNG seed.
    """
    if disease not in DISEASES:
        raise ValueError(f"Unknown disease: {disease}. Choose from {DISEASES}")
    rng = np.random.default_rng(seed)
    w, h = size
    img = np.zeros((h, w, 3), dtype=np.uint8)

    # Neutral dark-gray background (never mimics lesion hue).
    img[:, :] = (90, 90, 90)

    # Leaf: segmented green region (an ellipse in the middle).
    center = (w // 2, h // 2)
    axes = (int(w * 0.35), int(h * 0.30))
    leaf = np.zeros((h, w), dtype=np.uint8)
    cv2.ellipse(leaf, center, axes, 0, 0, 360, 255, -1)
    green = np.array([40, 120, 60], dtype=np.uint8)
    noise = rng.integers(-8, 8, (h, w, 3)).astype(np.int16)
    img_leaf = np.clip(green.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    img[leaf > 0] = img_leaf[leaf > 0]

    # Optionally add a mild vein pattern for realism (light ridges).
    _add_veins(img, leaf, center, axes, rng)

    if disease != "healthy" and lesion_ratio > 0:
        color = _LESION_COLORS[disease]
        _add_lesions(img, leaf, center, axes, lesion_ratio, color, rng, disease)

    return img


def _add_veins(img: np.ndarray, leaf: np.ndarray, center: Tuple[int, int],
               axes: Tuple[int, int], rng) -> None:
    """Add faint green vein ridges for realism."""
    h, w = img.shape[:2]
    for _ in range(int(0.10 * w)):
        ang = rng.uniform(-np.pi / 2, np.pi / 2)
        cx = int(center[0] + rng.uniform(-axes[0], axes[0]) * 0.7)
        cy = int(center[1] + rng.uniform(-axes[1], axes[1]) * 0.7)
        length = int(rng.uniform(10, 40))
        dx, dy = int(np.cos(ang) * length), int(np.sin(ang) * length)
        pt2 = (cx + dx, cy + dy)
        if 0 <= pt2[0] < w and 0 <= pt2[1] < h:
            cv2.line(img, (cx, cy), pt2, (35, 110, 55), 1, cv2.LINE_AA)


def _add_lesions(img: np.ndarray, leaf: np.ndarray, center: Tuple[int, int],
                 axes: Tuple[int, int], lesion_ratio: float,
                 color: Tuple[int, int, int], rng, disease: str) -> None:
    """Draw lesions until `lesion_ratio` of the leaf is covered."""
    target_px = int(cv2.countNonZero(leaf) * lesion_ratio)
    covered = 0
    guard = 0
    max_tries = 4000
    while covered < target_px and guard < max_tries:
        guard += 1
        base_r = 3 + int(lesion_ratio * 12)
        rx = rng.integers(base_r, base_r + 8)
        ry = rng.integers(base_r, base_r + 8)
        cx = rng.integers(center[0] - axes[0] + 15, center[0] + axes[0] - 15)
        cy = rng.integers(center[1] - axes[1] + 15, center[1] + axes[1] - 15)

        if disease == "rust":
            # Pustules: small, roundish, red-orange with darker rim.
            cv2.circle(img, (cx, cy), int(base_r), color, -1)
            cv2.circle(img, (cx, cy), int(base_r * 1.5), (20, 70, 150), 1)
        elif disease == "leaf_spot":
            # Dark spot with a yellow-ish ring.
            cv2.circle(img, (cx, cy), int(base_r), color, -1)
            cv2.circle(img, (cx, cy), int(base_r * 1.6), (60, 130, 160), 2)
        else:
            # Blight: irregular necrotic blotches.
            cv2.ellipse(img, (cx, cy), (rx, ry), rng.integers(0, 180),
                        0, 360, color, -1)

        covered = _lesion_count(img)


def _lesion_count(img: np.ndarray) -> int:
    """Count orange/red/dark lesion-colored pixels (a proxy metric)."""
    b, g, r = img[:, :, 0], img[:, :, 1], img[:, :, 2]
    return int((((r > 110) & (b < 80) & (g > 35)) |            # orange lesions
                ((b < 70) & (g < 90) & (r < 110) & (r > 40))).sum())


def make_dataset(out_dir: str,
                 images_per_class: int = 400,
                 size: Tuple[int, int] = (224, 224)) -> Dict[str, int]:
    """Generate a balanced training/validation image set per disease class.

    Returns a dict mapping class name -> number of generated images.
    """
    os.makedirs(out_dir, exist_ok=True)
    counts: Dict[str, int] = {}
    rng = np.random.default_rng(0)
    for disease in DISEASES:
        class_dir = os.path.join(out_dir, disease)
        os.makedirs(class_dir, exist_ok=True)
        n = 0
        seed0 = int(rng.integers(0, 100000))
        for i in range(images_per_class):
            # Severity varies; healthy stays clean.
            if disease == "healthy":
                ratio = 0.0
            else:
                ratio = float(rng.uniform(0.05, 0.75))
            img = make_leaf_scene(ratio, disease, size, seed=seed0 + i)
            cv2.imwrite(os.path.join(class_dir, f"{i:04d}.png"), img)
            n += 1
        counts[disease] = n
    return counts


def make_video(lesion_ratio: float,
               path: str,
               disease: str = "blight",
               frames: int = 30,
               fps: int = 10,
               seed: int = 0) -> str:
    """Write a synthetic clip of a leaf scene with the given disease/load."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(path, fourcc, fps, (224, 224))
    for i in range(frames):
        out.write(make_leaf_scene(lesion_ratio, disease, (224, 224), seed=seed + i))
    out.release()
    return path


def get_default_videos_dir() -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "sample_data", "videos")


def get_default_dataset_dir() -> str:
    base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(base, "sample_data", "dataset")