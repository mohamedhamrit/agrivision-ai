"""Offline evaluation harness producing metrics for the final report.

Computes three groups of numbers:

  1. CNN disease metrics on the generated dataset (pooled across the training
     split used by train_model.py): per-class precision / recall / F1, overall
     accuracy, and the fraction of confident (>0.9) predictions.
  2. Vision-pipeline severity agreement vs. known lesion loads, plus lesion
     ratio regression error (MAE / RMSE) on freshly rendered scenes.
  3. End-to-end latency (ms/frame) over the built-in demo clips.

Usage:
    python -m agrivision.evaluate [--disease-sample 200] [--severity-samples 30]

Writes sample_data/evaluation_report.json and prints a condensed report.
"""

from __future__ import annotations

import argparse
import json
import os
import time
from collections import defaultdict
from typing import Dict, List

import cv2
import numpy as np

from .pipeline import DiseaseClassifier, OpenCVPipeline
from .sample_data import get_default_dataset_dir, get_default_videos_dir, make_leaf_scene

_CLASSES = ("healthy", "blight", "rust", "leaf_spot")
_BUCKETS = (("healthy", 0.0), ("mild", 0.07), ("moderate", 0.40), ("severe", 0.70))


def _confusion_report(y_true: List[str], y_pred: List[str]) -> Dict:
    labels = sorted(set(y_true) | set(y_pred))
    cm = defaultdict(lambda: defaultdict(int))
    for t, p in zip(y_true, y_pred):
        cm[t][p] += 1

    per_class = {}
    for c in labels:
        tp = cm[c][c]
        fp = sum(cm[t][c] for t in labels if t != c)
        fn = sum(cm[c][p] for p in labels if p != c)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        per_class[c] = {
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "support": tp + fn,
        }
    n = len(y_true)
    macro_f1 = float(np.mean([v["f1"] for v in per_class.values()])) if per_class else 0.0
    accuracy = sum(cm[c][c] for c in labels) / n if n else 0.0
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "per_class": per_class,
        "confusion": {t: dict(cm[t]) for t in labels},
        "n": n,
    }


def disease_metrics(sample: int = 200) -> Dict:
    dataset_dir = get_default_dataset_dir()
    classifier = DiseaseClassifier()  # resolves the shipped leaf_cnn.h5
    y_true, y_pred, confidences = [], [], []
    for disease in _CLASSES:
        class_dir = os.path.join(dataset_dir, disease)
        files = sorted(os.listdir(class_dir))[:sample]
        for name in files:
            img = cv2.imread(os.path.join(class_dir, name))
            if img is None:
                continue
            pred, conf = classifier.predict(img, 0.0)
            y_true.append(disease)
            y_pred.append(pred)
            confidences.append(conf)
    report = _confusion_report(y_true, y_pred)
    report["confident_fraction"] = float(np.mean([c > 0.9 for c in confidences]))
    return report


def severity_agreement(samples: int = 30) -> Dict:
    pipeline = OpenCVPipeline()
    rows = []
    for bucket, target in _BUCKETS:
        correct = 0
        errors = []
        for k in range(samples):
            scene = make_leaf_scene(target, "blight", size=(256, 256), seed=1000 + k)
            res = pipeline.analyze_frame(scene, frame_index=k)
            measured = res.lesion_area_ratio
            errors.append(measured - target)
            if res.severity == bucket:
                correct += 1
        rows.append({
            "expected_severity": bucket,
            "target_ratio": target,
            "agreement": correct / samples,
            "mean_error": float(np.mean(errors)),
            "mae": float(np.mean(np.abs(errors))),
            "rmse": float(np.sqrt(np.mean(np.square(errors)))),
        })
    agreement = float(np.mean([r["agreement"] for r in rows]))
    return {"overall_severity_agreement": agreement, "buckets": rows}


def latency_report() -> Dict:
    pipeline = OpenCVPipeline()
    videos_dir = get_default_videos_dir()
    per_video = []
    for name in ("healthy", "mild", "moderate", "severe"):
        path = os.path.join(videos_dir, f"{name}.mp4")
        cap = cv2.VideoCapture(path)
        if not cap.isOpened():
            continue
        times = []
        frame_idx = 0
        while frame_idx < 15:
            ok, frame = cap.read()
            if not ok:
                break
            t0 = time.perf_counter()
            pipeline.analyze_frame(frame, frame_index=frame_idx)
            times.append((time.perf_counter() - t0) * 1000.0)
            frame_idx += 1
        cap.release()
        if times:
            per_video.append({
                "clip": name,
                "frames": len(times),
                "mean_ms_frame": float(np.mean(times)),
                "p95_ms_frame": float(np.percentile(times, 95)),
            })
    all_ms = [t["mean_ms_frame"] for t in per_video]
    return {
        "overall_mean_ms_frame": float(np.mean(all_ms)),
        "per_clip": per_video,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--disease-sample", type=int, default=200,
                        help="images per class for CNN metrics (default 200)")
    parser.add_argument("--severity-samples", type=int, default=30,
                        help="scenes per severity bucket (default 30)")
    parser.add_argument("--report-path", default="sample_data/evaluation_report.json")
    args = parser.parse_args()

    print("Evaluating CNN disease classification ...")
    disease = disease_metrics(args.disease_sample)

    print("Evaluating vision severity agreement ...")
    severity = severity_agreement(args.severity_samples)

    print("Measuring pipeline latency ...")
    latency = latency_report()

    report = {"disease": disease, "severity": severity, "latency": latency}
    report_path = os.path.abspath(args.report_path)
    with open(report_path, "w", encoding="utf-8") as fh:
        json.dump(report, fh, indent=2, default=float)
    print(f"Report saved to {report_path}")

    print("\n========== AgriVision AI Evaluation Report ==========")
    print(f"CNN overall accuracy : {disease['accuracy']:.4f}")
    print(f"CNN macro-F1         : {disease['macro_f1']:.4f}")
    print(f"CNN confident (p>0.9): {disease['confident_fraction']:.4f}")
    for c, m in disease["per_class"].items():
        print(f"  {c:10s} P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} (n={m['support']})")
    print(f"Severity agreement   : {severity['overall_severity_agreement']:.4f}")
    for b in severity["buckets"]:
        print(f"  {b['expected_severity']:9s} target={b['target_ratio']:.2f} "
              f"agree={b['agreement']:.3f} mae={b['mae']:.4f} rmse={b['rmse']:.4f}")
    print(f"Latency (OpenCV pipe): {latency['overall_mean_ms_frame']:.1f} ms/frame")
    for c in latency["per_clip"]:
        print(f"  {c['clip']:9s} mean={c['mean_ms_frame']:.1f}ms  p95={c['p95_ms_frame']:.1f}ms  "
              f"({c['frames']} frames)")


if __name__ == "__main__":
    main()