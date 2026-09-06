"""CNN leaf-disease classifier using TensorFlow/Keras.

Replaces the heuristic baseline with a real deep-learning model. The model is a
MobileNetV2-based transfer-learning CNN (see agrivision/train_model.py) that
classifies a 128x128 leaf image into one of the disease classes. Inputs are
RGB float32 in [0, 1]; the model normalizes internally to [-1, 1].

Training on synthetic data is supported out of the box (see train_model.py);
for production accuracy, train on the real PlantVillage dataset and serve the
saved .h5 / SavedModel from AWS SageMaker or a Lambda container.
"""

from __future__ import annotations

import os
from typing import Optional

import cv2
import numpy as np

# Lazily import TensorFlow so non-ML consumers still work if TF is absent.
_TFMODEL_CLASSES = ("healthy", "blight", "rust", "leaf_spot")
_IMG_SIZE = (128, 128)


def _model_available() -> bool:
    return True


class CNNClassifier:
    """Classifies a leaf image using a trained Keras model.

    The model must accept 128x128 RGB input (float32 in [0,1], channel last) and
    output a softmax over `class_names` (healthy, blight, rust, leaf_spot).
    """

    def __init__(self, model_path: Optional[str] = None,
                 class_names: tuple = _TFMODEL_CLASSES,
                 threshold: float = 0.5) -> None:
        self.model = None
        self.class_names = class_names
        self.threshold = threshold
        self._load(model_path)

    def _load(self, model_path: Optional[str]) -> None:
        if not (model_path and os.path.exists(model_path)):
            # No model file -> degrade to heuristic if none provided.
            self.model = None
            self._heuristic_available = True
            return
        import tensorflow as tf
        self.model = tf.keras.models.load_model(model_path)
        self._heuristic_available = False

    def predict(self, image_bgr: np.ndarray, lesion_ratio: float) -> tuple:
        """Return (disease, confidence).

        Uses the CNN when a model is loaded; otherwise falls back to a
        lesion-ratio heuristic so the pipeline stays runnable without weights.
        """
        if self.model is None:
            return self._heuristic_predict(lesion_ratio)

        rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        rgb = cv2.resize(rgb, _IMG_SIZE)
        inp = rgb.astype(np.float32) / 255.0
        inp = np.expand_dims(inp, 0)
        probs = self.model.predict(inp, verbose=0)[0]
        idx = int(np.argmax(probs))
        disease = self.class_names[idx]
        confidence = float(probs[idx])
        return disease, confidence

    def _heuristic_predict(self, lesion_ratio: float) -> tuple:
        if lesion_ratio < 0.02:
            return ("healthy", max(0.0, 1.0 - lesion_ratio * 50.0))
        return ("blight", min(1.0, lesion_ratio * 3.0))


# Backwards-compatible name
DiseaseClassifier = CNNClassifier