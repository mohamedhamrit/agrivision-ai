"""Train the leaf-disease CNN classifier used by the pipeline.

A compact transfer-learning model is built on a frozen ImageNet backbone
(MobileNetV2) because the generated dataset is small and training a deep CNN
from scratch on it collapses (val accuracy ~random while train overfits to
100% — see history notes). The frozen backbone gives ~99%+ held-out accuracy.

Generates a balanced synthetic dataset, trains the classification head with
early stopping on a held-out split, and saves the best model weights to
`agrivision/models/leaf_cnn.h5`.

Usage:
    python -m agrivision.train_model --epochs 12 --images-per-class 1200

Set images-per-class low (e.g. 200) for a quick smoke test. Requires network
access the first time (to fetch ImageNet weights); afterwards they are cached.
"""

from __future__ import annotations

import argparse
import os

import numpy as np

from .sample_data import DISEASES, make_dataset, get_default_dataset_dir

_IMG_SIZE = (128, 128)


def build_model(input_shape=(*_IMG_SIZE, 3), num_classes: int = len(DISEASES)):
    """Build the transfer-learning classifier (frozen MobileNetV2 + head)."""
    from tensorflow.keras import applications, layers, models, regularizers

    base = applications.MobileNetV2(
        input_shape=input_shape, include_top=False, weights="imagenet"
    )
    base.trainable = False

    inp = layers.Input(shape=input_shape)
    # Inputs arrive in [0, 1]; MobileNetV2 expects [-1, 1]. Rescaling is a
    # standard keras layer so it round-trips through saved-model loading.
    x = layers.Rescaling(scale=2.0, offset=-1.0)(inp)
    x = base(x, training=False)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Dense(num_classes, activation="softmax",
                       kernel_regularizer=regularizers.l2(1e-4))(x)
    model = models.Model(inp, out)
    model.compile(
        optimizer="adam",
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model


def load_data(dataset_dir: str, split: float = 0.85, size=_IMG_SIZE):
    """Load images+labels as float32 arrays, split into train/val."""
    import cv2

    X, y = [], []
    class_to_idx = {c: i for i, c in enumerate(DISEASES)}
    for disease in DISEASES:
        d = os.path.join(dataset_dir, disease)
        for f in sorted(os.listdir(d)):
            if not f.lower().endswith((".png", ".jpg", ".jpeg")):
                continue
            p = os.path.join(d, f)
            img = cv2.imread(p)
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, size).astype(np.float32) / 255.0
            X.append(img)
            y.append(class_to_idx[disease])

    idx = np.arange(len(X))
    np.random.seed(42)
    np.random.shuffle(idx)
    X = np.asarray(X)
    y = np.asarray(y)
    cut = int(len(X) * split)
    return X[idx[:cut]], X[idx[cut:]], y[idx[:cut]], y[idx[cut:]]


def main() -> None:
    import tensorflow as tf

    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--images-per-class", type=int, default=1200)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--dataset-dir", default=get_default_dataset_dir())
    ap.add_argument("--out", default=os.path.join(
        os.path.dirname(os.path.abspath(__file__)), "models", "leaf_cnn.h5"))
    args = ap.parse_args()

    tf.config.set_visible_devices([], "GPU")

    print("Generating dataset...")
    counts = make_dataset(args.dataset_dir, args.images_per_class, _IMG_SIZE)
    print(f"  classes: {counts}")

    print("Loading data...")
    Xtr, Xval, ytr, yval = load_data(args.dataset_dir)
    print(f"  train={Xtr.shape[0]} val={Xval.shape[0]}")

    print("Building CNN (transfer learning)...")
    model = build_model()
    model.summary()

    from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    callbacks = [
        EarlyStopping(monitor="val_loss", patience=4,
                      restore_best_weights=True, verbose=1),
        ModelCheckpoint(args.out, monitor="val_loss", save_best_only=True,
                        verbose=1),
    ]

    print("Training...")
    history = model.fit(
        Xtr, ytr,
        validation_data=(Xval, yval),
        batch_size=args.batch_size,
        epochs=args.epochs,
        verbose=1,
        callbacks=callbacks,
    )
    if not os.path.exists(args.out):
        model.save(args.out)
    print(f"Saved model -> {args.out}")
    best_val_acc = float(max(history.history.get("val_accuracy", [0.0])))
    best_val_loss = float(min(history.history.get("val_loss", [1e9])))
    print(f"Best validation accuracy: {best_val_acc:.3f} "
          f"(val_loss {best_val_loss:.3f})")


if __name__ == "__main__":
    main()