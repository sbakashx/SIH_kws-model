"""
train_model.py

Builds and trains the DS-CNN-S architecture (from "Hello Edge: Keyword
Spotting on Microcontrollers", Zhang et al. 2017) as a binary classifier
on the extracted MFCC features.

Run from inside your activated virtual environment:
    python train_model.py
"""

import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from pathlib import Path
from sklearn.model_selection import GroupShuffleSplit

# ---------------- CONFIG ----------------

BASE_DIR = Path(r"D:\kws-hackathon")
FEATURES_DIR = BASE_DIR / "features"
MODEL_DIR = BASE_DIR / "model"

BATCH_SIZE = 16
EPOCHS = 60
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# -----------------------------------------


def build_ds_cnn_s(input_shape):
    """
    DS-CNN-S from the Hello Edge paper, adapted for binary output.
    One standard conv layer, then 4 depthwise-separable conv blocks,
    then global average pooling into a single sigmoid output.

    Uses LayerNormalization instead of BatchNormalization to prevent small-batch
    mismatch issues on microcontrollers. Includes input dropout (SpecAugment)
    to prevent memorization.
    """
    inputs = keras.Input(shape=input_shape)

    # Input dropout to prevent memorizing exact feature frames
    x = layers.Dropout(0.1)(inputs)

    # Initial standard convolution
    x = layers.Conv2D(64, kernel_size=(10, 4), strides=(2, 2), padding="same")(x)
    x = layers.LayerNormalization()(x)
    x = layers.ReLU()(x)

    # 4 depthwise-separable conv blocks
    for _ in range(4):
        x = layers.DepthwiseConv2D(kernel_size=(3, 3), padding="same")(x)
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)
        x = layers.Conv2D(64, kernel_size=(1, 1), padding="same")(x)  # pointwise
        x = layers.LayerNormalization()(x)
        x = layers.ReLU()(x)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.4)(x)
    outputs = layers.Dense(1, activation="sigmoid")(x)  # binary: rob vs non-rob

    return keras.Model(inputs, outputs, name="DS_CNN_S")


def extract_group_id(filepath: str) -> str:
    """
    Derive a 'source recording' group ID from a feature's filename to prevent
    data leakage across train and test splits.
    """
    stem = Path(filepath).stem
    parts = stem.split("_")
    for i, p in enumerate(parts):
        if p.isdigit():
            return "_".join(parts[i + 1:]) or stem
    return stem

def convert_to_h_file(tflite_model, c_dump_path):
    """Converts TFLite binary into a C byte array header file for ESP32."""
    hex_lines = [f"0x{b:02x}" for b in tflite_model]
    
    c_array = "unsigned char kws_model_tflite[] = {\n  "
    c_array += ",\n  ".join([", ".join(hex_lines[i:i + 12]) for i in range(0, len(hex_lines), 12)])
    c_array += f"\n}};\nunsigned int kws_model_tflite_len = {len(tflite_model)};\n"

    with open(c_dump_path, "w") as f:
        f.write("#ifndef KWS_MODEL_H\n#define KWS_MODEL_H\n\n")
        f.write(c_array)
        f.write("\n#endif // KWS_MODEL_H\n")


def main():
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # ---- Load data ----
    X = np.load(FEATURES_DIR / "X.npy")
    y = np.load(FEATURES_DIR / "y.npy")
    with open(FEATURES_DIR / "filenames.txt", "r", encoding="utf-8") as fh:
        filenames = [line.strip() for line in fh if line.strip()]
    print(f"Loaded X={X.shape}, y={y.shape}")

    groups = np.array([extract_group_id(f) for f in filenames])
    print(f"  {len(set(groups))} distinct source-recording groups found")

    # ---- Group-aware split ----
    gss1 = GroupShuffleSplit(n_splits=1, test_size=(VAL_SPLIT + TEST_SPLIT), random_state=RANDOM_SEED)
    train_idx, temp_idx = next(gss1.split(X, y, groups=groups))

    val_ratio = VAL_SPLIT / (VAL_SPLIT + TEST_SPLIT)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=(1 - val_ratio), random_state=RANDOM_SEED)
    val_idx, test_idx = next(gss2.split(X[temp_idx], y[temp_idx], groups=groups[temp_idx]))
    val_idx, test_idx = temp_idx[val_idx], temp_idx[test_idx]

    X_train, y_train = X[train_idx], y[train_idx]
    X_val, y_val = X[val_idx], y[val_idx]
    X_test, y_test = X[test_idx], y[test_idx]

    mean = X_train.mean()
    std = X_train.std()
    np.save(MODEL_DIR / "normalization.npy", np.array([mean, std]))

    X_train = (X_train - mean) / std
    X_val = (X_val - mean) / std
    X_test = (X_test - mean) / std

    print(f"Train: {X_train.shape[0]}  Val: {X_val.shape[0]}  Test: {X_test.shape[0]}")

    # ---- Equal Class Weights to Prevent False-Positive Bias ----
    class_weight_dict = {0: 1.0, 1: 1.0}
    print(f"Class weights: {class_weight_dict}")

    # ---- Build model ----
    model = build_ds_cnn_s(input_shape=X.shape[1:])
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-3),
        loss="binary_crossentropy",
        metrics=["accuracy", keras.metrics.Precision(name="precision"), keras.metrics.Recall(name="recall")],
    )
    model.summary()

    # ---- Callbacks ----
    callbacks = [
        keras.callbacks.EarlyStopping(monitor="val_loss", patience=10, restore_best_weights=True),
        keras.callbacks.ModelCheckpoint(
            str(MODEL_DIR / "best_model.keras"), monitor="val_loss", save_best_only=True
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=5),
    ]

    # ---- Train ----
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        class_weight=class_weight_dict,
        callbacks=callbacks,
        verbose=2,
    )

    # ---- Evaluate on held-out test set ----
    print("\n--- Test set evaluation ---")
    results = model.evaluate(X_test, y_test, verbose=0)
    for name, value in zip(model.metrics_names, results):
        print(f"  {name}: {value:.4f}")

    # Multi-threshold false positive analysis
    y_probs = model.predict(X_test, verbose=0).flatten()
    print("\n--- False Positive Analysis Across Thresholds ---")
    for threshold in [0.50, 0.70, 0.75, 0.80, 0.85]:
        y_pred = (y_probs > threshold).astype(int)
        false_positives = np.sum((y_pred == 1) & (y_test == 0))
        true_negatives = np.sum(y_test == 0)
        fpr = (false_positives / true_negatives) * 100 if true_negatives > 0 else 0
        print(f"  Threshold {threshold:.2f} -> False Positives: {false_positives}/{true_negatives} ({fpr:.1f}%)")

    # ---- Save final model ----
    model.save(MODEL_DIR / "final_model.keras")
    print(f"\nModel saved to {MODEL_DIR / 'final_model.keras'}")
    print(f"Best checkpoint saved to {MODEL_DIR / 'best_model.keras'}")
    print(f"Normalization stats saved to {MODEL_DIR / 'normalization.npy'}")

    # ---- INT8 Quantization Export for ESP32 ----
    print("\nQuantizing Model to INT8 for TFLite Micro...")
    
    def representative_dataset_gen():
        for i in range(min(100, len(X_train))):
            yield [X_train[i:i+1].astype(np.float32)]

    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_dataset_gen
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    converter.inference_input_type = tf.int8
    converter.inference_output_type = tf.int8

    tflite_quant_model = converter.convert()

    tflite_path = MODEL_DIR / "kws_model.tflite"
    with open(tflite_path, "wb") as f:
        f.write(tflite_quant_model)

    header_path = MODEL_DIR / "kws_model.h"
    convert_to_h_file(tflite_quant_model, header_path)
    
    print(f"Saved INT8 model: {tflite_path}")
    print(f"Exported ESP32 Header file: {header_path}")


if __name__ == "__main__":
    main()
