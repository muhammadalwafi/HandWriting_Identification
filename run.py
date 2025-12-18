#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
run Script for CNN Handwriting Model (CPU Only)
"""

import os

# ================================
# FORCE CPU ONLY (must be before TensorFlow import)
# ================================
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# ================================
# CONFIGURATION
# ================================
MODEL_PATH = "model.keras"
TEST_DIR = Path("test")
IMG_SIZE = 64  # Must match training

# ================================
# LOAD MODEL
# ================================
print("[INFO] Running on CPU (GPU disabled)")
print(f"[INFO] TensorFlow version: {tf.__version__}")
print("[INFO] Loading trained model...")

if not Path(MODEL_PATH).exists():
    raise FileNotFoundError(f"Model file '{MODEL_PATH}' not found. Train the model first.")

model = tf.keras.models.load_model(MODEL_PATH)
num_classes = model.output_shape[-1]
print(f"[INFO] Model loaded successfully ({num_classes} classes)")

# ================================
# PREPROCESSING (MUST MATCH TRAINING EXACTLY)
# ================================
def preprocess(img_path):
    """
    Preprocess image exactly the same way as training.
    Critical: Any difference will cause poor predictions.
    """
    # Read as grayscale
    img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read image: {img_path}")

    # Resize to training size
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))

    # Apply adaptive threshold (SAME as training)
    img = cv2.adaptiveThreshold(
        img, 255,
        cv2.ADAPTIVE_THRESH_MEAN_C,
        cv2.THRESH_BINARY_INV,
        11, 2
    )

    # Normalize to [0, 1] (SAME as training)
    img = img.astype("float32") / 255.0
    
    # Add channel dimension (64, 64) -> (64, 64, 1)
    img = np.expand_dims(img, axis=-1)
    
    # Add batch dimension (64, 64, 1) -> (1, 64, 64, 1)
    img = np.expand_dims(img, axis=0)

    return img

# ================================
# RUN TEST
# ================================
if not TEST_DIR.exists():
    raise FileNotFoundError(f"Test directory '{TEST_DIR}' not found")

test_images = sorted(TEST_DIR.glob("*.png"))
if not test_images:
    raise FileNotFoundError(f"No PNG images found in '{TEST_DIR}'")

print(f"\n[INFO] Found {len(test_images)} test images")
print("[INFO] Starting predictions...\n")
print("-" * 60)

correct = 0
total = 0
results = []

for img_path in test_images:
    try:
        image = preprocess(img_path)
        
        # Get predictions
        preds = model.predict(image, verbose=0)
        pred_class = np.argmax(preds[0])  # 0-based index
        confidence = preds[0][pred_class] * 100
        
        # Get actual label from filename (if available)
        filename = img_path.name
        actual_class = None
        if filename[:2].isdigit():
            actual_class = int(filename[:2]) - 1  # Convert to 0-based
        
        # Check accuracy if we have ground truth
        is_correct = ""
        if actual_class is not None:
            total += 1
            if pred_class == actual_class:
                correct += 1
                is_correct = " ✓"
            else:
                is_correct = f" ✗ (actual: {actual_class + 1})"
        
        # Store result
        results.append({
            "file": filename,
            "predicted": pred_class + 1,  # Convert to 1-based for display
            "confidence": confidence,
            "actual": actual_class + 1 if actual_class is not None else None
        })
        
        print(f"{filename:30} -> Writer {pred_class + 1:02d}  ({confidence:6.2f}%){is_correct}")
        
    except Exception as e:
        print(f"{img_path.name:30} -> ERROR: {e}")

print("-" * 60)

# Print accuracy summary if we have ground truth labels
if total > 0:
    accuracy = (correct / total) * 100
    print(f"\n[ACCURACY] {correct}/{total} correct = {accuracy:.2f}%")

# Save results to file
output_file = "predictions.txt"
with open(output_file, "w") as f:
    f.write("=" * 60 + "\n")
    f.write("PREDICTION RESULTS\n")
    f.write("=" * 60 + "\n\n")
    
    for r in results:
        f.write(f"File: {r['file']}\n")
        f.write(f"  Predicted: Writer {r['predicted']:02d}\n")
        f.write(f"  Confidence: {r['confidence']:.2f}%\n")
        if r['actual']:
            status = "CORRECT" if r['predicted'] == r['actual'] else "WRONG"
            f.write(f"  Actual: Writer {r['actual']:02d} [{status}]\n")
        f.write("\n")
    
    if total > 0:
        f.write("=" * 60 + "\n")
        f.write(f"ACCURACY: {correct}/{total} = {accuracy:.2f}%\n")
        f.write("=" * 60 + "\n")

print(f"\n[INFO] Results saved to '{output_file}'")
print("[INFO] Testing completed")
