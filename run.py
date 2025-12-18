#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path
import csv

# ================================
# Configuration
# ================================
MODEL_PATH = "model.keras"
LABELS_PATH = "labels.json"
TEST_DIR = "test"
IMG_HEIGHT = 64
IMG_WIDTH = 64
RESULT_CSV = "result.csv"

# ================================
# Load model and labels
# ================================
model = tf.keras.models.load_model(MODEL_PATH)
with open(LABELS_PATH, "r") as f:
    labels = json.load(f)

# ================================
# Utilities
# ================================
def preprocess_image(fp):
    img = cv2.imread(fp)
    if img is None:
        raise ValueError(f"Failed to read image '{fp}'")
    # Convert to grayscale if training used grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (IMG_WIDTH, IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)  # channel dim
    img = np.expand_dims(img, axis=0)   # batch dim
    return img

def predict_image(fp):
    img = preprocess_image(fp)
    preds = model.predict(img, verbose=0)[0]
    idx = np.argmax(preds)
    return labels[idx], preds[idx]

# ================================
# Evaluate on test folder
# ================================
test_files = sorted([str(p) for p in Path(TEST_DIR).glob("*.png")])
if not test_files:
    raise SystemExit("No test images found!")

results = []
correct = 0

for fp in test_files:
    # Adjust actual label to match training labels (0-based)
    actual_label = int(Path(fp).name[:2]) - 1
    predicted_label, confidence = predict_image(fp)
    results.append({
        "filename": Path(fp).name,
        "actual": actual_label,
        "predicted": int(predicted_label)
    })
    if actual_label == int(predicted_label):
        correct += 1

average_accuracy = correct / len(test_files)
print(f"Average accuracy on test dataset: {average_accuracy*100:.2f}%")

# ================================
# Save results to CSV
# ================================
with open(RESULT_CSV, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["filename", "actual", "predicted"])
    writer.writeheader()
    for row in results:
        writer.writerow(row)

print(f"Saved results to {RESULT_CSV}")
