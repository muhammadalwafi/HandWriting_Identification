#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import random
from pathlib import Path
from collections import Counter
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
import json

# ================================
# Configuration
# ================================
class Config:
    TRAIN_DIR = Path("train")
    MODEL_OUT = "model.keras"
    LABELS_OUT = "labels.json"
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    BATCH_SIZE = 32
    EPOCHS = 50
    VAL_SPLIT = 0.2
    SEED = 42
    ROTATION = 0.1
    TRANSLATION = 0.05
    ZOOM = 0.1
    NOISE = 0.01

# Set seeds
random.seed(Config.SEED)
np.random.seed(Config.SEED)
tf.random.set_seed(Config.SEED)

# ================================
# Utilities
# ================================
def list_images(folder):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    return sorted([str(p) for p in Path(folder).rglob("*") if p.suffix.lower() in exts])

def label_from_filename(fp):
    return int(Path(fp).name[:2]) - 1

def load_and_preprocess_image(fp):
    img = cv2.imread(fp)
    if img is None:
        return None
    # Grayscale
    if len(img.shape) == 3:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    return img

# ================================
# Augmentation
# ================================
augmentation = tf.keras.Sequential([
    layers.RandomRotation(Config.ROTATION),
    layers.RandomTranslation(Config.TRANSLATION, Config.TRANSLATION),
    layers.RandomZoom(Config.ZOOM, Config.ZOOM),
    layers.GaussianNoise(Config.NOISE),
], name="light_augmentation")

def augment_image(img):
    img_tensor = tf.convert_to_tensor(img[None, ...], dtype=tf.float32)
    aug_img = augmentation(img_tensor)[0].numpy()
    return aug_img

# ================================
# Load Dataset
# ================================
files = list_images(Config.TRAIN_DIR)
if not files:
    raise SystemExit("No training images found.")

images, labels = [], []
for fp in files:
    img = load_and_preprocess_image(fp)
    if img is None:
        continue
    # augment each image once
    images.append(img)
    labels.append(label_from_filename(fp))
    images.append(augment_image(img))
    labels.append(label_from_filename(fp))

images = np.array(images, dtype=np.float32)
labels = np.array(labels, dtype=np.int32)
num_classes = len(np.unique(labels))
labels_cat = to_categorical(labels, num_classes=num_classes)

# Shuffle
perm = np.random.permutation(len(images))
images, labels_cat = images[perm], labels_cat[perm]

# Split train/val
val_count = int(Config.VAL_SPLIT * len(images))
X_val, y_val = images[:val_count], labels_cat[:val_count]
X_train, y_train = images[val_count:], labels_cat[val_count:]

# Class weights
counts = Counter(labels)
class_weight = {i:(len(labels)/(len(counts)*counts[i])) for i in counts}

# ================================
# Build Model
# ================================
def build_model(input_shape, num_classes):
    inp = layers.Input(shape=input_shape)
    x = layers.RandomRotation(0.05)(inp)
    x = layers.RandomTranslation(0.03,0.03)(x)
    for filters in [32,64,128,256]:
        x = layers.Conv2D(filters, 3, padding="same", activation="relu")(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPool2D()(x)
        x = layers.Dropout(0.2)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out)

model = build_model((Config.IMG_HEIGHT, Config.IMG_WIDTH,1), num_classes)
model.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# ================================
# Callbacks
# ================================
ckpt = callbacks.ModelCheckpoint(Config.MODEL_OUT, monitor="val_accuracy", save_best_only=True, verbose=1)
early = callbacks.EarlyStopping(monitor="val_accuracy", patience=10, restore_best_weights=True, verbose=1)
rlr = callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=4, verbose=1)

# ================================
# Training
# ================================
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=Config.EPOCHS,
    batch_size=Config.BATCH_SIZE,
    shuffle=True,
    class_weight=class_weight,
    callbacks=[ckpt, rlr, early]
)

# Save model & labels
model.save(Config.MODEL_OUT)
with open(Config.LABELS_OUT, "w") as f:
    json.dump([str(i) for i in range(num_classes)], f, indent=2)

print(f"Saved model to {Config.MODEL_OUT} and labels to {Config.LABELS_OUT}")
