"""Siamese Network - Handwriting Writer Recognition (Simplified)"""
import os, json, random
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, Model
from pathlib import Path

# === CONFIG ===
TRAIN_DIR, MODEL_OUT, EMBEDDINGS_OUT = "train", "model.keras", "embeddings.npy"
IMG_SIZE, EPOCHS, BATCH_SIZE, LR = 64, 50, 32, 0.0005

random.seed(42); np.random.seed(42); tf.random.set_seed(42)

# === FUNCTIONS ===
def get_files(folder):
    return sorted([str(f) for f in Path(folder).glob("*") if f.suffix.lower() in (".png",".jpg",".jpeg")])

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.createCLAHE(2.0, (8,8)).apply(img)
    return (img.astype(np.float32) / 255.0)[..., np.newaxis]

def augment(img):
    h, w = img.shape[:2]
    result = img[:,:,0].copy()
    # Rotation + Scale
    ang, sc = random.uniform(-10,10), random.uniform(0.9,1.1)
    M = cv2.getRotationMatrix2D((w/2,h/2), ang, sc)
    result = cv2.warpAffine(result, M, (w,h), borderValue=1.0)
    # Translate
    dx, dy = random.randint(-3,3), random.randint(-3,3)
    result = cv2.warpAffine(result, np.float32([[1,0,dx],[0,1,dy]]), (w,h), borderValue=1.0)
    # Brightness + Noise
    result = np.clip(result * random.uniform(0.8,1.2), 0, 1)
    if random.random() > 0.5:
        result = np.clip(result + np.random.normal(0, 0.015, result.shape), 0, 1)
    return result.astype(np.float32)[..., np.newaxis]

def build_embedding_net(shape):
    inp = layers.Input(shape)
    x = inp
    for f in [32, 64, 128]:
        x = layers.Conv2D(f, 3, padding='same', activation='relu')(x)
        x = layers.BatchNormalization()(x)
        x = layers.MaxPooling2D()(x)
        x = layers.Dropout(0.2 if f < 128 else 0.3)(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(128, activation='relu')(x)
    x = layers.Dropout(0.3)(x)
    out = layers.Lambda(lambda x: tf.math.l2_normalize(x, axis=1))(x)
    return Model(inp, out, name='embedding_net')

def build_siamese(shape):
    emb_net = build_embedding_net(shape)
    in_a, in_b = layers.Input(shape, name='a'), layers.Input(shape, name='b')
    dist = layers.Lambda(lambda x: tf.abs(x[0]-x[1]))([emb_net(in_a), emb_net(in_b)])
    out = layers.Dense(1, activation='sigmoid')(dist)
    return Model([in_a, in_b], out), emb_net

def create_pairs(images, labels, n_pairs=300):
    pairs, pair_labels = [], []
    label_imgs = {}
    for img, lbl in zip(images, labels):
        label_imgs.setdefault(lbl, []).append(img)
    all_lbls = list(label_imgs.keys())
    
    for lbl in all_lbls:
        imgs = label_imgs[lbl]
        for _ in range(n_pairs//2):  # Positive
            pairs.append([random.choice(imgs), augment(random.choice(imgs))])
            pair_labels.append(1)
        for _ in range(n_pairs//2):  # Negative
            other = random.choice([l for l in all_lbls if l != lbl])
            pairs.append([random.choice(imgs), random.choice(label_imgs[other])])
            pair_labels.append(0)
    return np.array(pairs), np.array(pair_labels)

# === MAIN ===
print("="*50 + "\nSIAMESE NETWORK TRAINING\n" + "="*50)

# Load training images
files = get_files(TRAIN_DIR)
images, labels = [], []
for f in files:
    img = load_img(f)
    if img is not None:
        images.append(img)
        labels.append(int(Path(f).name[:2]) - 1)

images, labels = np.array(images), np.array(labels)
print(f"Training: {len(images)} images, {len(np.unique(labels))} writers")

# Augment
print("Augmenting...")
aug_imgs = [images] + [np.array([augment(i) for i in images]) for _ in range(20)]
aug_lbls = [labels] * 21
all_imgs, all_lbls = np.concatenate(aug_imgs), np.concatenate(aug_lbls)
print(f"After augmentation: {len(all_imgs)} samples")

# Create pairs
print("Creating pairs...")
pairs, pair_lbls = create_pairs(all_imgs, all_lbls)
idx = np.random.permutation(len(pairs))
pairs, pair_lbls = pairs[idx], pair_lbls[idx]
split = int(0.9 * len(pairs))
train_p, val_p = pairs[:split], pairs[split:]
train_l, val_l = pair_lbls[:split], pair_lbls[split:]
print(f"Pairs: {len(train_p)} train, {len(val_p)} val")

# Build & Train
print("\nBuilding model...")
model, emb_net = build_siamese((IMG_SIZE, IMG_SIZE, 1))
model.compile(optimizer=tf.keras.optimizers.Adam(LR), loss='binary_crossentropy', metrics=['accuracy'])

cbs = [
    tf.keras.callbacks.EarlyStopping('val_loss', patience=10, restore_best_weights=True),
    tf.keras.callbacks.ReduceLROnPlateau('val_loss', factor=0.5, patience=5, min_lr=1e-6)
]

print("Training...")
model.fit([train_p[:,0], train_p[:,1]], train_l,
          validation_data=([val_p[:,0], val_p[:,1]], val_l),
          epochs=EPOCHS, batch_size=BATCH_SIZE, callbacks=cbs)

# Save
print("\nSaving...")
emb_net.save(MODEL_OUT)

# Reference embeddings
ref_embs = {}
for img, lbl in zip(images, labels):
    aug = np.array([img] + [augment(img) for _ in range(10)])
    emb = np.mean(emb_net.predict(aug, verbose=0), axis=0)
    ref_embs[lbl] = emb / np.linalg.norm(emb)

np.save(EMBEDDINGS_OUT, ref_embs)
with open("labels.json", "w") as f:
    json.dump({str(i): str(i+1) for i in range(len(np.unique(labels)))}, f)

print(f"\n{'='*50}\nDONE! Model: {MODEL_OUT}, Embeddings: {EMBEDDINGS_OUT}\nRun: python run.py\n{'='*50}")
