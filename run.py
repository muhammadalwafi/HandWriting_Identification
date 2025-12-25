"""Run Writer Recognition - Siamese Network"""
import os
import numpy as np
import tensorflow as tf
from pathlib import Path

# === CONFIG ===
TEST_DIR = "test"
MODEL_PATH = "model.keras"
EMBEDDINGS_PATH = "embeddings.npy"
OUTPUT_CSV = "result.csv"
IMG_SIZE = 128  # Must match train.py

# === FUNCTIONS ===
def get_files(folder):
    return sorted([str(f) for f in Path(folder).glob("*") if f.suffix.lower() in (".png",".jpg",".jpeg")])

def load_image(path):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    return img

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# === MAIN ===
print("="*60 + "\nWRITER RECOGNITION - SIAMESE NETWORK\n" + "="*60)

# Load encoder model
print("Loading model...")
encoder = tf.keras.models.load_model(MODEL_PATH)
print(f"Model: {MODEL_PATH}")

# Load reference embeddings
ref_embs = np.load(EMBEDDINGS_PATH, allow_pickle=True).item()
print(f"Writers: {len(ref_embs)}")

# Test files
files = get_files(TEST_DIR)
print(f"Test images: {len(files)}\n" + "="*60)

# Predict
results, correct = [], 0
for f in files:
    fname = Path(f).name
    actual = int(fname[:2])
    
    # Load & get embedding
    img = load_image(f)
    img_batch = tf.expand_dims(img, 0)
    emb = encoder.predict(img_batch, verbose=0)[0]
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    
    # Find best match
    best_id = max(ref_embs.keys(), key=lambda k: cosine_sim(emb, ref_embs[k]))
    predicted = best_id + 1
    
    if predicted == actual: correct += 1
    results.append((fname, actual, predicted))
    print(f"{fname}: Actual={actual}, Pred={predicted} {'✓' if predicted==actual else ''}")

# Results
acc = (correct / len(results) * 100) if results else 0
print(f"\n{'='*60}\nRESULTS: {correct}/{len(results)} | Accuracy: {acc:.2f}%\n{'='*60}")

# Save CSV
with open(OUTPUT_CSV, "w") as f:
    f.write("filename,actual,predicted\n")
    for r in results:
        f.write(f"{r[0]},{r[1]},{r[2]}\n")
print(f"Saved: {OUTPUT_CSV}")
