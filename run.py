"""Run Writer Recognition - Siamese Network (Simplified)"""
import cv2
import numpy as np
import tensorflow as tf
from pathlib import Path

# === CONFIG ===
TEST_DIR, MODEL_PATH, EMBEDDINGS_PATH, OUTPUT_CSV = "test", "model.keras", "embeddings.npy", "result.csv"
IMG_SIZE = 64

# === FUNCTIONS ===
def get_files(folder):
    return sorted([str(f) for f in Path(folder).glob("*") if f.suffix.lower() in (".png",".jpg",".jpeg")])

def load_img(path):
    img = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    if img is None: return None
    img = cv2.resize(img, (IMG_SIZE, IMG_SIZE))
    img = cv2.createCLAHE(2.0, (8,8)).apply(img)
    return (img.astype(np.float32) / 255.0)[..., np.newaxis]

def cosine_sim(a, b):
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)

# === MAIN ===
print("="*60 + "\nWRITER RECOGNITION - SIAMESE NETWORK\n" + "="*60)

# Load model & embeddings
model = tf.keras.models.load_model(MODEL_PATH)
ref_embs = np.load(EMBEDDINGS_PATH, allow_pickle=True).item()
print(f"Model: {MODEL_PATH} | Writers: {len(ref_embs)}")

# Test
files = get_files(TEST_DIR)
print(f"Test images: {len(files)}\n" + "="*60)

results, correct = [], 0
for f in files:
    fname = Path(f).name
    actual = int(fname[:2])
    
    img = load_img(f)
    if img is None: continue
    
    # Get embedding & find best match
    emb = model.predict(img[np.newaxis], verbose=0)[0]
    emb = emb / (np.linalg.norm(emb) + 1e-8)
    
    best_id = max(ref_embs.keys(), key=lambda k: cosine_sim(emb, ref_embs[k]))
    predicted = best_id + 1
    
    if predicted == actual: correct += 1
    results.append((fname, actual, predicted))
    print(f"{fname}: Actual={actual}, Pred={predicted} {'✓' if predicted==actual else ''}")

# Results
acc = (correct / len(results) * 100) if results else 0
print(f"\n{'='*60}\nRESULTS: {correct}/{len(results)} correct | Accuracy: {acc:.2f}%\n{'='*60}")

# Save CSV
with open(OUTPUT_CSV, "w") as f:
    f.write("filename,actual,predicted\n")
    for r in results:
        f.write(f"{r[0]},{r[1]},{r[2]}\n")
print(f"Saved: {OUTPUT_CSV}")
