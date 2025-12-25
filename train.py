"""Character Segmentation - Handwriting Writer Recognition"""
import os, json, random, numpy as np, cv2
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
from tensorflow.keras.utils import to_categorical
from collections import Counter

# === CONFIG ===
TRAIN_DIR, MODEL_PATH, LABELS_PATH = "train", "model.keras", "labels.json"
CHAR_H, CHAR_W, MIN_CHAR_W = 32, 32, 6
EPOCHS, BATCH_SIZE = 30, 64

random.seed(42); np.random.seed(42); tf.random.set_seed(42)

# === SEGMENTATION FUNCTIONS ===
def to_gray(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) if len(img.shape) == 3 else img

def segment_lines(gray):
    h = gray.shape[0]
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    proj = np.sum(th, axis=1)
    if proj.max() == 0: return [(0, h)]
    thresh = max(1, int(0.03 * proj.max()))
    lines, in_line, start = [], False, 0
    for y, v in enumerate(proj):
        if v > thresh and not in_line: in_line, start = True, y
        elif v <= thresh and in_line:
            in_line = False
            if y - start >= 6: lines.append((max(0, start-2), min(h, y+2)))
    if in_line: lines.append((start, h))
    return lines if lines else [(0, h)]

def segment_words(line_img):
    gray = to_gray(line_img)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    dilated = cv2.dilate(th, cv2.getStructuringElement(cv2.MORPH_RECT, (15, 3)), iterations=1)
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    bboxes = sorted([cv2.boundingRect(c) for c in contours if cv2.boundingRect(c)[2] >= 8], key=lambda b: b[0])
    return [line_img[y:y+h, x:x+w] for x, y, w, h in bboxes]

def segment_chars(word_img):
    gray = to_gray(word_img)
    _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
    cols = np.sum(th, axis=0)
    thresh = max(1, int(0.05 * cols.max()))
    chars, in_char, start = [], False, 0
    for i, sep in enumerate(cols <= thresh):
        if not sep and not in_char: in_char, start = True, i
        elif sep and in_char:
            in_char = False
            if i - start >= MIN_CHAR_W: chars.append(word_img[:, start:i])
    if in_char and len(cols) - start >= MIN_CHAR_W: chars.append(word_img[:, start:])
    return chars

def resize_char(ch_img):
    ch = cv2.cvtColor(ch_img, cv2.COLOR_GRAY2RGB) if len(ch_img.shape) == 2 else ch_img.copy()
    h, w = ch.shape[:2]
    scale = min(CHAR_W / w, CHAR_H / h)
    nw, nh = max(1, int(w * scale)), max(1, int(h * scale))
    resized = cv2.resize(ch, (nw, nh), interpolation=cv2.INTER_AREA)
    padded = 255 * np.ones((CHAR_H, CHAR_W, 3), dtype=np.uint8)
    padded[(CHAR_H-nh)//2:(CHAR_H-nh)//2+nh, (CHAR_W-nw)//2:(CHAR_W-nw)//2+nw] = resized
    return padded.astype(np.float32) / 255.0

def augment_img(img):
    """Simple augmentation"""
    h, w = img.shape[:2]
    augs = [img]
    # Rotate slightly
    for angle in [-5, 5]:
        M = cv2.getRotationMatrix2D((w//2, h//2), angle, 1.0)
        augs.append(cv2.warpAffine(img, M, (w, h), borderValue=(255,255,255)))
    # Brightness
    bright = cv2.convertScaleAbs(img, alpha=1.2, beta=10)
    dark = cv2.convertScaleAbs(img, alpha=0.8, beta=-10)
    augs.extend([bright, dark])
    return augs

def extract_chars(img, label_idx):
    X, y = [], []
    W = img.shape[1]
    patches = [img[:, :W//3], img[:, W//3:2*W//3], img[:, 2*W//3:]]
    # Add augmented patches
    all_patches = []
    for p in patches:
        all_patches.extend(augment_img(p))
    for patch in all_patches:
        for y1, y2 in segment_lines(to_gray(patch)):
            line_img = patch[y1:y2, :]
            words = segment_words(line_img) or [line_img]
            for w in words:
                chars = segment_chars(w) or [w]
                for c in chars:
                    X.append(resize_char(c))
                    y.append(label_idx)
    return X, y

# === BUILD MODEL ===
def build_model(num_classes):
    inp = layers.Input(shape=(CHAR_H, CHAR_W, 3))
    x = layers.Conv2D(32, 3, padding="same", activation="relu")(inp)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(64, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(128, 3, padding="same", activation="relu")(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPool2D()(x)
    x = layers.Dropout(0.2)(x)
    x = layers.Conv2D(256, 3, padding="same", activation="relu")(x)
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dense(512, activation="relu")(x)
    x = layers.Dropout(0.5)(x)
    out = layers.Dense(num_classes, activation="softmax")(x)
    return models.Model(inp, out)

# === MAIN ===
def main():
    print("=" * 50)
    print("CHARACTER SEGMENTATION - WRITER RECOGNITION")
    print("=" * 50)
    
    # Load images
    files = sorted([os.path.join(TRAIN_DIR, f) for f in os.listdir(TRAIN_DIR) if f.endswith(".png")])
    labels = sorted({os.path.basename(f)[:2] for f in files})
    label_to_idx = {l: i for i, l in enumerate(labels)}
    print(f"Writers: {len(labels)} | Images: {len(files)}")
    
    # Extract characters
    print("\nExtracting characters...")
    X_all, y_all = [], []
    for i, fp in enumerate(files):
        img = cv2.imread(fp)
        if img is None: continue
        X, y = extract_chars(img, label_to_idx[os.path.basename(fp)[:2]])
        X_all.extend(X); y_all.extend(y)
        if (i+1) % 20 == 0: print(f"  {i+1}/{len(files)} images, {len(X_all)} chars")
    
    print(f"\nTotal characters: {len(X_all)}")
    
    # Prepare data
    X = np.array(X_all, dtype=np.float32)
    y = np.array(y_all, dtype=np.int32)
    perm = np.random.permutation(len(X))
    X, y = X[perm], y[perm]
    y_cat = to_categorical(y, num_classes=len(labels))
    
    # Split
    val_n = max(1, int(0.1 * len(X)))
    X_val, y_val = X[:val_n], y_cat[:val_n]
    X_train, y_train = X[val_n:], y_cat[val_n:]
    print(f"Train: {len(X_train)} | Val: {len(X_val)}")
    
    # Class weights
    counts = Counter(y.tolist())
    class_weight = {i: len(y) / (len(counts) * counts[i]) for i in counts}
    
    # Build & train
    model = build_model(len(labels))
    model.compile(optimizer="adam", loss="categorical_crossentropy", metrics=["accuracy"])
    model.summary()
    
    cb = [
        callbacks.ModelCheckpoint(MODEL_PATH, monitor="val_accuracy", save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor="val_accuracy", patience=5, restore_best_weights=True)
    ]
    
    model.fit(X_train, y_train, validation_data=(X_val, y_val), epochs=EPOCHS, 
              batch_size=BATCH_SIZE, class_weight=class_weight, callbacks=cb)
    
    # Save
    model.save(MODEL_PATH)
    with open(LABELS_PATH, "w") as f: json.dump(labels, f)
    print(f"\n Model saved: {MODEL_PATH}")
    print(f" Labels saved: {LABELS_PATH}")

if __name__ == "__main__":
    main()
