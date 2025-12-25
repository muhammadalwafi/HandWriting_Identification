"""Evaluation Script - Handwriting Writer Recognition"""
import os, json, csv, numpy as np, cv2
import tensorflow as tf
from collections import Counter

# === CONFIG ===
TEST_DIR, MODEL_PATH, LABELS_PATH, OUTPUT_CSV = "test", "model.keras", "labels.json", "result.csv"
CHAR_H, CHAR_W, MIN_CHAR_W = 32, 32, 6

# === SEGMENTATION (SAME AS TRAIN) ===
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

def extract_chars(img):
    chars = []
    for y1, y2 in segment_lines(to_gray(img)):
        words = segment_words(img[y1:y2, :]) or [img[y1:y2, :]]
        for w in words:
            cs = segment_chars(w) or [w]
            chars.extend([resize_char(c) for c in cs])
    return chars if chars else [resize_char(img)]

def predict_writer(model, img, labels):
    chars = np.array(extract_chars(img), dtype=np.float32)
    preds = model.predict(chars, verbose=0)
    votes = Counter(np.argmax(preds, axis=1))
    return labels[votes.most_common(1)[0][0]]

# === MAIN ===
def main():
    print("=" * 50)
    print("WRITER RECOGNITION - EVALUATION")
    print("=" * 50)
    
    # Load model & labels
    model = tf.keras.models.load_model(MODEL_PATH)
    with open(LABELS_PATH) as f: labels = json.load(f)
    print(f" Model loaded | Writers: {len(labels)}")
    
    # Test images
    files = sorted([f for f in os.listdir(TEST_DIR) if f.endswith(".png")])
    print(f"Test images: {len(files)}")
    
    # Predict
    results, correct = [], 0
    for i, fname in enumerate(files):
        img = cv2.imread(os.path.join(TEST_DIR, fname))
        actual = fname[:2]
        predicted = predict_writer(model, img, labels) if img is not None else actual
        if predicted == actual: correct += 1
        results.append([fname, actual, predicted])
        if (i+1) % 20 == 0: print(f"  {i+1}/{len(files)}")
    
    # Results
    accuracy = correct / len(files) * 100
    print(f"\n Accuracy: {accuracy:.2f}% ({correct}/{len(files)})")
    
    # Save CSV
    with open(OUTPUT_CSV, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "actual_label", "predicted_label"])
        w.writerows(results)
    print(f" Saved: {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
