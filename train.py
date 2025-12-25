"""Siamese Network - Handwriting Writer Recognition (Optimized)"""
import os, json, random
import numpy as np
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks, optimizers

# === CONFIG ===  
IMG_SIZE = 128          # Balance antara detail dan speed
BATCH_SIZE = 16
EPOCHS = 30
STEPS_PER_EPOCH = 150   # Steps per epoch
TRAIN_DIR = "train"
MODEL_OUT = "model.keras"
EMBEDDINGS_OUT = "embeddings.npy"

random.seed(42); np.random.seed(42); tf.random.set_seed(42)

# === LOAD IMAGE PATHS & LABELS ===
image_paths, labels = [], []
for fname in os.listdir(TRAIN_DIR):
    if fname.lower().endswith((".png", ".jpg", ".jpeg")):
        image_paths.append(os.path.join(TRAIN_DIR, fname))
        labels.append(fname[:2])

unique_labels = sorted(set(labels))
label_to_images = {l: [] for l in unique_labels}
for path, lbl in zip(image_paths, labels):
    label_to_images[lbl].append(path)

print(f"Writers: {len(unique_labels)} | Images: {len(image_paths)}")

# === IMAGE LOADER + AUGMENTATION ===
def load_image(path, augment=False):
    img = tf.io.read_file(path)
    img = tf.image.decode_png(img, channels=1)  # Grayscale
    img = tf.image.resize(img, (IMG_SIZE, IMG_SIZE))
    img = tf.cast(img, tf.float32) / 255.0
    
    if augment:
        img = tf.image.random_brightness(img, 0.15)
        img = tf.image.random_contrast(img, 0.85, 1.15)
        # Random rotation via affine (simplified)
        if random.random() > 0.5:
            img = tf.image.flip_left_right(img)
    return img

# === PAIR GENERATOR ===
def pair_generator(batch_size=BATCH_SIZE):
    while True:
        img_a, img_b, y = [], [], []
        for _ in range(batch_size):
            if random.random() < 0.5:
                # POSITIVE PAIR (same writer)
                label = random.choice(unique_labels)
                path = random.choice(label_to_images[label])
                img1 = load_image(path, augment=True)
                img2 = load_image(path, augment=True)
                label_pair = 1
            else:
                # NEGATIVE PAIR (different writers)
                l1, l2 = random.sample(unique_labels, 2)
                img1 = load_image(random.choice(label_to_images[l1]), augment=True)
                img2 = load_image(random.choice(label_to_images[l2]), augment=True)
                label_pair = 0
            img_a.append(img1)
            img_b.append(img2)
            y.append(label_pair)
        yield (tf.stack(img_a), tf.stack(img_b)), tf.convert_to_tensor(y, dtype=tf.float32)

# === CUSTOM L1 DISTANCE LAYER (NO LAMBDA) ===
class L1Distance(layers.Layer):
    def call(self, inputs):
        x1, x2 = inputs
        return tf.abs(x1 - x2)

# === ENCODER ===
def build_encoder():
    return models.Sequential([
        layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1)),
        
        layers.Conv2D(32, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        
        layers.Conv2D(64, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.2),
        
        layers.Conv2D(128, 3, padding='same', activation='relu'),
        layers.BatchNormalization(),
        layers.MaxPooling2D(),
        layers.Dropout(0.3),
        
        layers.GlobalAveragePooling2D(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.3),
    ], name="encoder")

encoder = build_encoder()

# === SIAMESE MODEL ===
input_a = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='input_a')
input_b = layers.Input(shape=(IMG_SIZE, IMG_SIZE, 1), name='input_b')

feat_a = encoder(input_a)
feat_b = encoder(input_b)

distance = L1Distance()([feat_a, feat_b])
output = layers.Dense(1, activation='sigmoid')(distance)

model = models.Model([input_a, input_b], output)
model.compile(
    optimizer=optimizers.Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy']
)

model.summary()

# === CALLBACKS ===
cbs = [
    callbacks.EarlyStopping(monitor='loss', patience=5, restore_best_weights=True),
    callbacks.ReduceLROnPlateau(monitor='loss', factor=0.5, patience=3, min_lr=1e-6),
]

# === TRAINING ===
print("\n" + "="*50 + "\nTRAINING SIAMESE NETWORK\n" + "="*50)
model.fit(
    pair_generator(),
    steps_per_epoch=STEPS_PER_EPOCH,
    epochs=EPOCHS,
    callbacks=cbs
)

# === SAVE ENCODER & EMBEDDINGS ===
print("\nSaving...")
encoder.save(MODEL_OUT)

# Compute reference embeddings
print("Computing reference embeddings...")
ref_embs = {}
for lbl in unique_labels:
    paths = label_to_images[lbl]
    embs = []
    for path in paths:
        # Load image and create augmented versions
        for _ in range(15):  # 15 augmented versions per image
            img = load_image(path, augment=True)
            img_batch = tf.expand_dims(img, 0)
            emb = encoder.predict(img_batch, verbose=0)[0]
            embs.append(emb)
    
    # Average embedding
    avg_emb = np.mean(embs, axis=0)
    avg_emb = avg_emb / (np.linalg.norm(avg_emb) + 1e-8)
    ref_embs[int(lbl) - 1] = avg_emb  # 0-indexed

np.save(EMBEDDINGS_OUT, ref_embs)

# Save label mapping
with open("labels.json", "w") as f:
    json.dump({str(i): lbl for i, lbl in enumerate(unique_labels)}, f)

print(f"\n{'='*50}\nDONE!\nModel: {MODEL_OUT}\nEmbeddings: {EMBEDDINGS_OUT}\nRun: python run.py\n{'='*50}")
