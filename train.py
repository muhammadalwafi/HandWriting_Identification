import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split

# -------------------------------
# 1. Load Dataset using cv2
# -------------------------------
train_dir = "train"
image_size = (64, 64)

images = []
labels = []

for filename in os.listdir(train_dir):
    if filename.endswith(".png"):
        # Convert 1-based labels (01-70) to 0-based (0-69)
        label = int(filename[:2]) - 1
        labels.append(label)
        
        # Read image with cv2
        img_path = os.path.join(train_dir, filename)
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)  # load as grayscale

        # Resize
        img = cv2.resize(img, image_size)

        # Apply adaptive thresholding to enhance handwriting
        img = cv2.adaptiveThreshold(img, 255, cv2.ADAPTIVE_THRESH_MEAN_C,
                                    cv2.THRESH_BINARY_INV, 11, 2)

        # Normalize to [0,1]
        img = img.astype('float32') / 255.0

        # Expand dims to (64,64,1) for CNN
        img = np.expand_dims(img, axis=-1)
        images.append(img)

images = np.array(images)
labels = np.array(labels)

num_classes = len(np.unique(labels))
labels = to_categorical(labels, num_classes=num_classes)

print("Number of samples:", len(images))
print("Number of classes:", num_classes)
print("Shape of first image:", images[0].shape)

# Split into train and validation sets
X_train, X_val, y_train, y_val = train_test_split(images, labels, test_size=0.2, random_state=42)

# -------------------------------
# 2. Data Augmentation (Rotation)
# -------------------------------
def augment_image(img):
    # Random rotation between -15 to 15 degrees
    angle = np.random.uniform(-15, 15)
    M = cv2.getRotationMatrix2D((image_size[0]//2, image_size[1]//2), angle, 1)
    rotated = cv2.warpAffine(img, M, image_size, borderMode=cv2.BORDER_REPLICATE)
    return rotated

# Apply augmentation on the fly during training
def generator(X, y, batch_size=32):
    while True:
        idx = np.random.choice(len(X), batch_size)
        batch_X = []
        batch_y = []
        for i in idx:
            img = augment_image(X[i])
            batch_X.append(img)
            batch_y.append(y[i])
        yield np.array(batch_X), np.array(batch_y)

# -------------------------------
# 3. Build CNN Model
# -------------------------------
model = Sequential([
    Conv2D(32, (3,3), activation='relu', input_shape=(64,64,1)),
    MaxPooling2D((2,2)),
    Conv2D(64, (3,3), activation='relu'),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(128, activation='relu'),
    Dropout(0.5),
    Dense(num_classes, activation='softmax')
])

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
model.summary()

# -------------------------------
# 4. Train Model with Augmentation
# -------------------------------
history = model.fit(
    generator(X_train, y_train, batch_size=32),
    steps_per_epoch=len(X_train)*5,  # effectively augment each image 5 times per epoch
    epochs=60,
    validation_data=(X_val, y_val)
)

# -------------------------------
# 5. Predict on Validation Set
# -------------------------------
val_predictions = model.predict(X_val)
val_predicted_labels = np.argmax(val_predictions, axis=1)
val_percentages = np.max(val_predictions, axis=1) * 100

for i in range(len(X_val)):
    actual_label = np.argmax(y_val[i])
    print(f"Sample {i} | Actual: {actual_label} | Predicted: {val_predicted_labels[i]} | Confidence: {val_percentages[i]:.2f}%")

# -------------------------------
# 6. Save Model
# -------------------------------
model.save("model.keras")
print("Model saved as model.keras")