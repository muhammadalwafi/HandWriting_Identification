#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Train CNN Model for Image Classification

This script trains a Convolutional Neural Network for image classification.
Configured to run on CPU without GPU acceleration, compatible with Windows/Linux/Mac.

Author: Muhammad Al Wafi
Date: December 2025
"""

import os
import sys
import numpy as np
import cv2
import logging
from pathlib import Path

# Force TensorFlow to use CPU only
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'  # Reduce TensorFlow warnings

import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Conv2D, MaxPooling2D, Flatten, Dense, 
    Dropout, BatchNormalization, GlobalAveragePooling2D
)
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('training.log')
    ]
)
logger = logging.getLogger(__name__)

# ================================
# Configuration & Hyperparameters
# ================================

class Config:
    """Configuration class for training parameters"""
    # Paths
    TRAIN_DIR = Path("train")
    MODEL_OUT = "model.keras"
    
    # Image parameters
    IMG_HEIGHT = 64
    IMG_WIDTH = 64
    
    # Training parameters
    BATCH_SIZE = 16
    EPOCHS = 100
    VAL_SPLIT = 0.2
    LEARNING_RATE = 0.001
    
    # Reproducibility
    SEED = 42
    
    # Augmentation parameters
    ROTATION_RANGE = 15  # degrees
    
    @classmethod
    def validate(cls):
        """Validate configuration parameters"""
        if not cls.TRAIN_DIR.exists():
            raise FileNotFoundError(f"Training directory '{cls.TRAIN_DIR}' not found")
        if not any(cls.TRAIN_DIR.glob('*.png')):
            raise FileNotFoundError(f"No PNG images found in '{cls.TRAIN_DIR}'")

# Set random seeds for reproducibility
np.random.seed(Config.SEED)
tf.random.set_seed(Config.SEED)

logger.info("Configuration loaded successfully")
logger.info(f"TensorFlow version: {tf.__version__}")
logger.info(f"Running on CPU (GPU disabled)")

# ================================
# Dataset Loading & Preprocessing
# ================================

def load_images_labels(train_dir):
    """
    Load and preprocess images from training directory.
    
    Args:
        train_dir (Path): Path to training directory
        
    Returns:
        tuple: (images, labels) as numpy arrays
    """
    images, labels = [], []
    skipped = 0
    
    # Get all PNG files
    image_files = sorted(train_dir.glob('*.png'))
    logger.info(f"Found {len(image_files)} image files in {train_dir}")
    
    for img_path in image_files:
        try:
            # Extract label from filename (assuming format: "01_xxx.png")
            label = int(img_path.name[:2]) - 1  # Convert to 0-based indexing
            
            # Read image in grayscale
            img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
            if img is None:
                logger.warning(f"Failed to read image: {img_path.name}")
                skipped += 1
                continue
            
            # Resize image
            img = cv2.resize(img, (Config.IMG_WIDTH, Config.IMG_HEIGHT))
            
            # Apply adaptive thresholding for better feature extraction
            img = cv2.adaptiveThreshold(
                img, 255, 
                cv2.ADAPTIVE_THRESH_MEAN_C,
                cv2.THRESH_BINARY_INV, 
                11, 2
            )
            
            # Normalize to [0, 1]
            img = img.astype('float32') / 255.0
            
            # Add channel dimension
            img = np.expand_dims(img, axis=-1)
            
            images.append(img)
            labels.append(label)
            
        except Exception as e:
            logger.error(f"Error processing {img_path.name}: {e}")
            skipped += 1
            continue
    
    if skipped > 0:
        logger.warning(f"Skipped {skipped} corrupted/invalid images")
    
    logger.info(f"Successfully loaded {len(images)} images")
    return np.array(images), np.array(labels)

def prepare_dataset():
    """
    Load dataset and prepare train/validation splits.
    
    Returns:
        tuple: (X_train, X_val, y_train, y_val, num_classes, class_weights)
    """
    # Validate configuration
    Config.validate()
    
    # Load images and labels
    images, labels = load_images_labels(Config.TRAIN_DIR)
    
    # Get number of classes
    num_classes = len(np.unique(labels))
    logger.info(f"Number of classes: {num_classes}")
    
    # Check class distribution
    unique, counts = np.unique(labels, return_counts=True)
    logger.info("Class distribution:")
    for cls, count in zip(unique, counts):
        logger.info(f"  Class {cls + 1}: {count} samples")
    
    # Compute class weights for imbalanced datasets
    class_weights_array = compute_class_weight(
        class_weight='balanced',
        classes=np.unique(labels),
        y=labels
    )
    class_weights = dict(enumerate(class_weights_array))
    logger.info(f"Class weights computed: {class_weights}")
    
    # Convert labels to categorical
    labels_cat = to_categorical(labels, num_classes=num_classes)
    
    # Split into train and validation sets
    X_train, X_val, y_train, y_val = train_test_split(
        images, labels_cat, 
        test_size=Config.VAL_SPLIT, 
        random_state=Config.SEED,
        stratify=labels  # Maintain class distribution
    )
    
    logger.info(f"Training samples: {len(X_train)}")
    logger.info(f"Validation samples: {len(X_val)}")
    
    return X_train, X_val, y_train, y_val, num_classes, class_weights

# ================================
# Data Augmentation (Optional)
# ================================

def augment_image(img):
    """
    Apply random rotation augmentation to an image.
    
    Args:
        img (np.ndarray): Input image
        
    Returns:
        np.ndarray: Augmented image
    """
    angle = np.random.uniform(-Config.ROTATION_RANGE, Config.ROTATION_RANGE)
    center = (Config.IMG_WIDTH // 2, Config.IMG_HEIGHT // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(
        img, M, 
        (Config.IMG_WIDTH, Config.IMG_HEIGHT), 
        borderMode=cv2.BORDER_REPLICATE
    )
    return rotated

def data_generator(X, y, batch_size, augment=True):
    """
    Generate batches of augmented data.
    
    Args:
        X (np.ndarray): Input images
        y (np.ndarray): Labels
        batch_size (int): Batch size
        augment (bool): Whether to apply augmentation
        
    Yields:
        tuple: (batch_X, batch_y)
    """
    while True:
        idx = np.random.choice(len(X), batch_size, replace=False)
        batch_X, batch_y = [], []
        
        for i in idx:
            img = X[i]
            if augment:
                img = augment_image(img)
            batch_X.append(img)
            batch_y.append(y[i])
            
        yield np.array(batch_X), np.array(batch_y)

# ================================
# Model Architecture
# ================================

def build_model(input_shape, num_classes):
    """
    Build a CNN model for image classification.
    
    Architecture:
        - 3 Convolutional blocks with BatchNorm and Dropout
        - Global Average Pooling
        - Dense layer with regularization
        - Softmax output
    
    Args:
        input_shape (tuple): Shape of input images (height, width, channels)
        num_classes (int): Number of output classes
        
    Returns:
        tf.keras.Model: Compiled model
    """
    model = Sequential([
        # First convolutional block
        Conv2D(32, (3, 3), activation='relu', padding='same', 
               input_shape=input_shape, name='conv1'),
        BatchNormalization(name='bn1'),
        MaxPooling2D((2, 2), name='pool1'),
        Dropout(0.3, name='dropout1'),
        
        # Second convolutional block
        Conv2D(64, (3, 3), activation='relu', padding='same', name='conv2'),
        BatchNormalization(name='bn2'),
        MaxPooling2D((2, 2), name='pool2'),
        Dropout(0.3, name='dropout2'),
        
        # Third convolutional block
        Conv2D(128, (3, 3), activation='relu', padding='same', name='conv3'),
        BatchNormalization(name='bn3'),
        MaxPooling2D((2, 2), name='pool3'),
        Dropout(0.3, name='dropout3'),
        
        # Classification head
        GlobalAveragePooling2D(name='gap'),
        Dense(256, activation='relu', name='dense1'),
        BatchNormalization(name='bn4'),
        Dropout(0.5, name='dropout4'),
        Dense(num_classes, activation='softmax', name='output')
    ], name='CNN_Classifier')
    
    return model

# ================================
# Training Callbacks
# ================================

def create_callbacks():
    """
    Create training callbacks for model optimization.
    
    Returns:
        list: List of Keras callbacks
    """
    callbacks = [
        # Save best model
        tf.keras.callbacks.ModelCheckpoint(
            Config.MODEL_OUT,
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        
        # Early stopping to prevent overfitting
        tf.keras.callbacks.EarlyStopping(
            monitor='val_accuracy',
            patience=15,
            restore_best_weights=True,
            mode='max',
            verbose=1
        ),
        
        # Reduce learning rate on plateau
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        ),
        
        # TensorBoard logging (optional)
        tf.keras.callbacks.TensorBoard(
            log_dir='logs',
            histogram_freq=0,
            write_graph=False
        )
    ]
    
    return callbacks

# ================================
# Main Training Function
# ================================

def train_model():
    """
    Main training function that orchestrates the entire training process.
    """
    try:
        logger.info("="*60)
        logger.info("Starting Model Training")
        logger.info("="*60)
        
        # Prepare dataset
        X_train, X_val, y_train, y_val, num_classes, class_weights = prepare_dataset()
        
        # Build model
        logger.info("Building model...")
        input_shape = (Config.IMG_HEIGHT, Config.IMG_WIDTH, 1)
        model = build_model(input_shape, num_classes)
        
        # Compile model
        optimizer = tf.keras.optimizers.Adam(learning_rate=Config.LEARNING_RATE)
        model.compile(
            optimizer=optimizer,
            loss='categorical_crossentropy',
            metrics=['accuracy']
        )
        
        # Print model summary
        logger.info("Model architecture:")
        model.summary(print_fn=lambda x: logger.info(x))
        
        # Create callbacks
        callbacks = create_callbacks()
        
        # Train model
        logger.info("Starting training...")
        logger.info(f"Epochs: {Config.EPOCHS}, Batch size: {Config.BATCH_SIZE}")
        
        history = model.fit(
            X_train, y_train,
            batch_size=Config.BATCH_SIZE,
            validation_data=(X_val, y_val),
            epochs=Config.EPOCHS,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=1
        )
        
        # Training completed
        logger.info("="*60)
        logger.info("Training completed successfully!")
        logger.info(f"Best model saved to: {Config.MODEL_OUT}")
        logger.info("="*60)
        
        # Print final metrics
        final_train_acc = history.history['accuracy'][-1]
        final_val_acc = history.history['val_accuracy'][-1]
        best_val_acc = max(history.history['val_accuracy'])
        
        logger.info(f"Final training accuracy: {final_train_acc:.4f}")
        logger.info(f"Final validation accuracy: {final_val_acc:.4f}")
        logger.info(f"Best validation accuracy: {best_val_acc:.4f}")
        
        return model, history
        
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        raise

# ================================
# Script Entry Point
# ================================

if __name__ == "__main__":
    try:
        model, history = train_model()
        logger.info("Program finished successfully")
    except KeyboardInterrupt:
        logger.warning("Training interrupted by user")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Program terminated with error: {e}")
        sys.exit(1)
