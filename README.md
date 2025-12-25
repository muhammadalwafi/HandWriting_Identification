# Character Segmentation - Handwriting Writer Recognition

![Python](https://img.shields.io/badge/python-3.8+-blue)
![TensorFlow](https://img.shields.io/badge/tensorflow-2.x-orange)
![License](https://img.shields.io/badge/license-MIT-green)

## 🎯 Objective
Recognize handwriting writers using character-level segmentation and a CNN classifier.

**Key steps:**
1. Segment lines → words → characters
2. Normalize and augment characters
3. Train CNN to classify writers
4. Save model + labels for inference

## 🛠 Requirements
1. Python 3.8+
2. TensorFlow 2.x
3. OpenCV
4. NumPy
5. tqdm

## 💻 Installation
Install dependencies:
```bash
pip install tensorflow opencv-python numpy tqdm
```

Recommended: use a virtual environment:
```bash
python -m venv venv
```

### Activate environment
**Windows:**
```bash
venv\Scripts\activate
```

**Linux / Mac:**
```bash
source venv/bin/activate
```

Install dependencies inside venv:
```bash
pip install -r requirements.txt
```

**Process:**
- Load training images (filename starts with writer ID, e.g., 01_sample.png)
- Segment characters → augment → train CNN
- Save `model.keras` + `labels.json`

## ✨ Notes
- Character-level segmentation: lines → words → characters
- Simple augmentations: rotation, brightness adjustments
- Handles class imbalance with class weights
- Suitable for CPU inference


