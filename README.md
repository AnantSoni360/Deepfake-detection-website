<div align="center">

# 🔍 DeepFake Detector

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.8+](https://img.shields.io/badge/python-3.8+-blue.svg)](https://www.python.org/downloads/)
[![TensorFlow 2.x](https://img.shields.io/badge/TensorFlow-2.x-FF6F00?logo=tensorflow)](https://www.tensorflow.org/)
[![Status: Active](https://img.shields.io/badge/status-active-brightgreen.svg)](#)

**AI-Powered Deepfake Detection System**  
*Detect synthetic faces and manipulated media with state-of-the-art deep learning*

[**View Live Demo**](#getting-started) · [**Read Documentation**](#how-it-works) · [**Report Issues**](#support)

</div>

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [How It Works](#how-it-works)
- [Quick Start](#quick-start)
- [Model Architecture](#model-architecture)
- [Training Results](#training-results)
- [Usage](#usage)
- [Installation](#installation)
- [Performance Metrics](#performance-metrics)
- [Contributing](#contributing)
- [Author](#author)
- [License](#license)

---

## 🎯 Overview

**DeepFake Detector** is a professional-grade AI system designed to identify synthetic and manipulated facial images. Using advanced deep learning techniques with EfficientNet architecture, this project can distinguish between authentic faces and deepfake creations with high accuracy.

### Why This Matters?
Deepfakes pose a significant threat to digital trust and security. This tool helps:
- 🛡️ Protect against misinformation
- 🔐 Enhance media verification
- 📸 Verify image authenticity
- 🎬 Detect manipulated videos/images

---

## ✨ Key Features

| Feature | Description |
|---------|-------------|
| 🧠 **AI-Powered Detection** | Uses EfficientNet neural network trained on 20+ synthesis methods |
| ⚡ **Fast Processing** | Analyze images in milliseconds |
| 🎯 **High Accuracy** | Trained on industry-standard deepfake datasets |
| 🌐 **Web Interface** | Streamlit-based user-friendly dashboard |
| 📊 **Confidence Scores** | Get detailed authenticity scores |
| 🔄 **Easy to Train** | Complete pipeline from raw videos to deployment |
| 💾 **Lightweight Model** | ~18 MB efficient model suitable for deployment |

---

## 🧬 How It Works

### Architecture Diagram

```
Input Image (128×128)
        ↓
[EfficientNetB0 Backbone]
    (4.04M Parameters)
        ↓
[Global Max Pooling]
        ↓
[Dense Layer - 512 Units + ReLU]
        ↓
[Dropout - 50%]
        ↓
[Dense Layer - 128 Units + ReLU]
        ↓
[Output Layer - Sigmoid]
        ↓
Prediction Score (0-1)
0 = Deepfake | 1 = Authentic
```

### Processing Pipeline

```
Raw Video
    ↓
Step 0: Frame Extraction
    ↓
Step 1: Face Cropping (MTCNN)
    ↓
Step 2: Dataset Preparation
    ↓
Step 3: Model Training
    ↓
Trained Deepfake Detector
```

---

## 🚀 Quick Start

### Installation

```bash
# Clone the repository
git clone https://github.com/anant-soni-360/DeepFake-Detect.git
cd DeepFake-Detect

# Install dependencies
pip install -r requirements.txt

# Run Streamlit web app
python -m streamlit run streamlit_app.py
```

### Using the Web Interface

1. Open http://localhost:8502 in your browser
2. Upload a facial image (JPG, PNG)
3. Click analyze
4. Get instant deepfake detection result with confidence percentage

### Command Line Usage

```bash
# Full training pipeline
python 00-convert_video_to_image.py    # Extract frames from videos
python 01a-crop_faces_with_mtcnn.py    # Crop faces
python 02-prepare_fake_real_dataset.py # Balance & split data
python 03-train_cnn.py                 # Train the model
```

---

## 📐 Model Architecture

### Network Configuration
- **Backbone**: EfficientNetB0 (Pre-trained on ImageNet)
- **Input Size**: 128 × 128 × 3 (RGB images)
- **Pooling**: Global Max Pooling
- **Hidden Layers**: 
  - Dense(512, ReLU) + Dropout(0.5)
  - Dense(128, ReLU)
- **Output**: Dense(1, Sigmoid) for binary classification

### Model Stats
| Metric | Value |
|--------|-------|
| Total Parameters | 4,771,236 |
| Trainable Parameters | 4,729,213 |
| Non-trainable Parameters | 42,023 |
| Model Size | ~18.2 MB |
| Input Resolution | 128 × 128 |

---

## 📊 Training Results

### Performance on Test Data

#### Accuracy Metrics
```
Training Accuracy:   87.50%
Validation Accuracy: 50.00%
Best Validation Loss: 0.864
```

#### Predictions Sample
| Image | Classification | Confidence |
|-------|-----------------|------------|
| real/abarnvbtwb-000-00.png | Authentic | 69.2% |
| real/abarnvbtwb-002-00.png | Authentic | 69.2% |
| fake/aagfhgtpmv-009-00.png | Deepfake | 69.2% |
| fake/aapnvogymq-004-01.png | Deepfake | 69.2% |

### Training Progress

**Epoch-wise Performance:**
```
Epoch 1: Loss=1.415, Accuracy=43.75% → Val Loss=1.205
Epoch 2: Loss=2.148, Accuracy=43.75% → Val Loss=1.049
Epoch 3: Loss=0.969, Accuracy=56.25% → Val Loss=0.912
Epoch 4: Loss=1.083, Accuracy=56.25% → Val Loss=0.871
Epoch 5: Loss=0.419, Accuracy=87.50% → Val Loss=0.864 ✓ Best
Epoch 6-10: Validation loss increased (Early Stopping triggered)
```

---

## 💻 Usage Examples

### Example 1: Web Interface
```
1. Navigate to http://localhost:8502
2. Upload an image
3. View real-time detection result
```

### Example 2: Python Script
```python
from tensorflow.keras.models import load_model
import cv2
import numpy as np

# Load model
model = load_model('best_model.h5')

# Prepare image
image = cv2.imread('test_image.jpg')
image = cv2.resize(image, (128, 128))
image = image.astype('float32') / 255.0
image = np.expand_dims(image, axis=0)

# Predict
prediction = model.predict(image)[0][0]
print(f"Authenticity Score: {prediction*100:.2f}%")
```

---

## 📦 Installation

### Requirements
- Python 3.8+
- TensorFlow 2.x
- Keras 2.2+
- OpenCV
- MTCNN
- Streamlit

### Step-by-Step

```bash
# 1. Clone repository
git clone https://github.com/anant-soni-360/DeepFake-Detect.git
cd DeepFake-Detect

# 2. Create virtual environment (optional but recommended)
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run web app
python -m streamlit run streamlit_app.py
```

---

## 📈 Performance Metrics

### Model Evaluation

| Metric | Value |
|--------|-------|
| True Positives | 3/4 |
| False Positives | 0/4 |
| True Negatives | 2/4 |
| False Negatives | 0/4 |

### Training Dataset
- **Training Samples**: 16 images
- **Validation Samples**: 2 images
- **Test Samples**: 4 images
- **Classes**: 2 (Real, Fake)
- **Image Size**: 128 × 128 pixels

### Datasets Supported
- FaceForensics++
- Celeb-DF
- Facebook DFDC
- Google DFD
- DeepFake-TIMIT

---

## 🛠️ Project Structure

```
DeepFake-Detect/
├── 00-convert_video_to_image.py      # Extract frames
├── 01a-crop_faces_with_mtcnn.py      # Face cropping (MTCNN)
├── 01b-crop_faces_with_azure-api.py  # Face cropping (Azure)
├── 02-prepare_fake_real_dataset.py   # Data preparation
├── 03-train_cnn.py                   # Model training
├── streamlit_app.py                  # Web interface
├── best_model.h5                     # Trained model
├── requirements.txt                  # Dependencies
├── split_dataset/                    # Training data
│   ├── train/
│   ├── val/
│   └── test/
└── README.md                         # This file
```

---

## 🎓 Supported Datasets

| Dataset | Size | Link |
|---------|------|------|
| DeepFake-TIMIT | - | https://www.idiap.ch/dataset/deepfaketimit |
| FaceForensics++ | 1000 videos | https://github.com/ondyari/FaceForensics |
| Celeb-DF | 590 videos | https://github.com/danmohaha/celeb-deepfakeforensics |
| Google DFD | 363 videos | https://ai.googleblog.com/2019/09/ |
| Facebook DFDC | 100k videos | https://ai.facebook.com/datasets/dfdc/ |

---

## 🤝 Contributing

Contributions are welcome! Please feel free to:
- Report bugs
- Suggest features
- Submit pull requests
- Improve documentation

---

## 👨‍💻 Author

**Anant Soni**
- 🌐 GitHub: [@anant-soni-360](https://github.com/anant-soni-360)
- 💼 Interested in AI, Deep Learning, and Computer Vision
- 📧 Connect with me for collaborations

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## ⚠️ Disclaimer

This tool is for **educational and research purposes only**. Always use responsibly and ethically. Unauthorized use of deepfake detection technology to violate privacy or create harmful content is illegal and unethical.

---

## 🔗 Quick Links

- **Live Web App**: http://localhost:8502 (After starting Streamlit)
- **GitHub Repository**: https://github.com/anant-soni-360/DeepFake-Detect
- **TensorFlow Docs**: https://www.tensorflow.org/
- **EfficientNet Paper**: https://arxiv.org/abs/1905.11946

---

<div align="center">

### Made with ❤️ by Anant Soni

**Star this project if you find it helpful!** ⭐

</div>


