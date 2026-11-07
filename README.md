# 🩻 PulmoVision XAI
### Explainable AI for Chest X-Ray Pneumonia Classification Using CNN and Grad-CAM

PulmoVision XAI is a deep learning-based web application designed to classify chest X-ray images into **Normal** and **Pneumonia** categories using a Convolutional Neural Network (CNN).

The project integrates **Explainable Artificial Intelligence (XAI)** through Grad-CAM (Gradient-weighted Class Activation Mapping), generating visual heatmaps to highlight image regions that contribute to the model's prediction.

The application provides an interactive interface where users can upload chest X-ray images, receive classification results, view prediction scores, and explore visual explanations.

## 🚀 Key Features

- **Chest X-Ray Classification:** Classifies uploaded images as Normal or Pneumonia.
- **CNN Deep Learning Model:** Uses a trained TensorFlow/Keras image classification model.
- **Grad-CAM Visualization:** Generates heatmaps highlighting influential image regions.
- **Visual Explanations:** Displays original images, heatmaps, and Grad-CAM overlays.
- **Prediction Confidence:** Displays the model's classification score.
- **REST API:** FastAPI backend for image processing and model inference.
- **Image Upload:** Supports JPG, PNG, and WEBP images.
- **Interactive Web Interface:** Responsive frontend for model interaction.
- **Model Integration:** Loads an existing trained `.keras` model without requiring retraining.

## 🧠 AI Model Architecture

The project uses a CNN-based binary image classification model.

The model architecture includes:
- Image rescaling
- Convolutional layers
- Max pooling layers
- Flatten layer
- Fully connected Dense layer
- Sigmoid output layer

**Classification mapping:**
- Class 0: Normal
- Class 1: Pneumonia

The trained model is stored in Keras format:

`chest_xray.keras`

## 🔍 Explainable AI – Grad-CAM

PulmoVision integrates Grad-CAM to provide visual explanations of CNN predictions.

It generates three visual outputs:

1. Original chest X-ray image.
2. Grad-CAM heatmap.
3. Heatmap overlay on the original image.

These visualizations help explore which image regions influence the model's classification. They are not lesion segmentation or proof of pneumonia.

## 🛠️ Tech Stack

| Category | Technologies |
|---|---|
| Programming Language | Python |
| Deep Learning | TensorFlow, Keras |
| Model Architecture | CNN |
| Explainable AI | Grad-CAM |
| Backend | FastAPI |
| Image Processing | OpenCV, Pillow |
| Numerical Computing | NumPy |
| Frontend | HTML, CSS, JavaScript |
| Model Format | Keras `.keras` |

## 📁 Project Structure

```text
PulmoVision-XAI/
│
├── app/
│   ├── main.py
│   ├── xai.py
│   │
│   └── static/
│       ├── index.html
│       ├── style.css
│       └── script.js
│
├── models/
│   └── chest_xray.keras
│
├── test_model.py
├── requirements.txt
└── README.md
```

## ⚙️ Installation and Setup

### 1. Clone the repository

```bash
git clone https://github.com/YOUR-USERNAME/PulmoVision-XAI.git

cd PulmoVision-XAI
```

### 2. Create a virtual environment

```bash
python -m venv .venv
```

Activate on Windows:

```powershell
.venv\Scripts\activate
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Add the trained model

Place your trained model inside:

```text
models/chest_xray.keras
```

### 5. Run the model test

```bash
python test_model.py
```

### 6. Start the FastAPI application

```bash
uvicorn app.main:app --reload
```

### 7. Open the application

```text
http://127.0.0.1:8000
```

API documentation:

```text
http://127.0.0.1:8000/docs
```

## 📊 Project Objectives

- Develop an automated chest X-ray image classification system.
- Integrate deep learning into medical image analysis research.
- Explore model interpretability using Grad-CAM.
- Build an interactive web application for AI model inference.
- Demonstrate the integration of CNN-based classification and Explainable AI.

## 🔬 Future Improvements

- Evaluate performance on a properly separated held-out test dataset.
- Add accuracy, precision, recall, F1-score, and ROC-AUC reporting.
- Improve generalization and address class imbalance.
- Explore Transformer-based medical image classification.
- Add model comparison and evaluation dashboards.
- Improve Grad-CAM visualization and interpretability.
- Deploy the application to a cloud platform.

## ⚠️ Disclaimer

PulmoVision XAI is an educational and research project. It is not a clinically validated medical device and must not be used for diagnosis, treatment decisions, or replacement of professional medical advice.

Model scores are not calibrated clinical probabilities. Grad-CAM visualizations are interpretability aids, not proof of disease or lesion localization.

## 👨‍💻 Author

**Ahmad Zulfiqar**

AI/ML Engineer | Deep Learning | Computer Vision | Explainable AI | Generative AI

GitHub: https://github.com/malikahmad

---

⭐ If you find this project interesting, consider giving the repository a star!
