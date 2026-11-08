# PulmoVision XAI

**Chest X-ray image classification with FastAPI, TensorFlow, and Grad-CAM explainability.**

PulmoVision XAI is a local research prototype that wraps a binary chest X-ray CNN in a FastAPI application. It predicts the model's Normal or Pneumonia class and generates Grad-CAM visualizations to help inspect image regions associated with the model output.

> **Medical safety notice:** This is an educational/research demonstration, not a medical device. It must not be used for diagnosis, screening, triage, treatment, or other clinical decisions. A model score is not a calibrated clinical probability. Always consult a qualified healthcare professional.

## Features

- FastAPI application with image upload and JSON prediction endpoint.
- Binary image classification: Normal and Pneumonia.
- TensorFlow/Keras model loading.
- Grad-CAM heatmap and overlay generation.
- Health endpoint reporting model availability and selected Grad-CAM convolution layer.
- Browser interface served from the application's static directory.
- Configurable model path using the MODEL_PATH environment variable.
- Training script for a local folder-based dataset.

## Technology

| Area | Technology |
|---|---|
| Language | Python |
| API | FastAPI, Uvicorn |
| Deep learning | TensorFlow / Keras |
| Explainability | Grad-CAM |
| Image processing | OpenCV, NumPy, Pillow |
| Frontend | HTML, CSS, JavaScript |

## How it works

1. The FastAPI application loads a Keras model from models/chest_xray.keras by default.
2. The uploaded image is validated and converted to RGB.
3. The image is resized to 128 × 128 pixels and passed to the model.
4. The CNN returns a single sigmoid score.
5. The application maps the score to Normal or Pneumonia using a 0.5 threshold.
6. Grad-CAM uses the final convolutional layer to create a heatmap and overlay.

The model is expected to accept 128 × 128 RGB images and produce one sigmoid output, with Normal = 0 and Pneumonia = 1. The original model preprocessing includes Rescaling(1./255); the inference code therefore supplies pixel values in the 0–255 range.

## Repository structure

~~~text
PulmoVision-XAI/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── xai.py
│   └── static/
│       ├── index.html
│       ├── styles.css
│       └── app.js
├── data/
│   └── Chest_Xray/          # local dataset; not included
├── models/
│   └── chest_xray.keras     # generated model; not included
├── train.py
├── requirements.txt
└── README.md
~~~

Dataset and trained model artifacts are not included. Add them locally following the instructions below.

## Installation

TensorFlow support depends on Python version, operating system, and hardware. Python 3.10 or 3.11 is recommended; choose a TensorFlow release compatible with your environment.

### 1. Clone the repository

~~~bash
git clone https://github.com/malikahmad/PulmoVision-XAI.git
cd PulmoVision-XAI
~~~

### 2. Create a virtual environment

**Windows PowerShell**

~~~powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
~~~

**Linux / macOS**

~~~bash
python3 -m venv .venv
source .venv/bin/activate
~~~

### 3. Install dependencies

~~~bash
python -m pip install --upgrade pip
pip install -r requirements.txt
~~~

If TensorFlow installation fails, verify the supported Python and operating-system combination for the TensorFlow version available to your platform.

## Dataset preparation

Place the dataset under data/Chest_Xray/ with separate training and test folders:

~~~text
data/Chest_Xray/
├── train/
│   ├── normal/
│   └── pneumonia/       # "penumonia" is also accepted by the script
└── test/
    ├── normal/
    └── pneumonia/
~~~

The training script explicitly orders the classes as Normal = 0 and Pneumonia = 1.

## Train the model

~~~bash
python train.py --data-dir data/Chest_Xray --epochs 15
~~~

Optional batch size:

~~~bash
python train.py --data-dir data/Chest_Xray --epochs 15 --batch-size 32
~~~

The script saves:

- models/chest_xray.keras — best checkpoint selected by validation AUC.
- models/evaluation.json — metrics produced by the script.
- training_log.csv — epoch-level training history.

### Evaluation caveat

**Important:** The current training script uses the supplied test directory as validation_data during model fitting and then evaluates the selected model on that same directory. Those reported metrics are therefore not an independent final test result. For a trustworthy estimate, create three distinct splits:

- Training set — used to update model weights.
- Validation set — used for checkpoint selection and tuning.
- Untouched test set — used once for final reporting.

Check patient-level separation where relevant, class balance, duplicates, and possible data leakage. Report per-class precision, recall, F1-score, confusion matrix, and ROC-AUC. Do not advertise a model accuracy number without a reproducible evaluation protocol.

## Run the application

After placing a compatible model at models/chest_xray.keras:

~~~bash
uvicorn app.main:app --reload
~~~

Open:

- Web application: http://127.0.0.1:8000/
- API documentation: http://127.0.0.1:8000/docs
- Health check: http://127.0.0.1:8000/api/health

Use a production ASGI server configuration and disable development reload before deploying publicly.

## API reference

### Health

GET /api/health

Returns application status, whether the model is loaded, whether the model file exists, any loading error, and the selected Grad-CAM layer when available.

### Predict

POST /api/predict

Send a multipart form request with a file field named file:

~~~bash
curl -X POST "http://127.0.0.1:8000/api/predict" \
  -F "file=@chest-xray.png"
~~~

The response contains the model prediction and generated visualization data returned by the explainer. Uploads are limited to 12 MB; the application accepts JPEG, PNG, and WEBP image formats.

## Understanding Grad-CAM

Grad-CAM computes a coarse heatmap from gradients flowing into the final convolutional feature maps. It is overlaid on the input image to support visual inspection of model behavior.

A heatmap:

- Is not a segmentation mask.
- Does not identify or confirm a lesion.
- Does not establish that the model used clinically meaningful evidence.
- Can be misleading when the model is poorly trained or the input differs from its training distribution.

The displayed class score is the model's sigmoid output (or its complement for the selected class). It is not calibrated and must not be interpreted as a patient's probability of disease.

## Troubleshooting

**Model unavailable**
- Confirm models/chest_xray.keras exists.
- Set MODEL_PATH to a valid model file if it is stored elsewhere.
- Check the application log for TensorFlow loading errors.

**Grad-CAM error**
- The model must contain at least one Conv2D layer.
- The expected output is a single sigmoid score.
- Confirm the model accepts 128 × 128 RGB inputs.

**Unexpected class mapping**
- Confirm the training data folders are ordered Normal then Pneumonia.
- Check preprocessing and input dimensions against the saved model.

## Responsible use and limitations

- Dataset representativeness and image quality affect predictions.
- The model may not generalize to other hospitals, scanners, populations, or acquisition protocols.
- Grad-CAM visualizations are explanatory aids, not proof of model reasoning.
- No clinical validation or regulatory approval is claimed.

## Author

**Ahmad Zulfiqar**  
GitHub: [@malikahmad](https://github.com/malikahmad)
