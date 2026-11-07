# PulmoVision XAI — VS Code Project

A local FastAPI application for the binary chest X-ray CNN in `Explainable_AI_4.ipynb`, with a responsive animated interface and Grad-CAM visualization.

> **Important:** This is an educational/research demonstration, not a medical device. Do not use it for diagnosis, screening, triage, or treatment decisions.

## 1. Project structure

```text
PulmoVision_XAI_VSCode/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── xai.py
│   └── static/
│       ├── index.html
│       ├── styles.css
│       └── app.js
├── data/
│   └── Chest_Xray/           # place dataset here
├── models/                   # place trained model here
├── train.py
├── requirements.txt
└── README.md
```

## 2. Open in VS Code

1. Extract the ZIP.
2. Open the extracted `PulmoVision_XAI_VSCode` folder in VS Code.
3. Open Terminal → New Terminal.

### Windows PowerShell

```powershell
py -3.10 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

TensorFlow compatibility varies by Python and operating system. Python 3.10 or 3.11 is recommended. If TensorFlow installation fails, use a supported Python environment and follow the TensorFlow installation guide.

## 3. Add the dataset

Copy your dataset so it has this structure:

```text
data/Chest_Xray/
├── train/
│   ├── normal/
│   └── penumonia/       # spelling from the source notebook is accepted
└── test/
    ├── normal/
    └── penumonia/
```

The code also accepts the correctly spelled `pneumonia` directory. Labels are explicitly ordered `normal` = 0 and `pneumonia` = 1.

## 4. Train the model

```powershell
python train.py --data-dir data/Chest_Xray --epochs 15
```

The best model is saved to `models/chest_xray.keras`, with evaluation metrics in `models/evaluation.json`.

**Validation note:** this starter script uses the provided `test` folder as validation during training and then reports metrics on that same folder. For a trustworthy final evaluation, create separate train/validation/test folders and only evaluate once on an untouched test set.

## 5. Use an existing notebook model

If you already have the model exported from the notebook, copy it to:

```text
models/chest_xray.h5
```

Or set a custom path before launching:

```powershell
$env:MODEL_PATH="C:\path\to\your\model.h5"
```

The architecture should accept 128×128 RGB images and output one sigmoid value, matching the notebook model. The model's first layer should include the notebook's `Rescaling(1./255)` preprocessing.

## 6. Run the application

```powershell
uvicorn app.main:app --reload
```

Open `http://127.0.0.1:8000`.

API endpoints:
- `GET /api/health` — model status
- `POST /api/predict` — upload an X-ray image and receive prediction plus original/heatmap/overlay images

## 7. Understanding the output

- The sigmoid score is interpreted as the model's Pneumonia-class score.
- Score ≥ 0.5 maps to Pneumonia; below 0.5 maps to Normal.
- The displayed confidence is the model score for its selected class. It is **not calibrated** and should not be interpreted as a clinical probability.
- Grad-CAM highlights image regions that contribute to the selected class score. It is not a segmentation mask, does not identify a confirmed lesion, and is not proof of model reasoning.

## 8. Troubleshooting

- **Model not loaded:** check `models/chest_xray.h5` exists, or set `MODEL_PATH`.
- **Grad-CAM error:** the model must contain at least one `Conv2D` layer and have a single sigmoid output.
- **Wrong predictions:** verify the training label order is `normal` then `pneumonia`; check the model preprocessing and training distribution.
- **TensorFlow install errors:** use Python 3.10/3.11 and a TensorFlow-supported OS/Python combination.
