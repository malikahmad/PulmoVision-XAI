# PulmoVision XAI — Explainable Chest X-ray Classification Demo

PulmoVision XAI is a local FastAPI application for a binary chest X-ray CNN. It exposes model inference through an API and provides Grad-CAM visualizations to help inspect image regions associated with the model output.

> **Medical safety notice:** This is an educational and research demonstration, not a medical device. Do not use it for diagnosis, screening, triage, or treatment decisions. It is not a substitute for evaluation by a qualified clinician.

## Features

- FastAPI backend with a browser-based interface.
- Binary chest X-ray classification workflow.
- Grad-CAM heatmap and overlay generation.
- Model health endpoint.
- Configurable model path.
- Training script for a local dataset.
- Evaluation output saved by the training workflow.

## Technology

| Area | Technology |
|---|---|
| Language | Python |
| Deep learning | TensorFlow / Keras |
| API | FastAPI, Uvicorn |
| Explainability | Grad-CAM |
| Image processing | Pillow |
| Frontend | HTML, CSS, JavaScript |

## Project structure

```text
PulmoVision_XAI/
├── app/
│   ├── __init__.py
│   ├── main.py
│   ├── xai.py
│   └── static/
│       ├── index.html
│       ├── styles.css
│       └── app.js
├── data/
│   └── Chest_Xray/
├── models/
├── train.py
├── requirements.txt
└── README.md
```

## Setup

Use Python 3.10 or 3.11 with a TensorFlow build supported by your operating system.

Windows PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -r requirements.txt
```

## Dataset layout

Place the dataset under `data/Chest_Xray/`. The training script expects training and test directories with normal and pneumonia class folders. The source project accepts the historical `penumonia` spelling as well as `pneumonia`.

```text
data/Chest_Xray/
├── train/
│   ├── normal/
│   └── pneumonia/
└── test/
    ├── normal/
    └── pneumonia/
```

## Train

```powershell
python train.py --data-dir data/Chest_Xray --epochs 15
```

The training workflow saves the best model to `models/chest_xray.keras` and evaluation information to `models/evaluation.json`.

**Evaluation warning:** The current starter workflow uses the provided test folder as validation during training and then reports metrics on that same folder. Those results are not an independent final test. Create separate train, validation, and untouched test splits before reporting generalization performance.

## Run the API

If you already have a compatible model checkpoint, place it at `models/chest_xray.h5` or configure `MODEL_PATH`.

```powershell
uvicorn app.main:app --reload
```

Open [http://127.0.0.1:8000](http://127.0.0.1:8000).

API endpoints:

- `GET /api/health` — model status.
- `POST /api/predict` — image upload, prediction, and explainability outputs.

## Interpreting outputs

- The sigmoid output is treated as the model's pneumonia-class score.
- The 0.5 threshold maps scores to the two configured classes.
- The displayed confidence is a model score, not a calibrated clinical probability.
- Grad-CAM is a visualization aid, not a lesion segmentation or proof of clinical reasoning.

## Author

**Ahmad Zulfiqar** — [GitHub](https://github.com/malikahmad) · [LinkedIn](https://www.linkedin.com/in/malik-ahmad-06abb31a1/)
