
from pathlib import Path

import tensorflow as tf

from PIL import Image

from app.xai import XRayExplainer


print("\n================================")
print("PULMOVISION MODEL TEST")
print("================================")

MODEL_PATH = Path("models/chest_xray.keras")

# Check model file
print("\n1. Checking model file...")

if not MODEL_PATH.exists():
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH.resolve()}"
    )

print("Model found:", MODEL_PATH.resolve())

# TensorFlow version
print("\n2. TensorFlow version:")

print(tf.__version__)

# Load explainer
print("\n3. Loading model and Grad-CAM...")

explainer = XRayExplainer(
    str(MODEL_PATH),
    image_size=(128, 128)
)

print("Model loaded successfully.")

# Display architecture information
print("\n4. Model information:")

print("Input:", explainer.model.inputs[0].shape)

print("Output:", explainer.model.outputs[0].shape)

print("Grad-CAM layer:", explainer.last_conv_name)

# Synthetic software smoke test
print("\n5. Running prediction smoke test...")

test_image = Image.new(
    "RGB",
    (128, 128),
    color=(120, 120, 120)
)

result = explainer.predict(test_image)

print("\nPrediction result:")

print("Label:", result["label"])

print("Class index:", result["class_index"])

print("Score:", result["score"])

print("Confidence:", result["confidence"])

print("Grad-CAM layer:", result["gradcam_layer"])

print("\n6. Checking generated images:")

for name, value in result["images"].items():

    print(
        name,
        "generated:",
        value.startswith("data:image/png;base64,")
    )

print("\n================================")
print("MODEL TEST COMPLETED")
print("================================")
