
import base64
import io
import logging
from pathlib import Path

import cv2
import numpy as np
import tensorflow as tf

from PIL import Image


logger = logging.getLogger("PulmoVision")

CLASS_NAMES = ["Normal", "Pneumonia"]


def data_url(image: Image.Image) -> str:
    """Convert a PIL image into a base64 PNG data URL."""

    buffer = io.BytesIO()

    image.save(buffer, format="PNG")

    encoded = base64.b64encode(
        buffer.getvalue()
    ).decode("ascii")

    return f"data:image/png;base64,{encoded}"


class XRayExplainer:
    """
    Chest X-ray binary classifier with Grad-CAM.

    Expected model:
        - RGB input
        - 128 x 128 images
        - One sigmoid output
        - Normal = 0
        - Pneumonia = 1

    The original notebook model includes Rescaling(1./255),
    so this class passes image pixels in the 0-255 range.
    """

    def __init__(
        self,
        model_path: str,
        image_size=(128, 128)
    ):

        self.model_path = Path(model_path)

        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Model file does not exist: {self.model_path}"
            )

        self.image_size = tuple(image_size)

        logger.info("Loading model: %s", self.model_path)

        # --------------------------------------------------
        # LOAD MODEL
        # --------------------------------------------------

        self.model = tf.keras.models.load_model(
            str(self.model_path),
            compile=False
        )

        logger.info("Model loaded from disk.")

        # --------------------------------------------------
        # BUILD MODEL EXPLICITLY
        # --------------------------------------------------

        # Important fix:
        # Some Sequential models loaded from .keras files
        # have not established their symbolic graph yet.
        #
        # Calling the model with a real tensor builds it.

        dummy_input = tf.zeros(
            (
                1,
                self.image_size[0],
                self.image_size[1],
                3
            ),
            dtype=tf.float32
        )

        dummy_output = self.model(
            dummy_input,
            training=False
        )

        logger.info("Model built successfully.")

        # --------------------------------------------------
        # VALIDATE INPUT
        # --------------------------------------------------

        input_shape = tuple(self.model.inputs[0].shape)

        if len(input_shape) != 4:
            raise ValueError(
                f"Expected 4D image input. Received: {input_shape}"
            )

        model_height = input_shape[1]
        model_width = input_shape[2]
        channels = input_shape[3]

        if channels != 3:
            raise ValueError(
                f"Expected 3 RGB channels. Received: {channels}"
            )

        if model_height is not None:
            if int(model_height) != self.image_size[0]:
                raise ValueError(
                    f"Model expects height {model_height}, "
                    f"but configured height is {self.image_size[0]}."
                )

        if model_width is not None:
            if int(model_width) != self.image_size[1]:
                raise ValueError(
                    f"Model expects width {model_width}, "
                    f"but configured width is {self.image_size[1]}."
                )

        # --------------------------------------------------
        # VALIDATE OUTPUT
        # --------------------------------------------------

        output_shape = tuple(dummy_output.shape)

        if len(output_shape) != 2 or output_shape[-1] != 1:
            raise ValueError(
                "This application expects a binary classifier "
                f"with one sigmoid output. Received: {output_shape}"
            )

        # --------------------------------------------------
        # BUILD GRAD-CAM MODEL
        # --------------------------------------------------

        self._build_gradcam_model()

        logger.info("-----------------------------------")
        logger.info("PULMOVISION MODEL READY")
        logger.info("Input shape: %s", input_shape)
        logger.info("Output shape: %s", output_shape)
        logger.info("Grad-CAM layer: %s", self.last_conv_name)
        logger.info("-----------------------------------")

    def _build_gradcam_model(self):
        """
        Rebuild a symbolic forward graph using the existing
        trained layers and weights.

        Does not use model.output or layer.output.
        """

        symbolic_input = tf.keras.Input(
            shape=(
                self.image_size[0],
                self.image_size[1],
                3
            ),
            name="xray_input"
        )

        x = symbolic_input

        last_conv_output = None
        last_conv_name = None

        # Replay the original Sequential layers.
        # Existing layer weights are reused.
        for layer in self.model.layers:

            # Sequential models may expose an InputLayer.
            if isinstance(layer, tf.keras.layers.InputLayer):
                continue

            x = layer(x)

            if isinstance(
                layer,
                tf.keras.layers.Conv2D
            ):
                last_conv_output = x
                last_conv_name = layer.name

        if last_conv_output is None:
            raise ValueError(
                "No Conv2D layer was found. "
                "Grad-CAM requires a convolutional layer."
            )

        symbolic_output = x

        self.last_conv_name = last_conv_name

        # Create model that returns:
        # 1. Last convolutional activation
        # 2. Final classifier output

        self.grad_model = tf.keras.Model(
            inputs=symbolic_input,
            outputs=[
                last_conv_output,
                symbolic_output
            ],
            name="pulmovision_gradcam_model"
        )

        # Test graph execution
        test_tensor = tf.zeros(
            (
                1,
                self.image_size[0],
                self.image_size[1],
                3
            ),
            dtype=tf.float32
        )

        conv_result, prediction_result = self.grad_model(
            test_tensor,
            training=False
        )

        if len(conv_result.shape) != 4:
            raise ValueError(
                "Grad-CAM convolution output must be 4D. "
                f"Received: {conv_result.shape}"
            )

        if prediction_result.shape[-1] != 1:
            raise ValueError(
                "Grad-CAM model output is not a single binary score."
            )

        logger.info(
            "Grad-CAM graph built. Activation shape: %s",
            conv_result.shape
        )

    # ------------------------------------------------------
    # IMAGE PREPROCESSING
    # ------------------------------------------------------

    def _prepare(self, image: Image.Image):

        image = image.convert("RGB")

        resized = image.resize(
            (
                self.image_size[1],
                self.image_size[0]
            ),
            Image.Resampling.BILINEAR
        )

        array = np.asarray(
            resized,
            dtype=np.float32
        )

        # Keep 0-255 values.
        # The notebook's Rescaling layer handles normalization.

        tensor = np.expand_dims(
            array,
            axis=0
        )

        return tensor, resized

    # ------------------------------------------------------
    # GRAD-CAM
    # ------------------------------------------------------

    def _gradcam(
        self,
        tensor,
        class_index: int
    ):

        tensor = tf.convert_to_tensor(
            tensor,
            dtype=tf.float32
        )

        with tf.GradientTape() as tape:

            conv_output, prediction = self.grad_model(
                tensor,
                training=False
            )

            probability = prediction[:, 0]

            if class_index == 1:
                score = probability
            else:
                score = 1.0 - probability

        gradients = tape.gradient(
            score,
            conv_output
        )

        if gradients is None:
            raise RuntimeError(
                "Grad-CAM gradients are unavailable. "
                "The selected convolutional layer may not "
                "be connected to the model output."
            )

        # Global average pooling of gradients
        weights = tf.reduce_mean(
            gradients,
            axis=(1, 2),
            keepdims=True
        )

        # Weighted combination of feature maps
        cam = tf.reduce_sum(
            weights * conv_output,
            axis=-1
        )[0]

        # ReLU
        cam = tf.maximum(
            cam,
            0
        )

        maximum = tf.reduce_max(cam)

        cam = tf.where(
            maximum > 0,
            cam / (maximum + 1e-8),
            tf.zeros_like(cam)
        )

        return cam.numpy()

    # ------------------------------------------------------
    # PREDICTION
    # ------------------------------------------------------

    def predict(self, image: Image.Image):

        if not isinstance(image, Image.Image):
            raise TypeError(
                "Prediction requires a PIL Image."
            )

        # Prepare image
        tensor, resized = self._prepare(image)

        # Run classifier
        prediction = self.model(
            tensor,
            training=False
        )

        raw_score = float(
            np.asarray(prediction).reshape(-1)[0]
        )

        if not np.isfinite(raw_score):
            raise RuntimeError(
                "The model returned an invalid score."
            )

        if raw_score < 0 or raw_score > 1:
            raise RuntimeError(
                f"Expected sigmoid score between 0 and 1. "
                f"Received: {raw_score}"
            )

        # Binary classification
        class_index = (
            1 if raw_score >= 0.5 else 0
        )

        label = CLASS_NAMES[class_index]

        confidence = (
            raw_score
            if class_index == 1
            else 1.0 - raw_score
        )

        # Generate Grad-CAM
        heatmap = self._gradcam(
            tensor,
            class_index
        )

        original = np.asarray(
            resized,
            dtype=np.uint8
        )

        # Resize activation map to image dimensions
        heatmap = cv2.resize(
            heatmap,
            (
                original.shape[1],
                original.shape[0]
            ),
            interpolation=cv2.INTER_LINEAR
        )

        heatmap = np.clip(
            heatmap,
            0,
            1
        )

        heatmap_uint8 = np.uint8(
            heatmap * 255
        )

        # Apply color map
        colored = cv2.applyColorMap(
            heatmap_uint8,
            cv2.COLORMAP_JET
        )

        colored = cv2.cvtColor(
            colored,
            cv2.COLOR_BGR2RGB
        )

        # Blend original and heatmap
        overlay = cv2.addWeighted(
            original,
            0.58,
            colored,
            0.42,
            0
        )

        # Convert results into data URLs
        original_image = Image.fromarray(original)
        heatmap_image = Image.fromarray(colored)
        overlay_image = Image.fromarray(overlay)

        return {
            "label": label,
            "class_index": int(class_index),

            "score": round(
                raw_score,
                6
            ),

            "confidence": round(
                confidence * 100,
                2
            ),

            "model_output_note": (
                "Raw sigmoid score; not a calibrated "
                "clinical probability."
            ),

            "gradcam_layer": self.last_conv_name,

            "images": {
                "original": data_url(original_image),
                "heatmap": data_url(heatmap_image),
                "overlay": data_url(overlay_image)
            }
        }
