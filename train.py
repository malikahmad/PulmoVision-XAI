"""Train the notebook's binary CNN using a local folder dataset.

Expected layout:
data/Chest_Xray/
  train/normal, train/penumonia (or pneumonia)
  test/normal,  test/penumonia (or pneumonia)
"""
from pathlib import Path
import argparse, json
import tensorflow as tf

IMG_SIZE = (128, 128)
BATCH_SIZE = 32
SEED = 42

def find_class_dir(root: Path, split: str, canonical: str):
    base = root / split
    candidates = [canonical]
    if canonical == "pneumonia":
        candidates += ["penumonia"]
    for name in candidates:
        if (base / name).is_dir():
            return base / name
    raise FileNotFoundError(f"Missing {canonical} folder under {base}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default="data/Chest_Xray")
    parser.add_argument("--epochs", type=int, default=15)
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    root = Path(args.data_dir)
    train_dir = root / "train"
    test_dir = root / "test"
    if not train_dir.exists() or not test_dir.exists():
        raise FileNotFoundError("Expected train/ and test/ directories inside --data-dir.")

    # Explicit label order: Normal=0, Pneumonia=1.
    train_ds = tf.keras.utils.image_dataset_from_directory(
        train_dir, labels="inferred", label_mode="binary",
        class_names=["normal", "pneumonia"] if (train_dir/"pneumonia").exists()
                    else ["normal", "penumonia"],
        image_size=IMG_SIZE, batch_size=args.batch_size, seed=SEED
    )
    test_ds = tf.keras.utils.image_dataset_from_directory(
        test_dir, labels="inferred", label_mode="binary",
        class_names=["normal", "pneumonia"] if (test_dir/"pneumonia").exists()
                    else ["normal", "penumonia"],
        image_size=IMG_SIZE, batch_size=args.batch_size, shuffle=False
    )
    autotune = tf.data.AUTOTUNE
    train_ds = train_ds.prefetch(autotune)
    test_ds = test_ds.prefetch(autotune)

    model = tf.keras.Sequential([
        tf.keras.layers.Input(shape=(128, 128, 3)),
        tf.keras.layers.Rescaling(1./255),
        tf.keras.layers.Conv2D(32, (3, 3), activation="relu", name="conv2d"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Conv2D(64, (3, 3), activation="relu", name="conv2d_1"),
        tf.keras.layers.MaxPooling2D(),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation="relu"),
        tf.keras.layers.Dense(1, activation="sigmoid")
    ])
    model.compile(
        optimizer="adam", loss="binary_crossentropy",
        metrics=["accuracy", tf.keras.metrics.Precision(name="precision"),
                 tf.keras.metrics.Recall(name="recall"), tf.keras.metrics.AUC(name="auc")]
    )
    Path("models").mkdir(exist_ok=True)
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "models/chest_xray.keras", monitor="val_auc", mode="max",
            save_best_only=True
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=4, restore_best_weights=True
        ),
        tf.keras.callbacks.CSVLogger("training_log.csv")
    ]
    history = model.fit(train_ds, validation_data=test_ds, epochs=args.epochs,
                        callbacks=callbacks)
    best = tf.keras.models.load_model("models/chest_xray.keras")
    metrics = best.evaluate(test_ds, return_dict=True)
    with open("models/evaluation.json", "w") as f:
        json.dump({k: float(v) for k, v in metrics.items()}, f, indent=2)
    print("Saved models/chest_xray.keras")
    print("Evaluation:", metrics)

if __name__ == "__main__":
    main()
