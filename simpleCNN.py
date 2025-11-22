import os
import math
import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from datetime import datetime

# -----------------------------
# 1. Configuration
# -----------------------------

# Path to the root directory that contains the 5 class folders
DATA_DIR = "/Users/darrylad/Darryl/Projects/datapy/cwt-gen/outputs 0"   # <-- change this to the input folder

IMAGE_SIZE = (256, 256)           # images are already 256x256
BATCH_SIZE = 32
SEED = 123
NUM_CLASSES = 5

# Create outputs directory with timestamp
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs", timestamp)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Outputs will be saved to: {OUTPUT_DIR}")

# -----------------------------
# 2. Load train and validation datasets
# -----------------------------
# Here we let Keras do a 80/20 split; later we will further split
# the validation set into validation + test if needed.

train_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",           # integer labels 0..4
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.2,       # 80% train, 20% val
    subset="training",
)

val_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
    seed=SEED,
    validation_split=0.2,
    subset="validation",
)

# Split val_ds into val and test (50/50)
val_batches = tf.data.experimental.cardinality(val_ds)
test_ds = val_ds.take(val_batches // 2)
val_ds  = val_ds.skip(val_batches // 2)

class_names = train_ds.class_names
print("Class names:", class_names)

# -----------------------------
# 3. Performance optimizations
# -----------------------------

AUTOTUNE = tf.data.AUTOTUNE

def configure_for_performance(ds):
    ds = ds.cache()
    ds = ds.shuffle(1000, seed=SEED)
    ds = ds.prefetch(buffer_size=AUTOTUNE)
    return ds

train_ds = configure_for_performance(train_ds)
val_ds   = configure_for_performance(val_ds)
test_ds  = configure_for_performance(test_ds)

# -----------------------------
# 4. Compute class weights (handle imbalance)
# -----------------------------
# Iterate once through the original (un-shuffled) unbatched dataset to count labels.

# Recreate an unbatched dataset for counting
count_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=1,
    image_size=IMAGE_SIZE,
    shuffle=False
)

label_counts = dict.fromkeys(range(NUM_CLASSES), 0)

for _, labels in count_ds:
    label = int(labels.numpy()[0])
    label_counts[label] += 1

print("Label counts:", label_counts)

total_samples = sum(label_counts.values())
# Inverse-frequency class weights:
class_weights = {
    cls: (total_samples / (NUM_CLASSES * count))
    for cls, count in label_counts.items()
}

print("Class weights:", class_weights)

# -----------------------------
# 5. Data augmentation (gentle)
# -----------------------------
# For CWT images, avoid flips that invert frequency axis. You can still
# add small random contrast / brightness or tiny zooms along time.

data_augmentation = keras.Sequential(
    [
        layers.RandomBrightness(factor=0.1),
        layers.RandomContrast(factor=0.1),
        # layers.RandomZoom(0.05)  # optional, keep small
    ]
)

# -----------------------------
# 6. Build the CNN model
# -----------------------------

def make_model(input_shape=IMAGE_SIZE + (3,), num_classes=NUM_CLASSES):
    inputs = keras.Input(shape=input_shape)

    x = data_augmentation(inputs)
    x = layers.Rescaling(1.0 / 255.0)(x)   # scale pixels to [0,1]

    # Block 1
    x = layers.Conv2D(32, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 2
    x = layers.Conv2D(64, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Block 3
    x = layers.Conv2D(128, (3, 3), padding="same")(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2, 2))(x)

    # Global average pooling
    x = layers.GlobalAveragePooling2D()(x)

    # Dense layers
    x = layers.Dense(64, activation="relu")(x)
    x = layers.Dropout(0.3)(x)

    outputs = layers.Dense(num_classes, activation="softmax")(x)

    model = keras.Model(inputs, outputs, name="cwt_cnn")
    return model

model = make_model()
model.summary()

# Save model summary to file
with open(os.path.join(OUTPUT_DIR, "model_summary.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# -----------------------------
# 7. Compile the model
# -----------------------------

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=1e-3),
    loss="sparse_categorical_crossentropy",
    metrics=["accuracy"],
)

# -----------------------------
# 8. Train with early stopping
# -----------------------------

callbacks = [
    keras.callbacks.EarlyStopping(
        monitor="val_accuracy",
        patience=8,
        restore_best_weights=True
    ),
    keras.callbacks.ModelCheckpoint(
        os.path.join(OUTPUT_DIR, "best_model.keras"),
        monitor="val_accuracy",
        save_best_only=True
    ),
    keras.callbacks.CSVLogger(
        os.path.join(OUTPUT_DIR, "training_log.csv")
    )
]

EPOCHS = 40

history = model.fit(
    train_ds,
    epochs=EPOCHS,
    validation_data=val_ds,
    class_weight=class_weights,  # important for imbalance
    callbacks=callbacks,
)

# Save training history
import json
history_dict = {
    "loss": [float(x) for x in history.history["loss"]],
    "accuracy": [float(x) for x in history.history["accuracy"]],
    "val_loss": [float(x) for x in history.history["val_loss"]],
    "val_accuracy": [float(x) for x in history.history["val_accuracy"]]
}
with open(os.path.join(OUTPUT_DIR, "training_history.json"), "w") as f:
    json.dump(history_dict, f, indent=2)

# -----------------------------
# 9. Evaluate on the test set
# -----------------------------

test_loss, test_acc = model.evaluate(test_ds)
print("Test accuracy:", test_acc)

# Save evaluation results
with open(os.path.join(OUTPUT_DIR, "evaluation_results.txt"), "w") as f:
    f.write(f"Test Loss: {test_loss}\n")
    f.write(f"Test Accuracy: {test_acc}\n")

# Confusion matrix and classification report
import sklearn.metrics as skm

y_true = []
y_pred = []

for images, labels in test_ds:
    probs = model.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(preds)

cm = skm.confusion_matrix(y_true, y_pred)
print("Confusion matrix:\n", cm)

classification_rep = skm.classification_report(y_true, y_pred, target_names=class_names)
print("Classification report:\n", classification_rep)

# Save confusion matrix and classification report
with open(os.path.join(OUTPUT_DIR, "confusion_matrix.txt"), "w") as f:
    f.write("Confusion Matrix:\n")
    f.write(str(cm))
    f.write("\n\nClass Names Order: " + str(class_names) + "\n")

with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(classification_rep)

# Save metadata
with open(os.path.join(OUTPUT_DIR, "run_metadata.txt"), "w") as f:
    f.write(f"Timestamp: {timestamp}\n")
    f.write(f"Data Directory: {DATA_DIR}\n")
    f.write(f"Image Size: {IMAGE_SIZE}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Seed: {SEED}\n")
    f.write(f"Number of Classes: {NUM_CLASSES}\n")
    f.write(f"Class Names: {class_names}\n")
    f.write(f"Label Counts: {label_counts}\n")
    f.write(f"Class Weights: {class_weights}\n")
    f.write(f"Total Epochs Run: {len(history.history['loss'])}\n")

print(f"\nAll outputs saved to: {OUTPUT_DIR}")
