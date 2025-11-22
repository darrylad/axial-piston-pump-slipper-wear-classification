import os
import numpy as np
import tensorflow as tf
from tensorflow import keras
import sklearn.metrics as skm
from datetime import datetime
import json

# -----------------------------
# 1. Configuration
# -----------------------------

# Path to your CWT images
DATA_DIR = "/Users/darrylad/Darryl/Research/Axial Pison Pumps/NoisyData/CWTsnr10dbGaus"

# Path to the trained model (update this to your actual model path)
MODEL_PATH = "/Users/darrylad/Darryl/Research/Axial Pison Pumps/classification/outputs/20251122_190827/best_model.keras"

IMAGE_SIZE = (256, 256)
BATCH_SIZE = 32
SEED = 456  # Different seed for test split to ensure different data

# Create test outputs directory with human-readable timestamp
timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")  # e.g., 2025-11-22_19-08-25
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "test_outputs", timestamp)
os.makedirs(OUTPUT_DIR, exist_ok=True)

print(f"Test outputs will be saved to: {OUTPUT_DIR}")

# -----------------------------
# 2. Load the trained model
# -----------------------------

print(f"\nLoading model from: {MODEL_PATH}")
model = keras.models.load_model(MODEL_PATH)
print("Model loaded successfully!")
model.summary()

# Save model summary
with open(os.path.join(OUTPUT_DIR, "model_summary.txt"), "w") as f:
    model.summary(print_fn=lambda x: f.write(x + "\n"))

# -----------------------------
# 3. Create test dataset
# -----------------------------

# Option 1: Use validation_split with different seed for fresh test data
test_ds = tf.keras.utils.image_dataset_from_directory(
    DATA_DIR,
    labels="inferred",
    label_mode="int",
    color_mode="rgb",
    batch_size=BATCH_SIZE,
    image_size=IMAGE_SIZE,
    shuffle=True,
    seed=SEED,  # Different seed ensures different samples
    validation_split=0.2,
    subset="validation",
)

class_names = test_ds.class_names
print("\nClass names:", class_names)

# Configure for performance
AUTOTUNE = tf.data.AUTOTUNE
test_ds = test_ds.cache().prefetch(buffer_size=AUTOTUNE)

# Count test samples
test_sample_count = sum([x.shape[0] for x, _ in test_ds])
print(f"Total test samples: {test_sample_count}")

# -----------------------------
# 4. Evaluate on test set
# -----------------------------

print("\n" + "="*50)
print("EVALUATING MODEL ON TEST SET")
print("="*50)

test_loss, test_acc = model.evaluate(test_ds)
print(f"\nTest Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")

# Save evaluation results
with open(os.path.join(OUTPUT_DIR, "evaluation_results.txt"), "w") as f:
    f.write(f"Test Loss: {test_loss}\n")
    f.write(f"Test Accuracy: {test_acc}\n")
    f.write(f"Test Accuracy (%): {test_acc*100:.2f}%\n")
    f.write(f"Total Test Samples: {test_sample_count}\n")

# -----------------------------
# 5. Get predictions for detailed analysis
# -----------------------------

print("\nGenerating predictions...")
y_true = []
y_pred = []
y_probs = []

for images, labels in test_ds:
    probs = model.predict(images, verbose=0)
    preds = np.argmax(probs, axis=1)
    y_true.extend(labels.numpy())
    y_pred.extend(preds)
    y_probs.extend(probs)

y_true = np.array(y_true)
y_pred = np.array(y_pred)
y_probs = np.array(y_probs)

# -----------------------------
# 6. Confusion Matrix
# -----------------------------

print("\n" + "="*50)
print("CONFUSION MATRIX")
print("="*50)

cm = skm.confusion_matrix(y_true, y_pred)
print("\nConfusion Matrix:")
print(cm)
print(f"\nClass order: {class_names}")

# Save confusion matrix
with open(os.path.join(OUTPUT_DIR, "confusion_matrix.txt"), "w") as f:
    f.write("Confusion Matrix:\n")
    f.write("="*50 + "\n\n")
    f.write("Format: Rows = True Class, Columns = Predicted Class\n\n")
    
    # Header
    f.write("        ")
    for name in class_names:
        f.write(f"{name:>12}")
    f.write("\n")
    f.write("-" * (12 * len(class_names) + 8) + "\n")
    
    # Matrix with row labels
    for i, name in enumerate(class_names):
        f.write(f"{name:>8}")
        for j in range(len(class_names)):
            f.write(f"{cm[i][j]:>12}")
        f.write("\n")
    
    f.write("\n\nClass Names: " + str(class_names) + "\n")

# -----------------------------
# 7. Classification Report
# -----------------------------

print("\n" + "="*50)
print("CLASSIFICATION REPORT")
print("="*50)

classification_rep = skm.classification_report(y_true, y_pred, target_names=class_names)
print("\n" + classification_rep)

# Save classification report
with open(os.path.join(OUTPUT_DIR, "classification_report.txt"), "w") as f:
    f.write(classification_rep)

# -----------------------------
# 8. Per-class metrics (detailed)
# -----------------------------

print("\n" + "="*50)
print("DETAILED PER-CLASS METRICS")
print("="*50)

per_class_metrics = {}

for i, class_name in enumerate(class_names):
    # True positives, false positives, false negatives
    tp = cm[i, i]
    fp = cm[:, i].sum() - tp
    fn = cm[i, :].sum() - tp
    tn = cm.sum() - tp - fp - fn
    
    # Metrics
    accuracy = (tp + tn) / cm.sum() if cm.sum() > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    per_class_metrics[class_name] = {
        "support": int(cm[i, :].sum()),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_negatives": int(tn),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1_score": float(f1)
    }
    
    print(f"\n{class_name}:")
    print(f"  Support: {cm[i, :].sum()}")
    print(f"  True Positives: {tp}")
    print(f"  False Positives: {fp}")
    print(f"  False Negatives: {fn}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")

# Save detailed metrics
with open(os.path.join(OUTPUT_DIR, "detailed_metrics.json"), "w") as f:
    json.dump(per_class_metrics, f, indent=2)

# -----------------------------
# 9. Prediction confidence analysis
# -----------------------------

print("\n" + "="*50)
print("PREDICTION CONFIDENCE ANALYSIS")
print("="*50)

# Average confidence per class
confidence_by_class = {}
for i, class_name in enumerate(class_names):
    class_mask = y_true == i
    if class_mask.sum() > 0:
        avg_confidence = y_probs[class_mask, i].mean()
        confidence_by_class[class_name] = float(avg_confidence)
        print(f"{class_name}: {avg_confidence:.4f}")

# Overall confidence statistics
max_probs = y_probs.max(axis=1)
print(f"\nOverall confidence statistics:")
print(f"  Mean confidence: {max_probs.mean():.4f}")
print(f"  Min confidence: {max_probs.min():.4f}")
print(f"  Max confidence: {max_probs.max():.4f}")
print(f"  Std confidence: {max_probs.std():.4f}")

# Save confidence analysis
with open(os.path.join(OUTPUT_DIR, "confidence_analysis.json"), "w") as f:
    json.dump({
        "confidence_by_class": confidence_by_class,
        "overall_stats": {
            "mean": float(max_probs.mean()),
            "min": float(max_probs.min()),
            "max": float(max_probs.max()),
            "std": float(max_probs.std())
        }
    }, f, indent=2)

# -----------------------------
# 10. Find misclassified examples
# -----------------------------

misclassified = y_true != y_pred
num_misclassified = misclassified.sum()

print(f"\n" + "="*50)
print(f"MISCLASSIFICATION ANALYSIS")
print("="*50)
print(f"\nTotal misclassified: {num_misclassified} / {len(y_true)} ({num_misclassified/len(y_true)*100:.2f}%)")

if num_misclassified > 0:
    print("\nMisclassified examples:")
    misclassified_indices = np.where(misclassified)[0]
    for idx in misclassified_indices[:10]:  # Show first 10
        true_class = class_names[y_true[idx]]
        pred_class = class_names[y_pred[idx]]
        confidence = y_probs[idx, y_pred[idx]]
        print(f"  Sample {idx}: True={true_class}, Predicted={pred_class}, Confidence={confidence:.4f}")

# Save misclassification info
with open(os.path.join(OUTPUT_DIR, "misclassifications.txt"), "w") as f:
    f.write(f"Total misclassified: {num_misclassified} / {len(y_true)}\n")
    f.write(f"Misclassification rate: {num_misclassified/len(y_true)*100:.2f}%\n\n")
    
    if num_misclassified > 0:
        f.write("All misclassified examples:\n")
        f.write("-" * 80 + "\n")
        for idx in misclassified_indices:
            true_class = class_names[y_true[idx]]
            pred_class = class_names[y_pred[idx]]
            confidence = y_probs[idx, y_pred[idx]]
            f.write(f"Sample {idx}: True={true_class}, Predicted={pred_class}, Confidence={confidence:.4f}\n")

# -----------------------------
# 11. Summary metadata
# -----------------------------

with open(os.path.join(OUTPUT_DIR, "test_metadata.txt"), "w") as f:
    f.write(f"Test Run Timestamp: {timestamp}\n")
    f.write(f"Model Path: {MODEL_PATH}\n")
    f.write(f"Data Directory: {DATA_DIR}\n")
    f.write(f"Image Size: {IMAGE_SIZE}\n")
    f.write(f"Batch Size: {BATCH_SIZE}\n")
    f.write(f"Random Seed: {SEED}\n")
    f.write(f"Number of Classes: {len(class_names)}\n")
    f.write(f"Class Names: {class_names}\n")
    f.write(f"Total Test Samples: {test_sample_count}\n")
    f.write(f"\nTest Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)\n")
    f.write(f"Test Loss: {test_loss:.4f}\n")
    f.write(f"Misclassified: {num_misclassified} / {len(y_true)}\n")

print("\n" + "="*50)
print(f"All test outputs saved to: {OUTPUT_DIR}")
print("="*50)