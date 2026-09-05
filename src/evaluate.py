"""
DeepFER — Evaluation & Charts (local version of notebook cells 8,10,32-39)

Regenerates every result artifact from the Kaggle notebook using your
ALREADY-TRAINED models — no training happens here.

Produces, into outputs/charts/:
  - class_distribution.png
  - sample_images.png
  - confusion_matrix_cnn.png
  - confusion_matrix_tl.png
  - roc_curves_cnn.png
  - roc_curves_tl.png
  - metric_comparison.png
Plus prints full classification reports to the console.

NOTE ON TRAINING HISTORY (accuracy/loss-vs-epoch) CHARTS:
The Kaggle notebook plotted `history.history['accuracy']` etc. straight
from the Keras History object returned by `model.fit(...)`. That object
only exists in memory during training and the notebook never saved it to
a file — so those two specific curve charts cannot be regenerated from
the downloaded artifacts alone. If you still have your Kaggle session
running/history in memory, use `File > Download` on that plot cell there.
Everything else below is fully reproducible locally.
"""
import os
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow import keras
from sklearn.metrics import (confusion_matrix, roc_curve, auc,
                              classification_report, accuracy_score,
                              precision_recall_fscore_support)
from sklearn.preprocessing import label_binarize

from . import config as C
from . import data_pipeline as dp


def save(fig_name):
    path = os.path.join(C.CHARTS_DIR, fig_name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"Saved {path}")


def plot_class_distribution():
    if not os.path.isdir(C.TRAIN_DIR):
        print("Skipping class distribution — data/train/<class>/ not found.")
        return
    counts = {c: len(os.listdir(os.path.join(C.TRAIN_DIR, c)))
              for c in C.CLASS_NAMES if os.path.isdir(os.path.join(C.TRAIN_DIR, c))}
    if not counts:
        print("Skipping class distribution — no class subfolders found in data/train/.")
        return
    plt.figure(figsize=(9, 4))
    sns.barplot(x=list(counts.keys()), y=list(counts.values()), palette="viridis")
    plt.title("Training set class distribution")
    plt.ylabel("Image count")
    plt.xticks(rotation=30)
    save("class_distribution.png")


def plot_sample_images():
    if not os.path.isdir(C.TRAIN_DIR):
        print("Skipping sample images — data/train/<class>/ not found.")
        return
    fig, axes = plt.subplots(1, len(C.CLASS_NAMES), figsize=(18, 3))
    any_found = False
    for ax, c in zip(axes, C.CLASS_NAMES):
        folder = os.path.join(C.TRAIN_DIR, c)
        if os.path.isdir(folder) and os.listdir(folder):
            fname = os.listdir(folder)[0]
            img = plt.imread(os.path.join(folder, fname))
            ax.imshow(img, cmap='gray')
            any_found = True
        ax.set_title(c)
        ax.axis('off')
    if any_found:
        save("sample_images.png")
    else:
        plt.close()
        print("Skipping sample images — no images found.")


def plot_confusion(y_true, y_pred, title, fname):
    cm = confusion_matrix(y_true, y_pred)
    cm_norm = cm.astype('float') / cm.sum(axis=1, keepdims=True)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=C.CLASS_NAMES, yticklabels=C.CLASS_NAMES)
    plt.title(f'Confusion Matrix (normalized) — {title}')
    plt.xlabel('Predicted'); plt.ylabel('True')
    save(fname)


def plot_roc(y_true, probs, title, fname):
    y_bin = label_binarize(y_true, classes=list(range(C.NUM_CLASSES)))
    plt.figure(figsize=(8, 6))
    for i, cname in enumerate(C.CLASS_NAMES):
        fpr, tpr, _ = roc_curve(y_bin[:, i], probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"{cname} (AUC={roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], 'k--', alpha=0.4)
    plt.xlabel('False Positive Rate'); plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curves (one-vs-rest) — {title}')
    plt.legend(loc='lower right', fontsize=8)
    save(fname)


def full_report(y_true, y_pred, model_name):
    acc = accuracy_score(y_true, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_true, y_pred, average='weighted')
    print(f"===== {model_name} =====")
    print(f"Accuracy : {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall   : {recall:.4f}")
    print(f"F1-score : {f1:.4f}\n")
    print(classification_report(y_true, y_pred, target_names=C.CLASS_NAMES))
    return {"model": model_name, "accuracy": acc, "precision": precision, "recall": recall, "f1": f1}


def plot_metric_comparison(comparison_df):
    comparison_df.set_index('model')[['accuracy', 'precision', 'recall', 'f1']].plot(
        kind='bar', figsize=(9, 5), rot=0, colormap='viridis'
    )
    plt.title("Custom CNN vs Transfer Learning — Metric Comparison")
    plt.ylabel("Score"); plt.ylim(0, 1); plt.grid(axis='y', alpha=0.3)
    save("metric_comparison.png")


def evaluate_models_on_test_set():
    """Runs both saved models against data/test/ and returns fresh metrics.
    Requires data/test/<class>/*.jpg to be present locally."""
    cnn_path = C.CNN_BEST_PATH if os.path.exists(C.CNN_BEST_PATH) else C.CNN_FINAL_PATH
    tl_path = C.TL_BEST_PATH if os.path.exists(C.TL_BEST_PATH) else C.TL_FINAL_PATH

    if not os.path.exists(cnn_path) or not os.path.exists(tl_path):
        print("Model files not found in models/ — skipping live re-evaluation.")
        return None

    print("Loading models...")
    cnn_model = keras.models.load_model(cnn_path)
    tl_model = keras.models.load_model(tl_path)

    print("Preparing CNN test generator...")
    test_gen_cnn = dp.get_test_generator_cnn()
    y_true_cnn = test_gen_cnn.classes
    probs_cnn = cnn_model.predict(test_gen_cnn, verbose=1)
    y_pred_cnn = np.argmax(probs_cnn, axis=1)

    print("Preparing MobileNetV2 (TL) test dataset...")
    tl_gen, tl_ds = dp.get_test_generator_tl()
    steps = int(np.ceil(tl_gen.samples / C.BATCH_SIZE))
    probs_tl = tl_model.predict(tl_ds, steps=steps, verbose=1)[:tl_gen.samples]
    y_true_tl = tl_gen.classes
    y_pred_tl = np.argmax(probs_tl, axis=1)

    metrics_cnn = full_report(y_true_cnn, y_pred_cnn, "Custom CNN")
    metrics_tl = full_report(y_true_tl, y_pred_tl, "MobileNetV2 Transfer Learning")

    plot_confusion(y_true_cnn, y_pred_cnn, "Custom CNN", "confusion_matrix_cnn.png")
    plot_confusion(y_true_tl, y_pred_tl, "MobileNetV2 TL", "confusion_matrix_tl.png")
    plot_roc(y_true_cnn, probs_cnn, "Custom CNN", "roc_curves_cnn.png")
    plot_roc(y_true_tl, probs_tl, "MobileNetV2 TL", "roc_curves_tl.png")

    return pd.DataFrame([metrics_cnn, metrics_tl])


def main():
    print("=== DeepFER local evaluation & chart regeneration ===\n")

    plot_class_distribution()
    plot_sample_images()

    comparison_df = evaluate_models_on_test_set()

    if comparison_df is None:
        # Fall back to the metrics already saved in models/model_comparison.csv
        if os.path.exists(C.COMPARISON_CSV):
            print(f"Using existing metrics from {C.COMPARISON_CSV}")
            comparison_df = pd.read_csv(C.COMPARISON_CSV)
        else:
            print("No model_comparison.csv found either — cannot plot metric comparison.")
            comparison_df = None

    if comparison_df is not None:
        plot_metric_comparison(comparison_df)
        comparison_df.to_csv(os.path.join(C.OUTPUT_DIR, "model_comparison.csv"), index=False)
        print(f"\nWrote refreshed model_comparison.csv to {C.OUTPUT_DIR}")

    print("\nDone. Check the outputs/charts/ folder for all images.")


if __name__ == "__main__":
    main()
