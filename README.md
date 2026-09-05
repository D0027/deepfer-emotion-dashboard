# DeepFER — Local Project

This is your Kaggle "DeepFER: Facial Emotion Recognition" notebook, restructured
into a runnable local project. Your models are **already trained** — nothing
here retrains them by default.

## 1. Folder structure

```
deepfer_local/
├── data/
│   ├── train/<class>/*.jpg      ← put FER2013 train images here (optional, only needed for charts/retraining)
│   └── test/<class>/*.jpg       ← put FER2013 test images here (needed to regenerate confusion matrix / ROC)
├── models/                      ← PUT YOUR DOWNLOADED KAGGLE FILES HERE
│   ├── deepfer_custom_cnn_best.keras
│   ├── deepfer_custom_cnn_final.keras
│   ├── deepfer_mobilenetv2_best.keras
│   ├── deepfer_mobilenetv2_final.keras
│   ├── deepfer_deploy_model.tflite
│   ├── class_names.json
│   └── model_comparison.csv
├── outputs/
│   └── charts/                  ← regenerated charts land here
├── src/
│   ├── config.py                 paths & hyperparameters
│   ├── data_pipeline.py          FER2013 loading (dir- or CSV-based)
│   ├── models.py                 CNN + MobileNetV2 architectures
│   ├── evaluate.py                ★ regenerates all charts/metrics
│   ├── app_gradio.py              ★ local Gradio demo app
│   ├── export_tflite.py          optional — rebuild the .tflite file
│   ├── train_cnn.py              optional — retrain custom CNN
│   └── train_tl.py               optional — retrain MobileNetV2
├── run_evaluation.py             one-shot entry point
└── requirements.txt
```

## 2. Setup

```bash
cd deepfer_local
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

## 3. Put your files in place

1. Copy the 7 files you downloaded from Kaggle's **Output** panel
   (`deepfer_custom_cnn_best.keras`, `deepfer_custom_cnn_final.keras`,
   `deepfer_mobilenetv2_best.keras`, `deepfer_mobilenetv2_final.keras`,
   `deepfer_deploy_model.tflite`, `class_names.json`, `model_comparison.csv`)
   into `models/`.
2. (Optional, needed only for the class-distribution/sample-image/confusion-matrix/ROC
   charts) Download the FER2013 dataset locally and arrange it as:
   `data/train/<angry|disgust|fear|happy|neutral|sad|surprise>/*.jpg` and
   the same under `data/test/`. The common Kaggle copy is `msambare/fer2013`.

## 4. Regenerate all results & charts (no training)

```bash
python run_evaluation.py
```

This will:
- Load your saved `.keras` models
- Re-run them on `data/test/` to get fresh predictions
- Save every chart into `outputs/charts/`:
  `class_distribution.png`, `sample_images.png`,
  `confusion_matrix_cnn.png`, `confusion_matrix_tl.png`,
  `roc_curves_cnn.png`, `roc_curves_tl.png`, `metric_comparison.png`
- Print full precision/recall/F1 classification reports to the console
- Write a refreshed `model_comparison.csv` into `outputs/`

If `data/test/` isn't populated yet, it will skip the live re-evaluation and
just re-plot the metric comparison chart from your existing `models/model_comparison.csv`.

### Original Kaggle charts (already included)
`outputs/charts/` is pre-populated with the **exact original PNGs extracted
directly from your Kaggle notebook's saved cell outputs** — including the
accuracy/loss-vs-epoch training curves, which only exist there (that data was
in-memory during `model.fit()` and never saved separately):

- `class_distribution.png`
- `sample_images.png`
- `training_history_cnn.png`
- `training_history_tl_head.png`
- `training_history_tl_finetune.png`
- `confusion_matrix_cnn.png`, `confusion_matrix_tl.png`
- `roc_curves_cnn.png`, `roc_curves_tl.png`
- `metric_comparison.png`

Running `run_evaluation.py` will overwrite the non-training-history ones with
freshly regenerated versions from your local test set (if `data/test/` is
populated) — the training-history PNGs are left untouched since they can't be
regenerated.

## 5. Run the live demo app

```bash
python -m src.app_gradio
```

Opens a local Gradio UI at `http://127.0.0.1:7860` where you upload a photo
and get a live emotion prediction (face-detected via OpenCV Haar cascade,
classified with your custom CNN).

## 6. Optional — retrain or re-export

You almost certainly don't need these since your models are already trained:

```bash
python -m src.train_cnn        # retrain the custom CNN from scratch
python -m src.train_tl         # retrain MobileNetV2 (2-phase fine-tuning)
python -m src.export_tflite    # rebuild the quantized .tflite file
```
