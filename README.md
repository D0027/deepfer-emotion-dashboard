<div align="center">

# 🎭 DeepFER
### Facial Emotion Recognition — Custom CNN vs. MobileNetV2 Transfer Learning

*Real-time emotion detection powered by TensorFlow/Keras, deployed as an interactive Streamlit dashboard.*

[![Live Demo](https://img.shields.io/badge/🚀_Live_Demo-Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)](https://deepfer-emotion-dashboard-027.streamlit.app/)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=for-the-badge&logo=python&logoColor=white)](#)
[![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)](#)
[![License](https://img.shields.io/badge/License-MIT-blueviolet?style=for-the-badge)](#-license)

<br>

**[🌐 Try the Live App](https://deepfer-emotion-dashboard-027.streamlit.app/)** · **[📊 Dataset: FER2013](#-dataset)** · **[🧠 Models](#-models)** · **[⚡ Quickstart](#-quickstart)**

</div>

<br>

<div align="center">

| 🎯 7 Emotion Classes | 🧠 2 Model Architectures | 📦 TFLite Ready | ⚡ Real-Time Inference |
|:---:|:---:|:---:|:---:|
| angry · disgust · fear · happy · neutral · sad · surprise | Custom CNN + MobileNetV2 | Quantized for edge/mobile | Live webcam-style demo |

</div>

---

## ✨ Overview

**DeepFER** classifies human facial emotions from images using two competing deep learning approaches trained on the **FER2013** dataset — a **Custom CNN built from scratch** and a **MobileNetV2 transfer-learning model**. Both are benchmarked head-to-head on accuracy, precision, recall, F1, confusion matrices, and ROC/AUC, and the winner is exported as a lightweight, quantized **TFLite** model for fast, real-world deployment.

The project ships as a fully interactive **5-panel Streamlit dashboard** — not just a notebook dump — so anyone can explore the methodology, inspect the dataset, dig into model internals, review results, and run live predictions, all in the browser.

> ### 🔗 [**Launch the live dashboard →**](https://deepfer-emotion-dashboard-027.streamlit.app/)

---

## 🖥️ Dashboard Tour

The app is organized into five panels, accessible from the sidebar:

| Panel | What you'll find |
|---|---|
| 🧭 **Overview & Methodology** | Project KPIs, the full 10-step ML pipeline, key findings & challenges |
| 📊 **Dataset Review** | Class distribution charts, per-class sample gallery (train/test), live folder counts, one-click "test this image" |
| 🏗️ **Model Architecture** | Layer diagrams, real `model.summary()` outputs, live training logs (terminal-style), full hyperparameter config |
| 📈 **Results & Charts** | Training curves, confusion matrices, ROC curves, side-by-side metric comparison, full classification reports |
| 🎥 **Live Demo** | Upload a face photo (or send one from the gallery) → OpenCV face detection → real-time emotion prediction with confidence breakdown |

---

## 🧠 Models

<table>
<tr>
<td width="50%" valign="top">

### 🧩 Custom CNN
*Built from scratch*

```
Input: 48×48×1 grayscale

Conv2D(64)  → BN → ReLU  ×2 → MaxPool → Dropout
Conv2D(128) → BN → ReLU  ×2 → MaxPool → Dropout
Conv2D(256) → BN → ReLU  ×2 → MaxPool → Dropout
Flatten → Dense(256) → BN → ReLU → Dropout
Dense(7, softmax)
```

**Optimizer:** Adam (1e-3) · **Loss:** categorical cross-entropy

</td>
<td width="50%" valign="top">

### 🚀 MobileNetV2 (Transfer Learning)
*ImageNet-pretrained backbone*

```
Input: 96×96×3 (upsampled, RGB-replicated)

MobileNetV2 (frozen base)
→ GlobalAveragePooling2D
→ Dense(256) → BN → ReLU → Dropout
→ Dense(7, softmax)
```

**Phase 1:** train head only, Adam 1e-3
**Phase 2:** fine-tune top 30 layers, Adam 1e-5

</td>
</tr>
</table>

Both models are evaluated with **Precision, Recall, F1-score, Confusion Matrices, and one-vs-rest ROC/AUC curves**, then compared side by side to select the deployment candidate — exported as a **quantized `.tflite`** file for low-latency, on-device inference.

---

## 📊 Dataset

**[FER2013](https://www.kaggle.com/datasets/msambare/fer2013)** — 48×48 grayscale facial crops across 7 emotion classes: `angry`, `disgust`, `fear`, `happy`, `neutral`, `sad`, `surprise`.

- Handles known class imbalance (`disgust` is the smallest class) via computed class weights
- Supports both directory-based (`train/<class>/*.jpg`) and CSV-based (original Kaggle pixel-string) formats
- Augmentation: rotation, zoom, shift, horizontal flip, brightness jitter

---

## ⚡ Quickstart

```bash
git clone <this-repo>
cd deepfer_local

python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux

pip install -r requirements.txt
```

**1. Add your trained models** to `models/`:
`deepfer_custom_cnn_best.keras`, `deepfer_mobilenetv2_best.keras`, `deepfer_deploy_model.tflite`, `class_names.json`, `model_comparison.csv`

**2. (Optional) Add FER2013 data** to `data/train/<class>/` and `data/test/<class>/` to unlock live charts, the sample gallery, and dataset export.

**3. Run the dashboard:**

```bash
streamlit run streamlit_app.py
```

**4. Or regenerate all metrics/charts headlessly:**

```bash
python run_evaluation.py
```

---

## 📁 Project Structure

```
deepfer_local/
├── data/               train/ + test/ FER2013 images (optional)
├── models/             trained .keras / .tflite models + metadata
├── outputs/charts/     generated & original Kaggle charts
├── src/
│   ├── config.py          paths & hyperparameters
│   ├── data_pipeline.py   FER2013 loading (dir- or CSV-based)
│   ├── models.py          CNN + MobileNetV2 architectures
│   ├── evaluate.py        regenerates all charts/metrics
│   ├── app_gradio.py      lightweight Gradio demo (alt. to Streamlit)
│   ├── export_tflite.py   rebuilds the quantized .tflite
│   ├── train_cnn.py       retrain the custom CNN
│   └── train_tl.py        retrain MobileNetV2
├── streamlit_app.py    ★ the full 5-panel dashboard
├── run_evaluation.py   one-shot evaluation entry point
└── requirements.txt
```

---

## 🛠️ Tech Stack

<div align="center">

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=flat-square&logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=flat-square&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=flat-square&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=flat-square&logo=numpy&logoColor=white)

</div>

---

## 🔍 Key Findings

- ⚖️ **Class imbalance** — `disgust` is the smallest class in FER2013; class weighting helps, but it remains the hardest to classify
- 🔍 **Low native resolution (48×48)** limits fine detail; BatchNorm + augmentation improve generalization
- ⚡ **Transfer learning vs. custom CNN** — TL can converge faster, but the lightweight custom CNN wins on inference latency and, in this run, on generalization
- 📦 **Why MobileNetV2** — chosen specifically for its small footprint, enabling real-time deployment

---

## 🗺️ Roadmap

- [ ] Live webcam inference (beyond static upload)
- [ ] Model ensembling (CNN + MobileNetV2 voting)
- [ ] Mobile app powered by the exported TFLite model
- [ ] Expand to video-based temporal emotion tracking

---

## 📄 License

Released under the **MIT License** — free to use, modify, and distribute.

---

<div align="center">

### 🌟 If you found this project interesting, consider starring the repo!

**[🚀 Live Dashboard](https://deepfer-emotion-dashboard-027.streamlit.app/)**

</div>
