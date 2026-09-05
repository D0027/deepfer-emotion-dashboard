<div align="center">

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=220&section=header&text=DeepFER&fontSize=70&fontColor=ffffff&desc=Facial%20Emotion%20Recognition%20%E2%80%A2%20CNN%20vs%20MobileNetV2&descAlignY=58&descSize=20&animation=fadeIn" width="100%"/>

<br>

<a href="https://deepfer-emotion-dashboard-027.streamlit.app/">
  <img src="https://img.shields.io/badge/🚀_LAUNCH_LIVE_DASHBOARD-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white&labelColor=0e1117" height="45"/>
</a>

<br><br>

![Python](https://img.shields.io/badge/Python-3.10+-3776AB?style=flat-square&logo=python&logoColor=white)
![TensorFlow](https://img.shields.io/badge/TensorFlow-Keras-FF6F00?style=flat-square&logo=tensorflow&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-Dashboard-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-FaceDetect-5C3EE8?style=flat-square&logo=opencv&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-8A2BE2?style=flat-square)
![Status](https://img.shields.io/badge/Status-Live-4ade80?style=flat-square)

<br>

<img src="https://readme-typing-svg.demolab.com?font=Fira+Code&pause=1000&color=7C3AED&center=true&vCenter=true&width=650&lines=Classifying+7+human+emotions+in+real+time...;Custom+CNN+%F0%9F%86%9A+MobileNetV2+Transfer+Learning;Trained+on+FER2013+%E2%80%A2+Deployed+with+Streamlit;Quantized+to+TFLite+for+edge+inference" alt="Typing SVG"/>

</div>

<br>

<p align="center">
  <a href="#-live-preview">Live Preview</a> •
  <a href="#-dashboard-tour">Dashboard</a> •
  <a href="#-models">Models</a> •
  <a href="#-dataset">Dataset</a> •
  <a href="#-quickstart">Quickstart</a> •
  <a href="#-results-snapshot">Results</a> •
  <a href="#-tech-stack">Tech Stack</a>
</p>

<div align="center">

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

</div>

## 🌐 Live Preview

<div align="center">

### 👉 **[deepfer-emotion-dashboard-027.streamlit.app](https://deepfer-emotion-dashboard-027.streamlit.app/)** 👈

<table>
<tr>
<td align="center">🧭<br><b>Overview</b><br><sub>Pipeline & KPIs</sub></td>
<td align="center">📊<br><b>Dataset</b><br><sub>Gallery & Stats</sub></td>
<td align="center">🏗️<br><b>Architecture</b><br><sub>Layers & Logs</sub></td>
<td align="center">📈<br><b>Results</b><br><sub>Metrics & ROC</sub></td>
<td align="center">🎥<br><b>Live Demo</b><br><sub>Upload & Predict</sub></td>
</tr>
</table>

</div>

<br>

<div align="center">

## 🎯 At a Glance

<table>
<tr>
<th>🗂️ Classes</th>
<th>🧠 Models</th>
<th>🖼️ Input</th>
<th>📦 Deployment</th>
<th>⚡ Inference</th>
</tr>
<tr align="center">
<td><code>7 emotions</code></td>
<td><code>CNN + MobileNetV2</code></td>
<td><code>48×48 grayscale</code></td>
<td><code>Quantized TFLite</code></td>
<td><code>Real-time</code></td>
</tr>
</table>

`angry` `disgust` `fear` `happy` `neutral` `sad` `surprise`

</div>

<div align="center">

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

</div>

## ✨ Overview

**DeepFER** detects and classifies human facial emotions using two competing deep learning pipelines trained on **FER2013** — a **Custom CNN built from scratch** and a **MobileNetV2 transfer-learning model**. Both are rigorously benchmarked (accuracy, precision, recall, F1, confusion matrices, ROC/AUC) and the strongest performer is exported as a lightweight, **quantized TFLite** model for real-world, low-latency deployment.

Everything lives inside a polished, **5-panel interactive Streamlit dashboard** — no digging through notebooks required. Explore the methodology, browse the dataset, inspect model internals down to the layer, review every metric, and run **live predictions on your own photos**.

<br>

## 🖥️ Dashboard Tour

<details open>
<summary><b>🧭 Panel 1 — Overview & Methodology</b></summary>
<br>

> Animated KPI cards, the complete 10-step ML pipeline (data → preprocessing → training → evaluation → export), and a curated list of key findings & challenges.

</details>

<details>
<summary><b>📊 Panel 2 — Dataset Review</b></summary>
<br>

> Class distribution charts, live folder counts, a 10-image-per-class gallery for both train/test splits, one-click **"send to Live Demo"** on any sample, and full dataset zip export.

</details>

<details>
<summary><b>🏗️ Panel 3 — Model Architecture</b></summary>
<br>

> Side-by-side architecture diagrams, real `model.summary()` layer dumps, **terminal-styled live training logs** with syntax highlighting, and the full hyperparameter configuration table.

</details>

<details>
<summary><b>📈 Panel 4 — Results & Charts</b></summary>
<br>

> Training curves, confusion matrices, one-vs-rest ROC curves, animated metric comparison bars, and full precision/recall/F1 classification reports per class.

</details>

<details>
<summary><b>🎥 Panel 5 — Live Demo</b></summary>
<br>

> Upload any face photo (or send one straight from the Dataset gallery) → OpenCV Haar-cascade face detection → real-time Custom CNN prediction with a full confidence breakdown chart.

</details>

<div align="center">

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

</div>

## 🧠 Models

<table width="100%">
<tr>
<th width="50%">🧩 Custom CNN <sub>(from scratch)</sub></th>
<th width="50%">🚀 MobileNetV2 <sub>(transfer learning)</sub></th>
</tr>
<tr valign="top">
<td>

```text
Input: 48×48×1 grayscale
──────────────────────────
Conv2D(64)  → BN → ReLU  ×2
   → MaxPool → Dropout(0.25)
Conv2D(128) → BN → ReLU  ×2
   → MaxPool → Dropout(0.25)
Conv2D(256) → BN → ReLU  ×2
   → MaxPool → Dropout(0.30)
Flatten → Dense(256) → BN
   → ReLU → Dropout(0.50)
Dense(7, softmax)
```

`Optimizer:` Adam (1e-3)
`Loss:` categorical cross-entropy

</td>
<td>

```text
Input: 96×96×3 (upsampled, RGB)
──────────────────────────
MobileNetV2 (ImageNet, frozen)
   → GlobalAveragePooling2D
   → Dense(256) → BN → ReLU
   → Dropout(0.40)
Dense(7, softmax)
```

`Phase 1:` head only, Adam 1e-3
`Phase 2:` fine-tune top 30 layers, Adam 1e-5

</td>
</tr>
</table>

<div align="center">

| 🏆 Evaluated on | Metric suite |
|:---:|:---|
| Held-out FER2013 test set | Accuracy · Precision · Recall · F1 · Confusion Matrix · ROC/AUC |

</div>

<div align="center">

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

</div>

## 📊 Dataset

<table>
<tr>
<td width="70%">

**[FER2013](https://www.kaggle.com/datasets/msambare/fer2013)** — 48×48 grayscale facial crops across 7 emotion classes.

- ⚖️ Handles class imbalance (`disgust` is smallest) via computed class weights
- 🔀 Supports directory-based (`train/<class>/*.jpg`) **and** CSV pixel-string formats
- 🎛️ Augmentation: rotation · zoom · shift · h-flip · brightness jitter

</td>
<td width="30%" align="center">

**Classes**
😠 angry
🤢 disgust
😨 fear
😄 happy
😐 neutral
😢 sad
😲 surprise

</td>
</tr>
</table>

<div align="center">

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

</div>

## 📈 Results Snapshot

> Full metrics, ROC curves, confusion matrices, and per-class reports live inside the **Results & Charts** panel of the dashboard — this is just the headline.

<div align="center">

| Model | Strengths | Watch-outs |
|---|---|---|
| 🧩 **Custom CNN** | Best overall generalization across all 7 classes on this run · lower inference latency | Trained from scratch → needs more epochs to converge |
| 🚀 **MobileNetV2 TL** | Fast head-only convergence · ImageNet features | Fine-tuning phase showed signs of catastrophic forgetting on low-res 48×48 faces |

</div>

<div align="center">

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

</div>

## ⚡ Quickstart

```bash
# 1. Clone
git clone <this-repo>
cd deepfer_local

# 2. Set up environment
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # macOS/Linux
pip install -r requirements.txt
```

```text
3. Drop your trained models into models/
   ├── deepfer_custom_cnn_best.keras
   ├── deepfer_mobilenetv2_best.keras
   ├── deepfer_deploy_model.tflite
   ├── class_names.json
   └── model_comparison.csv
```

```bash
# 4. (Optional) add FER2013 to data/train/<class>/ and data/test/<class>/
#    to unlock live charts, sample gallery & dataset export

# 5. Launch the dashboard
streamlit run streamlit_app.py

# — or regenerate all metrics/charts headlessly —
python run_evaluation.py
```

<div align="center">

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

</div>

## 📁 Project Structure

```text
deepfer_local/
├── 📂 data/               train/ + test/ FER2013 images (optional)
├── 📂 models/             trained .keras / .tflite models + metadata
├── 📂 outputs/charts/     generated & original Kaggle charts
├── 📂 src/
│   ├── config.py             paths & hyperparameters
│   ├── data_pipeline.py      FER2013 loading (dir- or CSV-based)
│   ├── models.py             CNN + MobileNetV2 architectures
│   ├── evaluate.py           regenerates all charts/metrics
│   ├── app_gradio.py         lightweight Gradio demo (alt. UI)
│   ├── export_tflite.py      rebuilds the quantized .tflite
│   ├── train_cnn.py          retrain the custom CNN
│   └── train_tl.py           retrain MobileNetV2
├── 🎛️ streamlit_app.py    ★ the full 5-panel dashboard
├── 🚀 run_evaluation.py   one-shot evaluation entry point
└── 📋 requirements.txt
```

<div align="center">

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

</div>

## 🛠️ Tech Stack

<div align="center">

![TensorFlow](https://img.shields.io/badge/TensorFlow-FF6F00?style=for-the-badge&logo=tensorflow&logoColor=white)
![Keras](https://img.shields.io/badge/Keras-D00000?style=for-the-badge&logo=keras&logoColor=white)
![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=streamlit&logoColor=white)
![OpenCV](https://img.shields.io/badge/OpenCV-5C3EE8?style=for-the-badge&logo=opencv&logoColor=white)
![Plotly](https://img.shields.io/badge/Plotly-3F4F75?style=for-the-badge&logo=plotly&logoColor=white)
![Pandas](https://img.shields.io/badge/Pandas-150458?style=for-the-badge&logo=pandas&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?style=for-the-badge&logo=scikitlearn&logoColor=white)

</div>

<div align="center">

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

</div>

## 🔍 Key Findings

<table>
<tr><td>⚖️</td><td><b>Class imbalance</b> — <code>disgust</code> is the smallest class in FER2013; class weighting helps but it remains the hardest to classify</td></tr>
<tr><td>🔍</td><td><b>Low native resolution (48×48)</b> limits fine detail — BatchNorm + augmentation improve generalization</td></tr>
<tr><td>⚡</td><td><b>Transfer learning vs. custom CNN</b> — TL converges faster, but the lightweight custom CNN wins on latency and, in this run, generalization</td></tr>
<tr><td>📦</td><td><b>Why MobileNetV2</b> — chosen specifically for its small footprint, enabling real-time deployment</td></tr>
</table>

<div align="center">

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

</div>

## 🗺️ Roadmap

- [ ] 🎥 Live webcam inference (beyond static upload)
- [ ] 🤝 Model ensembling (CNN + MobileNetV2 voting)
- [ ] 📱 Mobile app powered by the exported TFLite model
- [ ] ⏱️ Video-based temporal emotion tracking

<div align="center">

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

</div>

## 📄 License

Released under the **MIT License** — free to use, modify, and distribute.

<br>

<div align="center">

### 🌟 If this project sparked joy, drop it a star!

<a href="https://deepfer-emotion-dashboard-027.streamlit.app/">
  <img src="https://img.shields.io/badge/🚀_OPEN_THE_LIVE_APP-7C3AED?style=for-the-badge&logoColor=white" height="45"/>
</a>

<br><br>

<img src="https://capsule-render.vercel.app/api?type=waving&color=gradient&customColorList=6,11,20&height=100&section=footer" width="100%"/>

</div>
