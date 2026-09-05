"""
DeepFER — Local Project Configuration
Mirrors the config from the original Kaggle notebook, adapted for local paths.
"""
import os

# ---- Global Config (same as Kaggle notebook) ----
IMG_SIZE      = 48          # native FER2013 resolution (grayscale 48x48)
IMG_SIZE_TL   = 96          # upsampled resolution for transfer-learning backbone
BATCH_SIZE    = 64
EPOCHS_CNN    = 40
EPOCHS_TL     = 25
NUM_CLASSES   = 7
CLASS_NAMES   = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']
SEED          = 42

# ---- Local paths (replaces /kaggle/input and /kaggle/working) ----
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

DATA_DIR   = os.path.join(PROJECT_ROOT, "data")
TRAIN_DIR  = os.path.join(DATA_DIR, "train")   # expects train/<class>/*.jpg
TEST_DIR   = os.path.join(DATA_DIR, "test")    # expects test/<class>/*.jpg

MODELS_DIR  = os.path.join(PROJECT_ROOT, "models")     # put your downloaded .keras/.tflite/json/csv here
OUTPUT_DIR  = os.path.join(PROJECT_ROOT, "outputs")
CHARTS_DIR  = os.path.join(OUTPUT_DIR, "charts")

CNN_BEST_PATH   = os.path.join(MODELS_DIR, "deepfer_custom_cnn_best.keras")
CNN_FINAL_PATH  = os.path.join(MODELS_DIR, "deepfer_custom_cnn_final.keras")
TL_BEST_PATH    = os.path.join(MODELS_DIR, "deepfer_mobilenetv2_best.keras")
TL_FINAL_PATH   = os.path.join(MODELS_DIR, "deepfer_mobilenetv2_final.keras")
TFLITE_PATH     = os.path.join(MODELS_DIR, "deepfer_deploy_model.tflite")
CLASS_NAMES_JSON = os.path.join(MODELS_DIR, "class_names.json")
COMPARISON_CSV   = os.path.join(MODELS_DIR, "model_comparison.csv")

os.makedirs(OUTPUT_DIR, exist_ok=True)
os.makedirs(CHARTS_DIR, exist_ok=True)
