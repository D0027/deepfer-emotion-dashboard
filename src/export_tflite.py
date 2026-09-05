"""
DeepFER — TFLite export (notebook cell 43)

OPTIONAL: you already have `deepfer_deploy_model.tflite` from Kaggle, so you
do NOT need to run this. It's included so the full pipeline is reproducible
locally if you ever retrain or want to re-export.

Note: the original notebook called a `convert_to_tflite(...)` helper that
was never actually defined in the notebook itself (it would have failed on
Kaggle too if reached). The standard dynamic-range-quantized conversion
below is the standard equivalent of what that call was doing.
"""
import os
from tensorflow.keras import mixed_precision
from tensorflow import keras
import tensorflow as tf

from . import config as C
from .models import build_custom_cnn


def convert_to_tflite(model, out_path, quantize=True):
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    if quantize:
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    with open(out_path, "wb") as f:
        f.write(tflite_model)
    print(f"Saved TFLite model to {out_path} ({len(tflite_model)/1024:.1f} KB)")


def main():
    best_path = C.CNN_BEST_PATH if os.path.exists(C.CNN_BEST_PATH) else C.CNN_FINAL_PATH
    if not os.path.exists(best_path):
        raise FileNotFoundError(f"No custom CNN model found at {best_path}")

    best_model = keras.models.load_model(best_path)

    mixed_precision.set_global_policy('float32')
    export_model = build_custom_cnn()
    export_model.set_weights(best_model.get_weights())

    out_path = os.path.join(C.OUTPUT_DIR, "deepfer_deploy_model.tflite")
    convert_to_tflite(export_model, out_path, quantize=True)
    mixed_precision.set_global_policy('mixed_float16')


if __name__ == "__main__":
    main()
