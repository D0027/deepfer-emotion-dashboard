"""
DeepFER — Data pipeline (local version of notebook cells 5-15)
Handles both directory-structured (train/<class>/*.jpg) and CSV-structured
FER2013 datasets, exactly like the Kaggle notebook.
"""
import os
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.utils.class_weight import compute_class_weight

from . import config as C


def detect_pipeline():
    """Returns True if using CSV pipeline, False if using directory pipeline."""
    return not os.path.isdir(C.TRAIN_DIR)


def csv_to_arrays(df_subset):
    pixels = df_subset['pixels'].apply(lambda x: np.array(x.split(), dtype='float32').reshape(48, 48, 1))
    X = np.stack(pixels.values)
    y = df_subset['emotion'].values
    return X, y


def load_csv_dataset(csv_path):
    df = pd.read_csv(csv_path)
    train_df = df[df['Usage'] == 'Training']
    val_df = df[df['Usage'] == 'PublicTest']
    test_df = df[df['Usage'] == 'PrivateTest']
    X_train, y_train = csv_to_arrays(train_df)
    X_val, y_val = csv_to_arrays(val_df)
    X_test, y_test = csv_to_arrays(test_df)
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)


def get_test_generator_cnn():
    """48x48 grayscale test generator for the custom CNN (directory pipeline)."""
    test_datagen = ImageDataGenerator(rescale=1. / 255)
    return test_datagen.flow_from_directory(
        C.TEST_DIR, target_size=(C.IMG_SIZE, C.IMG_SIZE), color_mode='grayscale',
        class_mode='categorical', batch_size=C.BATCH_SIZE, classes=C.CLASS_NAMES, shuffle=False
    )


def get_test_generator_tl():
    """96x96 RGB-equivalent test set for MobileNetV2, built from the same
    grayscale generator and converted on the fly (matches notebook cell 33)."""
    gen = get_test_generator_cnn()

    def tl_eval_dataset(generator):
        def gen_fn():
            generator.reset()
            for x_batch, y_batch in generator:
                x_batch = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_batch))
                x_batch = tf.image.resize(x_batch, (C.IMG_SIZE_TL, C.IMG_SIZE_TL))
                yield x_batch, y_batch

        return tf.data.Dataset.from_generator(
            gen_fn,
            output_signature=(
                tf.TensorSpec(shape=(None, C.IMG_SIZE_TL, C.IMG_SIZE_TL, 3), dtype=tf.float32),
                tf.TensorSpec(shape=(None, C.NUM_CLASSES), dtype=tf.float32),
            )
        )

    return gen, tl_eval_dataset(gen)


def compute_class_weights(y_train_labels):
    class_weights_arr = compute_class_weight(
        class_weight='balanced', classes=np.unique(y_train_labels), y=y_train_labels
    )
    return {i: w for i, w in enumerate(class_weights_arr)}
