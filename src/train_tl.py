"""
DeepFER — MobileNetV2 Transfer Learning training (notebook cells 25-30)

*** OPTIONAL — YOU ALREADY HAVE TRAINED MODELS. YOU DO NOT NEED TO RUN THIS. ***

Included only so the pipeline is complete/reproducible if you ever want to
retrain from scratch or fine-tune further on your own machine.
Requires data/train/<class>/*.jpg and data/test/<class>/*.jpg locally.
"""
import tensorflow as tf
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from . import config as C
from .models import build_transfer_model
from .data_pipeline import compute_class_weights


def to_tl_dataset(generator):
    def gen():
        for x_batch, y_batch in generator:
            x_batch = tf.image.grayscale_to_rgb(tf.convert_to_tensor(x_batch))
            x_batch = tf.image.resize(x_batch, (C.IMG_SIZE_TL, C.IMG_SIZE_TL))
            yield x_batch, y_batch

    return tf.data.Dataset.from_generator(
        gen,
        output_signature=(
            tf.TensorSpec(shape=(None, C.IMG_SIZE_TL, C.IMG_SIZE_TL, 3), dtype=tf.float32),
            tf.TensorSpec(shape=(None, C.NUM_CLASSES), dtype=tf.float32),
        )
    )


def main():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255, rotation_range=15, width_shift_range=0.1, height_shift_range=0.1,
        zoom_range=0.15, horizontal_flip=True, brightness_range=[0.85, 1.15], validation_split=0.1,
    )

    train_gen = train_datagen.flow_from_directory(
        C.TRAIN_DIR, target_size=(C.IMG_SIZE, C.IMG_SIZE), color_mode='grayscale',
        class_mode='categorical', batch_size=C.BATCH_SIZE, classes=C.CLASS_NAMES,
        subset='training', shuffle=True, seed=C.SEED,
    )
    val_gen = train_datagen.flow_from_directory(
        C.TRAIN_DIR, target_size=(C.IMG_SIZE, C.IMG_SIZE), color_mode='grayscale',
        class_mode='categorical', batch_size=C.BATCH_SIZE, classes=C.CLASS_NAMES,
        subset='validation', shuffle=False, seed=C.SEED,
    )
    class_weight_dict = compute_class_weights(train_gen.classes)

    train_ds_tl = to_tl_dataset(train_gen)
    val_ds_tl = to_tl_dataset(val_gen)
    steps_per_epoch_tl = len(train_gen)
    val_steps_tl = len(val_gen)

    tl_model, tl_base = build_transfer_model()
    tl_model.compile(optimizer=optimizers.Adam(1e-3), loss='categorical_crossentropy', metrics=['accuracy'])

    tl_ckpt_path = C.TL_BEST_PATH
    tl_callbacks = [
        callbacks.ModelCheckpoint(tl_ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=6, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-6, verbose=1),
    ]

    print("Phase 1 — training classification head (base frozen)...")
    tl_model.fit(
        train_ds_tl, validation_data=val_ds_tl,
        steps_per_epoch=steps_per_epoch_tl, validation_steps=val_steps_tl,
        epochs=C.EPOCHS_TL, class_weight=class_weight_dict, callbacks=tl_callbacks,
    )

    print("Phase 2 — fine-tuning top layers of MobileNetV2...")
    tl_base.trainable = True
    FINE_TUNE_AT = len(tl_base.layers) - 30
    for layer in tl_base.layers[:FINE_TUNE_AT]:
        layer.trainable = False

    tl_model.compile(optimizer=optimizers.Adam(1e-5), loss='categorical_crossentropy', metrics=['accuracy'])
    tl_model.fit(
        train_ds_tl, validation_data=val_ds_tl,
        steps_per_epoch=steps_per_epoch_tl, validation_steps=val_steps_tl,
        epochs=C.EPOCHS_TL, class_weight=class_weight_dict, callbacks=tl_callbacks,
    )

    tl_model.save(C.TL_FINAL_PATH)
    print("Saved final model to", C.TL_FINAL_PATH)


if __name__ == "__main__":
    main()
