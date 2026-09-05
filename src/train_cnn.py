"""
DeepFER — Custom CNN training (notebook cells 12-22)

*** OPTIONAL — YOU ALREADY HAVE TRAINED MODELS. YOU DO NOT NEED TO RUN THIS. ***

Included only so the pipeline is complete/reproducible if you ever want to
retrain from scratch or fine-tune further on your own machine.
Requires data/train/<class>/*.jpg and data/test/<class>/*.jpg locally.
"""
import time
from tensorflow.keras import optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator

from . import config as C
from .models import build_custom_cnn
from .data_pipeline import compute_class_weights


def main():
    train_datagen = ImageDataGenerator(
        rescale=1. / 255,
        rotation_range=15,
        width_shift_range=0.1,
        height_shift_range=0.1,
        zoom_range=0.15,
        horizontal_flip=True,
        brightness_range=[0.85, 1.15],
        validation_split=0.1,
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

    cnn_model = build_custom_cnn()
    cnn_model.compile(optimizer=optimizers.Adam(learning_rate=1e-3),
                       loss='categorical_crossentropy', metrics=['accuracy'])

    cnn_ckpt_path = C.CNN_BEST_PATH
    cnn_callbacks = [
        callbacks.ModelCheckpoint(cnn_ckpt_path, monitor='val_accuracy', save_best_only=True, verbose=1),
        callbacks.EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True, verbose=1),
        callbacks.ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-6, verbose=1),
    ]

    start = time.time()
    history_cnn = cnn_model.fit(
        train_gen, validation_data=val_gen,
        epochs=C.EPOCHS_CNN, class_weight=class_weight_dict, callbacks=cnn_callbacks,
    )
    print(f"Custom CNN training time: {(time.time() - start) / 60:.1f} min")

    cnn_model.save(C.CNN_FINAL_PATH)
    print("Saved final model to", C.CNN_FINAL_PATH)


if __name__ == "__main__":
    main()
