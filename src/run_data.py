"""
DeepFER — Real training-run data captured from the Kaggle notebook execution.
Used by streamlit_app.py to render terminal logs, model summaries, and
classification reports with the ACTUAL numbers from your run (not placeholders).
"""

# ---------------- Custom CNN ----------------
CNN_SUMMARY_TEXT = '''Model: "DeepFER_CustomCNN"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer (InputLayer)        │ (None, 48, 48, 1)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d (Conv2D)                 │ (None, 48, 48, 64)     │           640 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization             │ (None, 48, 48, 64)     │           256 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation (Activation)         │ (None, 48, 48, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_1 (Conv2D)               │ (None, 48, 48, 64)     │        36,928 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_1           │ (None, 48, 48, 64)     │           256 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation_1 (Activation)       │ (None, 48, 48, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d (MaxPooling2D)    │ (None, 24, 24, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout (Dropout)               │ (None, 24, 24, 64)     │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_2 (Conv2D)               │ (None, 24, 24, 128)    │        73,856 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_2           │ (None, 24, 24, 128)    │           512 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation_2 (Activation)       │ (None, 24, 24, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_3 (Conv2D)               │ (None, 24, 24, 128)    │       147,584 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_3           │ (None, 24, 24, 128)    │           512 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation_3 (Activation)       │ (None, 24, 24, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_1 (MaxPooling2D)  │ (None, 12, 12, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_1 (Dropout)             │ (None, 12, 12, 128)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_4 (Conv2D)               │ (None, 12, 12, 256)    │       295,168 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_4           │ (None, 12, 12, 256)    │         1,024 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation_4 (Activation)       │ (None, 12, 12, 256)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ conv2d_5 (Conv2D)               │ (None, 12, 12, 256)    │       590,080 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_5           │ (None, 12, 12, 256)    │         1,024 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation_5 (Activation)       │ (None, 12, 12, 256)    │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ max_pooling2d_2 (MaxPooling2D)  │ (None, 6, 6, 256)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_2 (Dropout)             │ (None, 6, 6, 256)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d        │ (None, 256)            │             0 │
│ (GlobalAveragePooling2D)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense (Dense)                   │ (None, 256)            │        65,792 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_6           │ (None, 256)            │         1,024 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation_6 (Activation)       │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_3 (Dropout)             │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_1 (Dense)                 │ (None, 7)              │         1,799 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 1,216,455 (4.64 MB)
 Trainable params: 1,214,151 (4.63 MB)
 Non-trainable params: 2,304 (9.00 KB)'''

CNN_PARAMS = {"total": "1,216,455", "total_mb": "4.64 MB",
              "trainable": "1,214,151", "trainable_mb": "4.63 MB",
              "non_trainable": "2,304", "non_trainable_mb": "9.00 KB"}

CNN_TRAINING_LOG = '''Epoch 1/40
449/449 ━━━━━━━━━━━━━━━━━━━━ 362s 754ms/step - accuracy: 0.1602 - loss: 2.1300 - val_accuracy: 0.0914 - val_loss: 1.9537 - learning_rate: 0.0010
Epoch 2/40
449/449 ━━━━━━━━━━━━━━━━━━━━ 72s 159ms/step - accuracy: 0.1763 - loss: 1.9877 - val_accuracy: 0.1513 - val_loss: 1.9131 - learning_rate: 0.0010
Epoch 3/40
449/449 ━━━━━━━━━━━━━━━━━━━━ 61s 135ms/step - accuracy: 0.1985 - loss: 1.8866 - val_accuracy: 0.2754 - val_loss: 1.8544 - learning_rate: 0.0010
Epoch 4/40
449/449 ━━━━━━━━━━━━━━━━━━━━ 65s 144ms/step - accuracy: 0.2594 - loss: 1.7851 - val_accuracy: 0.3334 - val_loss: 1.6912 - learning_rate: 0.0010
Epoch 5/40
449/449 ━━━━━━━━━━━━━━━━━━━━ 61s 136ms/step - accuracy: 0.3204 - loss: 1.6882 - val_accuracy: 0.3337 - val_loss: 1.8438 - learning_rate: 0.0010
Epoch 7/40
449/449 ━━━━━━━━━━━━━━━━━━━━ 81s 181ms/step - accuracy: 0.4118 - loss: 1.5080 - val_accuracy: 0.4026 - val_loss: 1.6033 - learning_rate: 0.0010
Epoch 10/40
449/449 ━━━━━━━━━━━━━━━━━━━━ 60s 132ms/step - accuracy: 0.4723 - loss: 1.3599 - val_accuracy: 0.4763 - val_loss: 1.3642 - learning_rate: 0.0010
Epoch 13/40
449/449 ━━━━━━━━━━━━━━━━━━━━ 59s 132ms/step - accuracy: 0.5125 - loss: 1.2590 - val_accuracy: 0.5258 - val_loss: 1.2594 - learning_rate: 0.0010
Epoch 16/40
449/449 ━━━━━━━━━━━━━━━━━━━━ 65s 146ms/step - accuracy: 0.5311 - loss: 1.2099 - val_accuracy: 0.5568 - val_loss: 1.1848 - learning_rate: 0.0010
Epoch 17/40
449/449 ━━━━━━━━━━━━━━━━━━━━ 66s 146ms/step - accuracy: 0.5408 - loss: 1.1807 - val_accuracy: 0.5666 - val_loss: 1.1378 - learning_rate: 0.0010
Epoch 21: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
Epoch 22/40
449/449 ━━━━━━━━━━━━━━━━━━━━ 55s 124ms/step - accuracy: 0.5750 - loss: 1.0753 - val_accuracy: 0.6056 - val_loss: 1.0348 - learning_rate: 5.0000e-04
Epoch 26: ReduceLROnPlateau reducing learning rate to 0.0002500000118743628.
Epoch 27/40
449/449 ━━━━━━━━━━━━━━━━━━━━ 57s 127ms/step - accuracy: 0.6024 - loss: 0.9879 - val_accuracy: 0.6273 - val_loss: 0.9758 - learning_rate: 2.5000e-04
Epoch 31: ReduceLROnPlateau reducing learning rate to 0.0001250000059371814.
Epoch 32/40
449/449 ━━━━━━━━━━━━━━━━━━━━ 53s 118ms/step - accuracy: 0.6174 - loss: 0.9543 - val_accuracy: 0.6351 - val_loss: 0.9780 - learning_rate: 1.2500e-04
Epoch 35/40
449/449 ━━━━━━━━━━━━━━━━━━━━ 53s 117ms/step - accuracy: 0.6201 - loss: 0.9407 - val_accuracy: 0.6390 - val_loss: 0.9669 - learning_rate: 1.2500e-04
Epoch 40/40
449/449 ━━━━━━━━━━━━━━━━━━━━ 71s 157ms/step - accuracy: 0.6311 - loss: 0.9182 - val_accuracy: 0.6400 - val_loss: 0.9643 - learning_rate: 1.2500e-04
Restoring model weights from the end of the best epoch: 39.
Custom CNN training time: 44.0 min'''

# ---------------- MobileNetV2 Transfer Learning ----------------
TL_SUMMARY_TEXT = '''Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/mobilenet_v2/mobilenet_v2_weights_tf_dim_ordering_tf_kernels_1.0_96_no_top.h5
9406464/9406464 ━━━━━━━━━━━━━━━━━━━━ 0s 0us/step
Model: "DeepFER_MobileNetV2_TL"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ input_layer_2 (InputLayer)      │ (None, 96, 96, 3)      │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ mobilenetv2_1.00_96             │ (None, 3, 3, 1280)     │     2,257,984 │
│ (Functional)                    │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ global_average_pooling2d_1      │ (None, 1280)           │             0 │
│ (GlobalAveragePooling2D)        │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_2 (Dense)                 │ (None, 256)            │       327,936 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ batch_normalization_7           │ (None, 256)            │         1,024 │
│ (BatchNormalization)            │                        │               │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ activation_7 (Activation)       │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dropout_4 (Dropout)             │ (None, 256)            │             0 │
├─────────────────────────────────┼────────────────────────┼───────────────┤
│ dense_3 (Dense)                 │ (None, 7)              │         1,799 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 2,588,743 (9.88 MB)
 Trainable params: 330,247 (1.26 MB)
 Non-trainable params: 2,258,496 (8.62 MB)'''

TL_PARAMS = {"total": "2,588,743", "total_mb": "9.88 MB",
             "trainable": "330,247", "trainable_mb": "1.26 MB",
             "non_trainable": "2,258,496", "non_trainable_mb": "8.62 MB"}

TL_TRAINING_LOG = '''Epoch 1/10
449/449 ━━━━━━━━━━━━━━━━━━━━ 158s 274ms/step - accuracy: 0.3120 - loss: 1.8186 - val_accuracy: 0.4035 - val_loss: 1.5794 - learning_rate: 0.0010
Epoch 2/10
449/449 ━━━━━━━━━━━━━━━━━━━━ 54s 120ms/step - accuracy: 0.3677 - loss: 1.6209 - val_accuracy: 0.3846 - val_loss: 1.5857 - learning_rate: 0.0010
Epoch 3/10
449/449 ━━━━━━━━━━━━━━━━━━━━ 56s 125ms/step - accuracy: 0.3816 - loss: 1.5661 - val_accuracy: 0.3722 - val_loss: 1.6364 - learning_rate: 0.0010
Epoch 4/10
449/449 ━━━━━━━━━━━━━━━━━━━━ 56s 125ms/step - accuracy: 0.3937 - loss: 1.5378 - val_accuracy: 0.4401 - val_loss: 1.4913 - learning_rate: 0.0010
Epoch 5/10
449/449 ━━━━━━━━━━━━━━━━━━━━ 53s 118ms/step - accuracy: 0.4023 - loss: 1.5225 - val_accuracy: 0.4394 - val_loss: 1.4612 - learning_rate: 0.0010
Epoch 6/10
449/449 ━━━━━━━━━━━━━━━━━━━━ 51s 113ms/step - accuracy: 0.4061 - loss: 1.5047 - val_accuracy: 0.4152 - val_loss: 1.5267 - learning_rate: 0.0010
Epoch 7/10
449/449 ━━━━━━━━━━━━━━━━━━━━ 52s 117ms/step - accuracy: 0.4098 - loss: 1.4768 - val_accuracy: 0.4436 - val_loss: 1.4447 - learning_rate: 0.0010
Epoch 8/10
449/449 ━━━━━━━━━━━━━━━━━━━━ 48s 107ms/step - accuracy: 0.4174 - loss: 1.4690 - val_accuracy: 0.4401 - val_loss: 1.4683 - learning_rate: 0.0010
Epoch 9/10
449/449 ━━━━━━━━━━━━━━━━━━━━ 50s 111ms/step - accuracy: 0.4220 - loss: 1.4607 - val_accuracy: 0.4401 - val_loss: 1.4570 - learning_rate: 0.0010
Epoch 10: ReduceLROnPlateau reducing learning rate to 0.0005000000237487257.
Epoch 10/10
449/449 ━━━━━━━━━━━━━━━━━━━━ 52s 117ms/step - accuracy: 0.4211 - loss: 1.4466 - val_accuracy: 0.4253 - val_loss: 1.5096 - learning_rate: 0.0010
Restoring model weights from the end of the best epoch: 7.'''

TL_TRAINING_LOG_PHASE2 = '''Epoch 1/25
449/449 ━━━━━━━━━━━━━━━━━━━━ 141s 240ms/step - accuracy: 0.3205 - loss: 1.7699 - val_accuracy: 0.3154 - val_loss: 2.0710 - learning_rate: 1.0000e-05
Epoch 2/25
449/449 ━━━━━━━━━━━━━━━━━━━━ 56s 125ms/step - accuracy: 0.3556 - loss: 1.6395 - val_accuracy: 0.3667 - val_loss: 1.7892 - learning_rate: 1.0000e-05
Epoch 3: ReduceLROnPlateau reducing learning rate to 4.999999873689376e-06.
Epoch 3/25
449/449 ━━━━━━━━━━━━━━━━━━━━ 54s 121ms/step - accuracy: 0.3677 - loss: 1.6003 - val_accuracy: 0.3991 - val_loss: 1.6319 - learning_rate: 1.0000e-05
Epoch 4/25
449/449 ━━━━━━━━━━━━━━━━━━━━ 52s 116ms/step - accuracy: 0.3830 - loss: 1.5532 - val_accuracy: 0.4125 - val_loss: 1.5608 - learning_rate: 5.0000e-06
Epoch 5/25
449/449 ━━━━━━━━━━━━━━━━━━━━ 51s 114ms/step - accuracy: 0.3883 - loss: 1.5250 - val_accuracy: 0.4199 - val_loss: 1.5286 - learning_rate: 5.0000e-06
Epoch 6: ReduceLROnPlateau reducing learning rate to 2.499999936844688e-06.
Epoch 6/25
449/449 ━━━━━━━━━━━━━━━━━━━━ 54s 121ms/step - accuracy: 0.3950 - loss: 1.4962 - val_accuracy: 0.4189 - val_loss: 1.5233 - learning_rate: 5.0000e-06
Epoch 6: early stopping
Restoring model weights from the end of the best epoch: 1.'''

# ---------------- Classification reports (actual test-set numbers) ----------------
CNN_REPORT_ROWS = [
    ("angry", 0.55, 0.59, 0.57, 958),
    ("disgust", 0.54, 0.66, 0.60, 111),
    ("fear", 0.49, 0.37, 0.42, 1024),
    ("happy", 0.89, 0.82, 0.86, 1774),
    ("neutral", 0.52, 0.75, 0.61, 1233),
    ("sad", 0.59, 0.39, 0.47, 1247),
    ("surprise", 0.69, 0.85, 0.76, 831),
]
CNN_OVERALL = {"accuracy": 0.6378, "precision": 0.6438, "recall": 0.6378, "f1": 0.6309, "support": 7178}
CNN_MACRO_AVG = (0.61, 0.63, 0.61)
CNN_WEIGHTED_AVG = (0.64, 0.64, 0.63)

TL_REPORT_ROWS = [
    ("angry", 0.40, 0.08, 0.13, 958),
    ("disgust", 0.03, 0.89, 0.07, 111),
    ("fear", 0.39, 0.10, 0.16, 1024),
    ("happy", 0.56, 0.62, 0.59, 1774),
    ("neutral", 0.39, 0.32, 0.35, 1233),
    ("sad", 0.60, 0.02, 0.04, 1247),
    ("surprise", 0.59, 0.55, 0.57, 831),
]
TL_OVERALL = {"accuracy": 0.3154, "precision": 0.4866, "recall": 0.3154, "f1": 0.3201, "support": 7178}
TL_MACRO_AVG = (0.42, 0.37, 0.27)
TL_WEIGHTED_AVG = (0.49, 0.32, 0.32)