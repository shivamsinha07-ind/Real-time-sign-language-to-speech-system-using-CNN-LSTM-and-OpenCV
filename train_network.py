import cv2
import numpy as np
import os
import tensorflow as tf

layers = tf.keras.layers
models = tf.keras.models

from settings import *

def build():
    model = models.Sequential([
        layers.Input(shape=(FRAMES_PER_SEQ, FRAME_SIZE[0], FRAME_SIZE[1], 1)),

        # 🔥 Very small CNN
        layers.TimeDistributed(layers.Conv2D(16, (3,3), activation='relu')),
        layers.TimeDistributed(layers.MaxPooling2D(2,2)),

        layers.TimeDistributed(layers.Conv2D(32, (3,3), activation='relu')),
        layers.TimeDistributed(layers.MaxPooling2D(2,2)),

        layers.TimeDistributed(layers.Flatten()),

        # 🔥 Tiny LSTM
        layers.LSTM(32),

        layers.Dense(32, activation='relu'),
        layers.Dense(len(CLASSES), activation='softmax')
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )

    return model


def load():
    X, y = [], []

    for i, cls in enumerate(CLASSES):
        path = os.path.join(ROOT_DATA, cls)

        if not os.path.exists(path):
            continue

        for seq in os.listdir(path):
            frames = []

            for f in range(FRAMES_PER_SEQ):
                img_path = os.path.join(path, seq, f"{f}.jpg")

                img = cv2.imread(img_path, 0)

                if img is None:
                    img = np.zeros((FRAME_SIZE[0], FRAME_SIZE[1]), dtype=np.uint8)
                else:
                    img = cv2.resize(img, (FRAME_SIZE[1], FRAME_SIZE[0]))

                img = img.astype(np.float32) / 255.0
                frames.append(np.expand_dims(img, -1))

            X.append(frames)
            y.append(i)

    X = np.array(X)
    y = tf.keras.utils.to_categorical(y, len(CLASSES))

    print("Dataset loaded:", X.shape)

    return X, y


# -------- MAIN --------
X, y = load()

model = build()

model.fit(
    X, y,
    epochs=3,              # 🔥 VERY FAST
    batch_size=8,
    validation_split=0.2
)

model.save("gesture_model_v2.keras")

print("✅ Model saved")