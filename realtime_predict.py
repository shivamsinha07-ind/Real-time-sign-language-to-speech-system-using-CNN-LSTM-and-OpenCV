import cv2
import numpy as np
import tensorflow as tf
import pyttsx3

from settings import *

# ---------- LOAD MODEL ----------
model = tf.keras.models.load_model("gesture_model_v2.keras", compile=False, safe_mode=False)

# ---------- SPEECH ENGINE ----------
engine = pyttsx3.init()

# ---------- CAMERA ----------
cap = cv2.VideoCapture(0)

sequence = []

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Flip for natural mirror
    frame = cv2.flip(frame, 1)

    # Convert to grayscale
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Resize
    img = cv2.resize(gray, (FRAME_SIZE[1], FRAME_SIZE[0]))

    # Normalize
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, -1)

    sequence.append(img)

    # Keep last N frames
    if len(sequence) > FRAMES_PER_SEQ:
        sequence.pop(0)

    # Prediction
    if len(sequence) == FRAMES_PER_SEQ:
        input_data = np.expand_dims(sequence, axis=0)

        preds = model.predict(input_data, verbose=0)[0]
        pred = np.argmax(preds)
        confidence = preds[pred]

        word = CLASSES[pred]

        # Show on screen
        cv2.putText(frame, f"{word} ({confidence:.2f})",
                    (10, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1,
                    (0, 255, 0),
                    2)

        # 🔊 SPEAK (only if confident)
        if confidence > 0.8:
            engine.say(word)
            engine.runAndWait()

    # Show camera
    cv2.imshow("Sign Detection", frame)

    # Press Q to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()