import tensorflow as tf
import numpy as np
import cv2
import os
from tensorflow.keras.applications.efficientnet import preprocess_input


# =========================
# CONFIG
# =========================
MODEL_PATH = r"new_efficientnet_model_fixed.h5"
IMAGE_PATH = r"Image For Testing/tb4.jpg"
IMG_SIZE = (224, 224)
CLASS_NAMES = ["Normal", "Tuberculosis"]

# =========================
# LOAD MODEL
# =========================
model = tf.keras.models.load_model(MODEL_PATH)
print("✅ Model loaded successfully")

# =========================
# LOAD & PREPROCESS IMAGE
# =========================
if not os.path.exists(IMAGE_PATH):
    raise FileNotFoundError(f"Image not found: {IMAGE_PATH}")

# Read image (BGR)
image = cv2.imread(IMAGE_PATH)

# Convert BGR → RGB
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Resize
image = cv2.resize(image, IMG_SIZE)

# Normalize (0–1)
image = preprocess_input(image.astype("float32"))

# Add batch dimension
image = np.expand_dims(image, axis=0)

# =========================
# PREDICTION
# =========================
pred = model.predict(image)[0][0]  

if pred >= 0.5:
    predicted_class = "Tuberculosis"
    confidence = pred
else:
    predicted_class = "Normal"
    confidence = 1 - pred

# =========================
# RESULT
# =========================
print("\n🔍 Prediction Result")
print("-------------------")
print(f"Image Path     : {IMAGE_PATH}")
print(f"Predicted Class: {predicted_class}")
print(f"Confidence     : {confidence * 100:.2f}%")