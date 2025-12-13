import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from tensorflow.keras.applications.efficientnet import preprocess_input
from PIL import Image

# =========================
# CONFIG
# =========================
MODEL_PATH = r"new_efficientnet_model_fixed.h5"
IMG_SIZE = (224, 224)

# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# =========================
# UI
# =========================
st.set_page_config(page_title="TB Detection", layout="centered")
st.title("🫁 Tuberculosis Detection")
st.write("Upload a chest X-ray image to classify **Normal** or **Tuberculosis**.")

uploaded_file = st.file_uploader(
    "Upload Chest X-ray Image",
    type=["jpg", "jpeg", "png"]
)

# =========================
# IMAGE PROCESSING FUNCTION
# =========================
def preprocess_image(img):
    img = np.array(img)
    img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
    img = cv2.resize(img, IMG_SIZE)
    img = preprocess_input(img.astype("float32"))
    img = np.expand_dims(img, axis=0)
    return img

# =========================
# PREDICTION
# =========================
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="Uploaded Image", use_container_width=True)

    processed_image = preprocess_image(image)

    with st.spinner("Analyzing image..."):
        pred = model.predict(processed_image)[0][0]

    if pred >= 0.5:
        label = "Tuberculosis"
        confidence = pred
        st.error(f"🧬 **{label}** detected")
    else:
        label = "Normal"
        confidence = 1 - pred
        st.success(f"✅ **{label}**")

    st.metric("Confidence", f"{confidence * 100:.2f}%")

    # Optional probability bar
    st.progress(float(confidence))
