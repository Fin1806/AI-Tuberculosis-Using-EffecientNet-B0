import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
from tensorflow.keras.models import load_model
import os

# ==========================================
# 1. KONFIGURASI & PATH
# ==========================================
st.set_page_config(page_title="TB Detection - EfficientNet", layout="wide")

# Cari folder tempat script ini berada (Biar gak File Not Found)
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_NAME = "v7_efficientnetb0_tb_best.keras"
MODEL_PATH = os.path.join(BASE_DIR, MODEL_NAME)

IMG_SIZE = (224, 224)
LAST_CONV_LAYER = 'top_activation' # Layer untuk Grad-CAM

# ==========================================
# 2. LOADER MODEL (DENGAN FIX 92%)
# ==========================================
@st.cache_resource
def load_ai_model(path):
    try:
        # Load model tanpa compile biar cepat
        model = load_model(path, compile=False)
        
        # 🔥 RAHASIA 92%: Matikan aktivasi layer terakhir
        # Agar outputnya murni (Logits), lalu kita Sigmoid-kan manual nanti.
        model.layers[-1].activation = None
        
        return model
    except Exception as e:
        st.error(f"❌ Gagal memuat model. Pastikan file '{MODEL_NAME}' ada di folder: {BASE_DIR}")
        st.error(f"Detail Error: {e}")
        return None

model = load_ai_model(MODEL_PATH)

# ==========================================
# 3. PREPROCESSING (TEKNIK MASKING)
# ==========================================
def process_image(image_file):
    # 1. Buka Gambar & Resize
    img = Image.open(image_file).convert('RGB')
    img = img.resize(IMG_SIZE)
    original_array = np.array(img) # Gambar asli (bersih) untuk display manusia
    
    # 2. Buat Versi Masking (Untuk AI)
    # Ini meniru logika "Good Code" kamu: Sudut-sudut dihitamkan 30%
    masked_array = original_array.copy()
    h, w, _ = masked_array.shape
    mask_h, mask_w = int(h * 0.30), int(w * 0.30)
    
    masked_array[0:mask_h, 0:mask_w, :] = 0.0      # Kiri Atas
    masked_array[0:mask_h, w-mask_w:w, :] = 0.0    # Kanan Atas
    # (Biasanya masking paru-paru bagian atas saja yang sering noise)
    
    # 3. Preprocess Input (EfficientNet standard)
    img_tensor = np.expand_dims(masked_array, axis=0)
    img_tensor = preprocess_input(img_tensor)
    
    return img_tensor, original_array

# ==========================================
# 4. GRAD-CAM (VISUALISASI)
# ==========================================
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    grads = tape.gradient(class_channel, last_conv_layer_output)
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(clean_img, heatmap):
    # Resize heatmap agar sama dengan gambar asli
    heatmap_resized = cv2.resize(heatmap, (clean_img.shape[1], clean_img.shape[0]))
    
    # Ubah jadi warna (JET Colormap)
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB) # Balik ke RGB
    
    # Tempelkan ke gambar asli (Superimpose)
    superimposed_img = cv2.addWeighted(clean_img, 0.6, heatmap_color, 0.4, 0)
    return superimposed_img

# ==========================================
# 5. USER INTERFACE (STREAMLIT)
# ==========================================
st.title("🩻 EfficientNet-B0 TB Detector")
st.write("Versi Original (Single Model) dengan Masking Preprocessing.")

uploaded_file = st.file_uploader("Upload X-Ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file and model is not None:
    # A. Proses Gambar
    img_tensor, original_display = process_image(uploaded_file)
    
    # B. Prediksi
    preds = model.predict(img_tensor, verbose=0)
    
    # 🔥 Manual Sigmoid (Karena activation di model sudah dimatikan)
    score = tf.nn.sigmoid(preds[0][0]).numpy()
    
    # C. Logic Label
    threshold = 0.5
    if score > threshold:
        label = "TUBERCULOSIS"
        confidence = score * 100
        color = "#d32f2f" # Merah
        icon = "⚠️"
    else:
        label = "NORMAL"
        confidence = (1 - score) * 100
        color = "#2e7d32" # Hijau
        icon = "✅"

    # D. Tampilkan Hasil
    st.divider()
    col1, col2 = st.columns(2)
    
    with col1:
        st.image(original_display, caption="Gambar Asli", use_container_width=True)
        
    with col2:
        # Tampilan Kartu Hasil
        st.markdown(f"""
        <div style="
            background-color: white; 
            padding: 20px; 
            border-radius: 10px; 
            border: 2px solid {color};
            text-align: center;
            box-shadow: 0 4px 6px rgba(0,0,0,0.1);">
            <h2 style="color: {color}; margin:0;">{icon} {label}</h2>
            <h1 style="font-size: 3em; margin: 10px 0;">{confidence:.2f}%</h1>
            <p style="color: grey;">Confidence Level</p>
        </div>
        """, unsafe_allow_html=True)
        
        # Visualisasi Grad-CAM
        try:
            heatmap = make_gradcam_heatmap(img_tensor, model, LAST_CONV_LAYER)
            final_cam = overlay_heatmap(original_display, heatmap)
            st.image(final_cam, caption="🔍 Analisis AI (Grad-CAM)", use_container_width=True)
        except Exception as e:
            st.warning(f"Grad-CAM gagal: {e}")

elif model is None:
    st.warning("⚠️ Model belum dimuat. Pastikan file .keras ada di folder yang sama.")