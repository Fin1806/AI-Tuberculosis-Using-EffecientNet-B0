import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- 1. CONFIGURATION & PAGE SETUP ---
st.set_page_config(
    page_title="Multi-Model TB Comparator",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed" # Hide sidebar biar luas
)

# --- CUSTOM CSS ---
st.markdown("""
    <style>
    .main { background-color: #f8f9fa; }
    .result-card {
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 10px;
    }
    .stProgress > div > div > div > div {
        background-color: #2c3e50;
    }
    </style>
    """, unsafe_allow_html=True)

# Constants
IMG_SIZE = (224, 224)

# ==========================================
# ⚡ CONFIGURATION: DAFTARKAN MODEL DISINI
# ==========================================
# Ganti nama file .keras dan nama layer conv terakhir (untuk GradCAM)
# Tips: Untuk EfficientNetB0 biasanya 'top_activation' atau 'block7a_project_conv'
# Untuk ResNet50 biasanya 'conv5_block3_out'
MODELS_CONFIG = {
    "EfficientNet-B0": {
        "path": "v7_efficientnetb0_tb_best.keras", 
        "layer": "top_activation"
    },
    "ResNet50": {
        "path": "best_resnet50_tb.pth",  # Ganti dengan file modelmu
        "layer": "conv5_block3_out"         # Sesuaikan layer
    },
    "TB-Net": {
        "path": "best_tbnet_model.keras",     # Ganti dengan file modelmu
        "layer": "block5_conv3"             # Sesuaikan layer
    }
}

# --- 2. MODEL LOADING (MULTI MODEL) ---
@st.cache_resource
def load_all_models():
    loaded_models = {}
    for name, config in MODELS_CONFIG.items():
        try:
            # Load model (compile=False biar cepat)
            model = tf.keras.models.load_model(config['path'], compile=False)
            
            # Hapus aktivasi layer terakhir untuk Grad-CAM yang lebih baik (Opsional)
            # Cek dulu apakah layer terakhir ada aktivasi
            if hasattr(model.layers[-1], 'activation'):
                model.layers[-1].activation = None
                
            loaded_models[name] = model
        except Exception as e:
            st.error(f"⚠️ Gagal memuat model {name}: {e}")
    return loaded_models

# Load Models saat aplikasi mulai
models_dict = load_all_models()

# --- 3. PROCESSING FUNCTIONS ---
def process_image(image_file):
    img = Image.open(image_file).convert('RGB')
    img = img.resize(IMG_SIZE)
    original_array = np.array(img)
    
    # --- LOGIKA MASKING (SAMA UNTUK SEMUA MODEL) ---
    masked_array = original_array.copy()
    height, width, _ = masked_array.shape
    mask_h = int(height * 0.30) 
    mask_w = int(width * 0.30)
    
    # Masking Pojok
    masked_array[0:mask_h, 0:mask_w, :] = 0.0  
    masked_array[0:mask_h, width-mask_w:width, :] = 0.0 
    
    masked_float = masked_array.astype(np.float32)
    # Preprocessing EfficientNet (bisa beda kalau model lain butuh input beda)
    preprocessed_img = preprocess_input(masked_float)
    img_batch = np.expand_dims(preprocessed_img, axis=0)
    
    return img_batch, original_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            last_conv_layer_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, last_conv_layer_output)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        last_conv_layer_output = last_conv_layer_output[0]
        heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except Exception as e:
        return None # Return None jika nama layer salah

def overlay_heatmap(clean_img, heatmap, alpha=0.4):
    if heatmap is None: return clean_img
    heatmap_resized = cv2.resize(heatmap, (clean_img.shape[1], clean_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(clean_img, 0.6, heatmap_color, 0.4, 0)
    return superimposed_img

# --- 4. MAIN UI ---
st.title("🔬 TB Detection: Model Comparison")
st.write("Bandingkan performa 3 model AI berbeda secara real-time.")

uploaded_file = st.file_uploader("📂 Upload X-Ray Image", type=["jpg", "png", "jpeg"])

if uploaded_file and models_dict:
    # 1. Process Image
    processed_input, display_img = process_image(uploaded_file)
    
    # Tampilkan Gambar Asli di Tengah
    c1, c2, c3 = st.columns([1, 2, 1])
    with c2:
        st.image(display_img, caption="Original Input (Clean)", use_container_width=True)
    
    st.markdown("---")
    st.subheader("📊 Hasil Perbandingan")

    # Siapkan Kolom sesuai jumlah model
    cols = st.columns(len(models_dict))
    
    # Dataframe untuk rang