import streamlit as st
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2
import pandas as pd
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- 1. CONFIGURATION ---
st.set_page_config(page_title="Hybrid Model Comparator", layout="wide")

# Constants
IMG_SIZE = (224, 224)

# ==========================================
# ⚙️ CONFIG MODEL
# ==========================================
MODELS_CONFIG = {
    "EfficientNet-B0": {
        "type": "tensorflow",
        "path": "v7_efficientnetb0_tb_best.keras",
        "layer": "top_activation", # Hanya ini yang butuh nama layer
        "use_gradcam": True
    },
    "TB-Net": {
        "type": "tensorflow",
        "path": "tb_net_fixed.keras", # Pastikan file .keras/.h5 (bukan zip/folder)
        "layer": None,
        "use_gradcam": False # Matikan GradCAM
    },
    "ResNet50 (PyTorch)": {
        "type": "pytorch",
        "path": "best_resnet50_tb.pth", # File .pth kamu
        "layer": None,
        "use_gradcam": False # Matikan GradCAM
    }
}

# --- 2. LOADERS ---

@st.cache_resource
def load_tf_model(path):
    try:
        model = tf.keras.models.load_model(path, compile=False)
        return model
    except Exception as e:
        st.error(f"Gagal load TF model {path}: {e}")
        return None

@st.cache_resource
def load_torch_model(path):
    try:
        # Asumsi kamu save full model (torch.save(model, path))
        # Kalau kamu cuma save weights, kodenya beda dikit (harus define arsitektur dulu)
        device = torch.device('cpu')
        model = torch.load(path, map_location=device)
        model.eval() # Set ke mode evaluasi
        return model
    except Exception as e:
        st.error(f"Gagal load PyTorch model. Pastikan format .pth benar. Error: {e}")
        return None

# Load Semua Model ke Dictionary
models_loaded = {}
for name, conf in MODELS_CONFIG.items():
    if conf['type'] == 'tensorflow':
        models_loaded[name] = load_tf_model(conf['path'])
    elif conf['type'] == 'pytorch':
        models_loaded[name] = load_torch_model(conf['path'])

# --- 3. IMAGE PREPROCESSING ---

def process_image_tf(image_file):
    """Preprocessing khusus TensorFlow (EfficientNet)"""
    img = Image.open(image_file).convert('RGB')
    img = img.resize(IMG_SIZE)
    arr = np.array(img)
    
    # Masking Logic (Sama kayak training)
    masked = arr.copy()
    h, w, _ = masked.shape
    masked[0:int(h*0.3), 0:int(w*0.3), :] = 0
    masked[0:int(h*0.3), w-int(w*0.3):w, :] = 0
    
    # Preprocess Input EfficientNet
    inp = preprocess_input(masked.astype(np.float32))
    return np.expand_dims(inp, axis=0), arr

def process_image_torch(image_file):
    """Preprocessing khusus PyTorch (ResNet)"""
    img = Image.open(image_file).convert('RGB')
    
    # PyTorch butuh transformasi spesifik
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(), # Ubah jadi 0-1 dan Channel First (CHW)
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = preprocess(img)
    input_batch = input_tensor.unsqueeze(0) # Tambah batch dimension
    return input_batch

# --- 4. GRAD-CAM (Hanya TF) ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name):
    try:
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
    except:
        return None

def overlay_heatmap(clean_img, heatmap):
    heatmap_resized = cv2.resize(heatmap, (clean_img.shape[1], clean_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    superimposed_img = cv2.addWeighted(clean_img, 0.6, cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB), 0.4, 0)
    return superimposed_img

# --- 5. MAIN UI ---
st.title("🔬 Hybrid AI Comparison (TF + PyTorch)")

uploaded_file = st.file_uploader("Upload X-Ray", type=["jpg", "png"])

if uploaded_file:
    # Tampilkan Gambar Asli
    img_display = Image.open(uploaded_file)
    st.image(img_display, caption="Original Image", width=300)
    
    st.markdown("---")
    cols = st.columns(len(models_loaded))
    
    # Loop Models
    for idx, (name, model) in enumerate(models_loaded.items()):
        if model is None: continue
        
        config = MODELS_CONFIG[name]
        
        with cols[idx]:
            st.subheader(name)
            
            # --- PREDIKSI ---
            probability = 0.0
            
            # A. Kalo Model TensorFlow
            if config['type'] == 'tensorflow':
                input_tf, _ = process_image_tf(uploaded_file)
                preds = model.predict(input_tf, verbose=0)
                probability = tf.nn.sigmoid(preds[0][0]).numpy()
            
            # B. Kalo Model PyTorch
            elif config['type'] == 'pytorch':
                input_torch = process_image_torch(uploaded_file)
                with torch.no_grad():
                    outputs = model(input_torch)
                    # Asumsi output layer linear terakhir 2 kelas (atau 1 kelas sigmoid)
                    # Sesuaikan logika ini dengan model PyTorch mu:
                    probs = torch.softmax(outputs, dim=1) 
                    probability = probs[0][1].item() # Ambil probabilitas kelas 1 (TB)
            
            # --- TAMPILKAN HASIL ---
            threshold = 0.5
            if probability > threshold:
                label = "TUBERCULOSIS"
                color = "#ffebee"
                txt_color = "red"
            else:
                label = "NORMAL"
                color = "#e8f5e9"
                txt_color = "green"
            
            conf_percent = probability * 100 if label=="TUBERCULOSIS" else (1-probability)*100
            
            st.markdown(f"""
            <div style="background-color: {color}; padding: 10px; border-radius: 5px; text-align: center; border: 1px solid {txt_color};">
                <h4 style="color: {txt_color}; margin:0;">{label}</h4>
                <h2 style="margin:0;">{conf_percent:.1f}%</h2>
            </div>
            """, unsafe_allow_html=True)
            
            # --- VISUALISASI GRAD-CAM (Khusus EfficientNet) ---
            if config['use_gradcam']:
                st.write("---")
                st.caption("Visualisasi AI:")
                input_tf, original_arr = process_image_tf(uploaded_file)
                heatmap = make_gradcam_heatmap(input_tf, model, config['layer'])
                if heatmap is not None:
                    final_img = overlay_heatmap(original_arr, heatmap)
                    st.image(final_img, caption="Area Deteksi", use_container_width=True)
                else:
                    st.warning("Layer GradCAM tidak ditemukan.")
            else:
                st.write("---")
                st.caption("Visualisasi tidak tersedia untuk model ini.")

elif not models_loaded:
    st.error("Belum ada model yang berhasil di-load.")