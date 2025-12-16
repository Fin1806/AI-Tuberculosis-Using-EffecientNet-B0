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

# ==========================================
# 1. SETUP HALAMAN & CSS (TAMPILAN UI)
# ==========================================
st.set_page_config(
    page_title="TB Detection Dashboard",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# Custom CSS untuk membuat tampilan "Flex" dan "Center"
st.markdown("""
    <style>
    /* Mengatur background global */
    .stApp {
        background-color: #f8f9fa;
    }
    
    /* Judul di tengah */
    h1 {
        text-align: center;
        color: #2c3e50;
        font-family: 'Helvetica', sans-serif;
        margin-bottom: 10px;
    }
    p {
        text-align: center;
        color: #7f8c8d;
    }

    /* Style untuk Kartu Hasil (Card) */
    .result-card {
        background-color: white;
        padding: 25px;
        border-radius: 15px;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        text-align: center;
        transition: transform 0.3s;
        margin-bottom: 20px;
    }
    .result-card:hover {
        transform: translateY(-5px);
    }
    .model-name {
        font-weight: bold;
        color: #34495e;
        margin-bottom: 10px;
        font-size: 1.1rem;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    .prediction-label {
        font-size: 1.5rem;
        font-weight: 800;
        margin: 10px 0;
    }
    .confidence-score {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
    }
    
    /* Menghilangkan padding atas default Streamlit */
    .block-container {
        padding-top: 2rem;
    }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DEFINISI CLASS PYTORCH (WAJIB SESUAI)
# ==========================================
# ⚠️ PASTIKAN STRUKTUR INI SAMA DENGAN NOTEBOOK TRAINING KAMU
class ResNet50_TB(nn.Module):
    def __init__(self, num_classes=1): # Cek num_classes kamu (1 atau 2?)
        super(ResNet50_TB, self).__init__()
        self.resnet = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
        
        # Ganti head sesuai kebiasaan umum
        in_features = self.resnet.fc.in_features
        self.resnet.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid() # Hapus baris ini jika saat training kamu pakai BCEWithLogitsLoss
        )

    def forward(self, x):
        return self.resnet(x)

# ==========================================
# 3. KONFIGURASI MODEL & PATH
# ==========================================
MODELS_CONFIG = {
    "EfficientNet-B0": {
        "type": "tf_eff", 
        "path": "v7_efficientnetb0_tb_best.keras", 
        "layer": "top_activation",
        "cam": True # Nyalakan Grad-CAM
    },
    "TB-Net (Custom)": {
        "type": "tf_custom", 
        "path": "tb_net_fixed.keras",
        "layer": "conv2d_final", 
        "cam": False 
    },
    "ResNet50 (PyTorch)": {
        "type": "torch",
        "path": "best_resnet50_tb.pth", 
        "layer": None,
        "cam": False
    }
}

IMG_SIZE = (224, 224)

# ==========================================
# 4. LOADERS (CACHE)
# ==========================================
@st.cache_resource
def load_tf_model(path):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        st.error(f"❌ Gagal load {path}: {e}")
        return None

@st.cache_resource
def load_torch_model(path):
    try:
        device = torch.device('cpu') 
        # Load full model (Architecture + Weights)
        model = torch.load(path, map_location=device)
        model.eval()
        return model
    except AttributeError:
        st.error("❌ Struktur Class PyTorch beda! Copy class dari notebook trainingmu ke script ini.")
        return None
    except Exception as e:
        st.error(f"❌ Gagal load PyTorch: {e}")
        return None

# Load Models
models_loaded = {}
for name, conf in MODELS_CONFIG.items():
    if "tf" in conf['type']:
        models_loaded[name] = load_tf_model(conf['path'])
    elif conf['type'] == "torch":
        models_loaded[name] = load_torch_model(conf['path'])

# ==========================================
# 5. PREPROCESSING FUNCTIONS
# ==========================================
def process_tf_eff(image_file):
    img = Image.open(image_file).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img)
    # Masking
    h, w, _ = arr.shape
    arr[0:int(h*0.3), 0:int(w*0.3), :] = 0
    arr[0:int(h*0.3), w-int(w*0.3):w, :] = 0
    # EfficientNet Preprocessing (-1 to 1)
    return np.expand_dims(preprocess_input(arr.astype(np.float32)), axis=0), arr

def process_tf_custom(image_file):
    img = Image.open(image_file).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
    # Simple Rescale (0 to 1) <-- Fix untuk TB-Net 50%
    arr = arr / 255.0
    return np.expand_dims(arr, axis=0)

def process_torch(image_file):
    img = Image.open(image_file).convert('RGB')
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])
    return transform(img).unsqueeze(0)

# ==========================================
# 6. GRAD-CAM UTILS
# ==========================================
def make_gradcam(img_array, model, layer_name):
    try:
        grad_model = tf.keras.models.Model(
            [model.inputs], [model.get_layer(layer_name).output, model.output]
        )
        with tf.GradientTape() as tape:
            conv_out, preds = grad_model(img_array)
            idx = tf.argmax(preds[0])
            channel = preds[:, idx]
        
        grads = tape.gradient(channel, conv_out)
        pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
        heatmap = conv_out[0] @ pooled_grads[..., tf.newaxis]
        heatmap = tf.squeeze(heatmap)
        heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
        return heatmap.numpy()
    except:
        return None

def overlay_cam(clean_img, heatmap):
    heatmap = cv2.resize(heatmap, (clean_img.shape[1], clean_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(clean_img, 0.6, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), 0.4, 0)

# ==========================================
# 7. MAIN UI LAYOUT
# ==========================================

# A. Header Section (Centered)
st.markdown("<h1>🩻 AI Tuberculosis Diagnosis System</h1>", unsafe_allow_html=True)
st.markdown("<p>Upload foto X-Ray Paru-paru untuk dianalisis oleh 3 model AI sekaligus.</p>", unsafe_allow_html=True)

# B. Upload Section (Centered with Columns)
col_spacer1, col_upload, col_spacer2 = st.columns([1, 2, 1])
with col_upload:
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file and models_loaded:
    # C. Display Original Image (Centered)
    input_eff, display_img = process_tf_eff(uploaded_file)
    
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.image(display_img, caption="📷 Gambar Input Asli", use_container_width=True)

    st.markdown("<br>", unsafe_allow_html=True) # Spacer

    # D. Prediction Loop & Flex Cards
    # Kita pakai st.columns sesuai jumlah model
    result_cols = st.columns(len(models_loaded))
    
    for idx, (name, model) in enumerate(models_loaded.items()):
        if model is None: continue
        
        config = MODELS_CONFIG[name]
        prob = 0.0
        
        # --- LOGIC PREDIKSI ---
        try:
            if config['type'] == 'tf_eff':
                preds = model.predict(input_eff, verbose=0)
                prob = tf.nn.sigmoid(preds[0][0]).numpy()
            
            elif config['type'] == 'tf_custom':
                inp = process_tf_custom(uploaded_file)
                preds = model.predict(inp, verbose=0)
                prob = preds[0][0] # Asumsi output sudah probability (0-1)
                
            elif config['type'] == 'torch':
                inp = process_torch(uploaded_file)
                with torch.no_grad():
                    out = model(inp)
                    prob = out.item() # Asumsi output 1 neuron sigmoid
        except Exception as e:
            st.error(f"Error {name}: {e}")
            continue

        # --- LOGIC TAMPILAN KARTU ---
        is_tb = prob > 0.5
        label = "TUBERCULOSIS" if is_tb else "NORMAL"
        # Warna: Merah muda (TB) / Hijau muda (Normal)
        color_code = "#d32f2f" if is_tb else "#2e7d32" 
        icon = "⚠️" if is_tb else "✅"
        conf_percent = prob * 100 if is_tb else (1-prob)*100
        
        with result_cols[idx]:
            # HTML Card Injection
            st.markdown(f"""
            <div class="result-card">
                <div class="model-name">{name}</div>
                <div class="prediction-label" style="color: {color_code};">
                    {icon} {label}
                </div>
                <div class="confidence-score">
                    {conf_percent:.1f}%
                </div>
                <div style="color: grey; font-size: 0.8rem; margin-top:5px;">Confidence Level</div>
            </div>
            """, unsafe_allow_html=True)
            
            # --- GRAD-CAM DISPLAY ---
            if config['cam']:
                heatmap = make_gradcam(input_eff, model, config['layer'])
                if heatmap is not None:
                    final_cam = overlay_cam(display_img, heatmap)
                    st.image(final_cam, caption="🔍 Area Deteksi AI", use_container_width=True)
            elif name == "ResNet50 (PyTorch)":
                st.caption("Visualisasi tidak tersedia untuk PyTorch.")

elif not models_loaded:
    st.warning("⚠️ Menunggu model dimuat... Pastikan file model ada di folder yang sama.")