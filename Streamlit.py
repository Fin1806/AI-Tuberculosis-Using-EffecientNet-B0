import streamlit as st
import tensorflow as tf
import torch
import torch.nn as nn
from torchvision import transforms, models
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input
import os

# ==========================================
# 1. SETUP HALAMAN & CSS
# ==========================================
st.set_page_config(
    page_title="TB Detection System",
    page_icon="🩻",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
    <style>
    .stApp { background-color: #f8f9fa; }
    h1 { text-align: center; color: #2c3e50; font-family: sans-serif; }
    p { text-align: center; color: #7f8c8d; }
    .result-card {
        background-color: white;
        padding: 20px;
        border-radius: 15px;
        box-shadow: 0 4px 10px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    .model-name { font-weight: bold; color: #34495e; margin-bottom: 5px; text-transform: uppercase; }
    .prediction-label { font-size: 1.5rem; font-weight: 800; margin: 10px 0; }
    .confidence-score { font-size: 2.5rem; font-weight: bold; color: #2c3e50; }
    </style>
""", unsafe_allow_html=True)

# ==========================================
# 2. DEFINISI CLASS PYTORCH (Body Model)
# ==========================================
# ⚠️ PENTING: Jika error "AttributeError", GANTI NAMA CLASS INI
class ResNet50_TB(nn.Module):
    def __init__(self, num_classes=1): 
        super(ResNet50_TB, self).__init__()
        # Load arsitektur dasar ResNet50
        # (weights=None agar tidak download ulang imagenet, kita mau load punya kamu)
        self.base_model = models.resnet50(weights=None) 
        
        # Definisikan Layer Akhir (Head)
        # Struktur ini standar. Kalau trainingmu beda, sesuaikan baris ini.
        self.base_model.fc = nn.Sequential(
            nn.Linear(2048, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, num_classes),
            nn.Sigmoid() 
        )

    def forward(self, x):
        return self.base_model(x)

# ==========================================
# 3. KONFIGURASI PATH (GANTI BAGIAN INI!)
# ==========================================
MODELS_CONFIG = {
    "EfficientNet-B0": {
        "type": "tf_eff", 
        "path": "v7_efficientnetb0_tb_best.keras", # Pastikan file ini ada
        "layer": "top_activation",
        "cam": True 
    },
    "TB-Net (Custom)": {
        "type": "tf_custom", 
        "path": "tb_net_fixed.keras", # Pastikan file ini ada
        "layer": "conv2d_final", 
        "cam": False 
    },
    "ResNet50 (PyTorch)": {
        "type": "torch",
        # 👇👇👇 GANTI PATH INI DENGAN PATH LENGKAP LAPTOP KAMU 👇👇👇
        # Contoh: r"D:\AI\Project\best_resnet50_tb.pth"
        "path": r"best_resnet50_tb.pth", 
        "layer": None,
        "cam": False
    }
}

IMG_SIZE = (224, 224)

# ==========================================
# 4. SMART LOADERS (ANTI-CRASH)
# ==========================================
@st.cache_resource
def load_tf_model(path):
    try:
        return tf.keras.models.load_model(path, compile=False)
    except Exception as e:
        st.error(f"❌ Gagal load TF {path}: {e}")
        return None

@st.cache_resource
def load_torch_model(path):
    device = torch.device('cpu') 
    try:
        if not os.path.exists(path):
            st.error(f"❌ File tidak ditemukan: {path}. Cek Path di CONFIG!")
            return None

        # 1. Coba Load File
        loaded_object = torch.load(path, map_location=device)
        
        # 2. Cek Tipe Isinya
        if isinstance(loaded_object, dict):
            # KASUS A: Isinya cuma Bobot (State Dict)
            # st.toast("Info: File .pth berisi weights. Membangun arsitektur...", icon="ℹ️")
            model = ResNet50_TB() # Panggil Class di atas
            model.load_state_dict(loaded_object, strict=False)
            model.eval()
            return model
            
        elif isinstance(loaded_object, nn.Module):
            # KASUS B: Isinya Model Utuh
            loaded_object.eval()
            return loaded_object
            
        else:
            st.error("❌ Format file tidak dikenali (Bukan dict / Module).")
            return None

    except AttributeError as e:
        st.error(f"❌ Class Name Salah! File .pth ini butuh class lain, bukan 'ResNet50_TB'.\nError Detail: {e}")
        return None
    except Exception as e:
        st.error(f"❌ Error PyTorch: {e}")
        return None

# Load All Models
models_loaded = {}
for name, conf in MODELS_CONFIG.items():
    if "tf" in conf['type']:
        models_loaded[name] = load_tf_model(conf['path'])
    elif conf['type'] == "torch":
        models_loaded[name] = load_torch_model(conf['path'])

# ==========================================
# 5. PREPROCESSING
# ==========================================
def process_tf_eff(image_file):
    img = Image.open(image_file).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img)
    # Masking
    h, w, _ = arr.shape
    arr[0:int(h*0.3), 0:int(w*0.3), :] = 0
    arr[0:int(h*0.3), w-int(w*0.3):w, :] = 0
    return np.expand_dims(preprocess_input(arr.astype(np.float32)), axis=0), arr

def process_tf_custom(image_file):
    img = Image.open(image_file).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
    arr = arr / 255.0 # Fix 50% issue
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
# 6. GRAD-CAM
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
# 7. MAIN UI
# ==========================================
st.markdown("<h1>🩻 AI Tuberculosis Diagnosis System</h1>", unsafe_allow_html=True)
st.markdown("<p>Multi-Model Analysis: EfficientNet vs TB-Net vs ResNet50</p>", unsafe_allow_html=True)

# Upload
col_spacer1, col_upload, col_spacer2 = st.columns([1, 2, 1])
with col_upload:
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file and models_loaded:
    # Display Image
    input_eff, display_img = process_tf_eff(uploaded_file)
    
    st.markdown("---")
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.image(display_img, caption="📷 Gambar Input", use_container_width=True)
    st.markdown("<br>", unsafe_allow_html=True)

    # Prediction
    cols = st.columns(len(models_loaded))
    
    for idx, (name, model) in enumerate(models_loaded.items()):
        if model is None: continue
        
        config = MODELS_CONFIG[name]
        prob = 0.0
        
        try:
            if config['type'] == 'tf_eff':
                preds = model.predict(input_eff, verbose=0)
                prob = tf.nn.sigmoid(preds[0][0]).numpy()
            
            elif config['type'] == 'tf_custom':
                inp = process_tf_custom(uploaded_file)
                preds = model.predict(inp, verbose=0)
                prob = preds[0][0] # Asumsi output 0-1
                
            elif config['type'] == 'torch':
                inp = process_torch(uploaded_file)
                with torch.no_grad():
                    out = model(inp)
                    prob = out.item() 
        except Exception as e:
            st.error(f"Error {name}: {e}")
            continue

        # UI Card
        is_tb = prob > 0.5
        label = "TUBERCULOSIS" if is_tb else "NORMAL"
        color = "#d32f2f" if is_tb else "#2e7d32" 
        icon = "⚠️" if is_tb else "✅"
        conf = prob * 100 if is_tb else (1-prob)*100
        
        with cols[idx]:
            st.markdown(f"""
            <div class="result-card">
                <div class="model-name">{name}</div>
                <div class="prediction-label" style="color: {color};">
                    {icon} {label}
                </div>
                <div class="confidence-score">
                    {conf:.1f}%
                </div>
                <div style="color: grey; font-size: 0.8rem;">Confidence</div>
            </div>
            """, unsafe_allow_html=True)
            
            if config['cam']:
                heatmap = make_gradcam(input_eff, model, config['layer'])
                if heatmap is not None:
                    final = overlay_cam(display_img, heatmap)
                    st.image(final, caption="Visualisasi Grad-CAM", use_container_width=True)
            elif name == "ResNet50 (PyTorch)":
                st.caption("Visualisasi tidak tersedia.")

elif not models_loaded:
    st.warning("⚠️ Menunggu model dimuat... Silakan cek path file di kodingan.")