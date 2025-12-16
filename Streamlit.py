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
# 1. SETUP & PATH FINDER (Supaya Gak File Not Found)
# ==========================================
st.set_page_config(page_title="TB Detection System", layout="wide")

# Cari lokasi folder tempat script ini berada
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

st.markdown(f"""
    <div style='background-color: #e3f2fd; padding: 10px; border-radius: 5px; margin-bottom: 20px;'>
        <small>📂 Working Directory: {BASE_DIR}</small>
    </div>
""", unsafe_allow_html=True)

# ==========================================
# 2. DEFINISI CLASS PYTORCH (Body Model)
# ==========================================
class ResNet50_TB(nn.Module):
    def __init__(self, num_classes=1): 
        super(ResNet50_TB, self).__init__()
        self.base_model = models.resnet50(weights=None) 
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
# 3. KONFIGURASI MODEL (PATH OTOMATIS)
# ==========================================
# Kita pakai os.path.join biar path-nya otomatis bener
MODELS_CONFIG = {
    "EfficientNet-B0": {
        "type": "tf_eff", 
        "path": os.path.join(BASE_DIR, "v7_efficientnetb0_tb_best.keras"), 
        "layer": "top_activation",
        "cam": True 
    },
    "TB-Net (Custom)": {
        "type": "tf_custom", 
        "path": os.path.join(BASE_DIR, "tb_net_fixed.keras"),
        "layer": "conv2d_final", 
        "cam": False 
    },
    "ResNet50 (PyTorch)": {
        "type": "torch",
        "path": os.path.join(BASE_DIR, "best_resnet50_tb.pth"), 
        "layer": None,
        "cam": False
    }
}

IMG_SIZE = (224, 224)

# ==========================================
# 4. LOADERS (DENGAN FIX 92%)
# ==========================================
@st.cache_resource
def load_tf_model(path, model_type):
    try:
        if not os.path.exists(path):
            st.error(f"❌ File hilang: {path}")
            return None
            
        model = tf.keras.models.load_model(path, compile=False)
        
        # === FIX 92% (CRUCIAL STEP) ===
        # Kita matikan aktivasi layer terakhir agar outputnya RAW LOGITS.
        # Ini menyamakan logika dengan code 'Good Code' kamu.
        if model_type == 'tf_eff':
            model.layers[-1].activation = None
            
        return model
    except Exception as e:
        st.error(f"❌ Gagal load TF {os.path.basename(path)}: {e}")
        return None

@st.cache_resource
def load_torch_model(path):
    device = torch.device('cpu') 
    try:
        if not os.path.exists(path):
            st.error(f"❌ File hilang: {path}")
            return None

        loaded_object = torch.load(path, map_location=device)
        
        if isinstance(loaded_object, dict):
            model = ResNet50_TB()
            model.load_state_dict(loaded_object, strict=False)
            model.eval()
            return model
        elif isinstance(loaded_object, nn.Module):
            loaded_object.eval()
            return loaded_object
        else:
            st.error("❌ Format .pth tidak dikenali.")
            return None
    except Exception as e:
        st.error(f"❌ Error PyTorch: {e}")
        return None

# Load All Models
models_loaded = {}
for name, conf in MODELS_CONFIG.items():
    if "tf" in conf['type']:
        models_loaded[name] = load_tf_model(conf['path'], conf['type'])
    elif conf['type'] == "torch":
        models_loaded[name] = load_torch_model(conf['path'])

# ==========================================
# 5. PREPROCESSING (MASKING YANG BENAR)
# ==========================================
def process_tf_eff(image_file):
    img = Image.open(image_file).convert('RGB').resize(IMG_SIZE)
    original_array = np.array(img)
    
    # --- MASKING LOGIC (SAMA PERSIS DENGAN CODE TESTING) ---
    masked_array = original_array.copy()
    h, w, _ = masked_array.shape
    mask_h, mask_w = int(h * 0.30), int(w * 0.30)
    
    # Black out corners
    masked_array[0:mask_h, 0:mask_w, :] = 0.0 
    masked_array[0:mask_h, w-mask_w:w, :] = 0.0 
    
    # Preprocess
    img_tensor = np.expand_dims(masked_array, axis=0)
    img_tensor = preprocess_input(img_tensor) # EfficientNet specific
    
    return img_tensor, original_array

def process_tf_custom(image_file):
    img = Image.open(image_file).convert('RGB').resize(IMG_SIZE)
    arr = np.array(img).astype(np.float32)
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
    except: return None

def overlay_cam(clean_img, heatmap):
    heatmap = cv2.resize(heatmap, (clean_img.shape[1], clean_img.shape[0]))
    heatmap = np.uint8(255 * heatmap)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_JET)
    return cv2.addWeighted(clean_img, 0.6, cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB), 0.4, 0)

# ==========================================
# 7. MAIN UI
# ==========================================
st.markdown("<h1 style='text-align: center;'>🩻 AI Tuberculosis Diagnosis</h1>", unsafe_allow_html=True)

col_spacer1, col_upload, col_spacer2 = st.columns([1, 2, 1])
with col_upload:
    uploaded_file = st.file_uploader("", type=["jpg", "png", "jpeg"])

if uploaded_file and models_loaded:
    input_eff, display_img = process_tf_eff(uploaded_file)
    
    c1, c2, c3 = st.columns([1, 1, 1])
    with c2:
        st.image(display_img, caption="Original Image", use_container_width=True)

    cols = st.columns(len(models_loaded))
    
    for idx, (name, model) in enumerate(models_loaded.items()):
        if model is None: continue
        config = MODELS_CONFIG[name]
        prob = 0.0
        
        try:
            if config['type'] == 'tf_eff':
                preds = model.predict(input_eff, verbose=0)
                # KARENA KITA SUDAH MATIKAN AKTIVASI DI LOADER,
                # KITA WAJIB PAKE SIGMOID DISINI (Logikanya jadi benar)
                prob = tf.nn.sigmoid(preds[0][0]).numpy()
            
            elif config['type'] == 'tf_custom':
                inp = process_tf_custom(uploaded_file)
                preds = model.predict(inp, verbose=0)
                prob = preds[0][0]
                
            elif config['type'] == 'torch':
                inp = process_torch(uploaded_file)
                with torch.no_grad():
                    out = model(inp)
                    prob = out.item() 
        except Exception as e:
            st.error(f"Error {name}: {e}")
            continue

        is_tb = prob > 0.5
        label = "TUBERCULOSIS" if is_tb else "NORMAL"
        color = "#d32f2f" if is_tb else "#2e7d32"
        conf = prob * 100 if is_tb else (1-prob)*100
        
        with cols[idx]:
            st.markdown(f"""
            <div style="background:white; padding:15px; border-radius:10px; text-align:center; box-shadow:0 2px 5px rgba(0,0,0,0.1);">
                <div style="font-weight:bold; margin-bottom:5px;">{name}</div>
                <h3 style="color:{color}; margin:0;">{label}</h3>
                <h1 style="color:#2c3e50; margin:0;">{conf:.1f}%</h1>
            </div>
            """, unsafe_allow_html=True)
            
            if config['cam']:
                heatmap = make_gradcam(input_eff, model, config['layer'])
                if heatmap is not None:
                    final = overlay_cam(display_img, heatmap)
                    st.image(final, caption="Grad-CAM", use_container_width=True)

elif not models_loaded:
    st.warning("⚠️ Beberapa model gagal dimuat. Cek pesan error di atas.")