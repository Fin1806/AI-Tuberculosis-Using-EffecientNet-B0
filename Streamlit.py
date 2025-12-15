import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- 1. CONFIGURATION & PAGE SETUP ---
st.set_page_config(
    page_title="Tuberculosis AI Detector",
    page_icon="🫁",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CUSTOM CSS FOR BETTER UI ---
st.markdown("""
    <style>
    .main {
        background-color: #f8f9fa; 
    }
    .stButton>button {
        width: 100%;
        border-radius: 5px;
        height: 3em;
    }
    .reportview-container .main .block-container{
        padding-top: 2rem;
    }
    h1 {
        color: #2c3e50;
    }
    .result-card {
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
        margin-bottom: 20px;
    }
    </style>
    """, unsafe_allow_html=True)

# Disable scientific notation
np.set_printoptions(suppress=True)

# Constants
IMG_SIZE = (224, 224)
LAST_CONV_LAYER = 'top_activation'

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_tb_model():
    try:
        # Load model without compiling needed for inference only
        model = tf.keras.models.load_model('v6_efficientnetb0_tb_best.keras', compile=False)
        # Remove activation for better Grad-CAM
        model.layers[-1].activation = None
        return model
    except Exception as e:
        st.error(f"Gagal memuat model: {e}")
        return None

model = load_tb_model()

# --- 3. PROCESSING FUNCTIONS ---
def process_image(image_file):
    """
    Returns:
    1. img_batch: Masked (For AI)
    2. original_array: Clean (For Display)
    """
    img = Image.open(image_file).convert('RGB')
    img = img.resize(IMG_SIZE)
    original_array = np.array(img)
    
    # Create Masked Version for AI
    masked_array = original_array.copy()
    height, width, _ = masked_array.shape
    mask_h = int(height * 0.30) 
    mask_w = int(width * 0.30)
    
    masked_array[0:mask_h, 0:mask_w, :] = 0.0  
    masked_array[0:mask_h, width-mask_w:width, :] = 0.0 
    
    masked_float = masked_array.astype(np.float32)
    preprocessed_img = preprocess_input(masked_float)
    img_batch = np.expand_dims(preprocessed_img, axis=0)
    
    return img_batch, original_array

def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
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

def overlay_heatmap(clean_img, heatmap, alpha=0.4):
    heatmap_resized = cv2.resize(heatmap, (clean_img.shape[1], clean_img.shape[0]))
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    superimposed_img = cv2.addWeighted(clean_img, 0.6, heatmap_color, 0.4, 0)
    return superimposed_img

# --- 4. SIDEBAR ---
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3004/3004458.png", width=100)
    st.title("TB AI Diagnostic")
    st.markdown("---")
    st.write("Upload hasil X-Ray dada pasien untuk melakukan deteksi dini Tuberculosis.")
    
    uploaded_file = st.file_uploader("📂 Upload X-Ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
    
    st.markdown("---")
    st.info("**Panduan:**\nPastikan gambar X-Ray terlihat jelas, tidak buram, dan mencakup seluruh area paru-paru.")

# --- 5. MAIN CONTENT ---
st.title("🫁 Sistem Deteksi Tuberculosis")
st.markdown("Menggunakan **EfficientNet-B0** dan **Explainable AI (Grad-CAM)**")

if uploaded_file is None:
    # Landing Page State
    st.warning("👈 Silakan unggah gambar melalui panel di sebelah kiri untuk memulai analisa.")
    
    # Placeholder / Demo UI
    col_demo1, col_demo2, col_demo3 = st.columns(3)
    with col_demo1:
        st.markdown("### 1. Upload")
        st.caption("Unggah citra Chest X-Ray pasien.")
    with col_demo2:
        st.markdown("### 2. Analisis AI")
        st.caption("Model memproses gambar & mencari pola infeksi.")
    with col_demo3:
        st.markdown("### 3. Hasil & Visualisasi")
        st.caption("Dapatkan hasil deteksi beserta peta panas area infeksi.")

else:
    # Processing State
    if model:
        # Process
        processed_input, display_img = process_image(uploaded_file)
        
        # Predict
        preds = model.predict(processed_input)
        probability = tf.nn.sigmoid(preds[0][0]).numpy()
        
        threshold = 0.5
        if probability > threshold:
            label = "TUBERCULOSIS"
            status_color = "#ffebee" # Light Red
            text_color = "#c62828"   # Dark Red
            confidence = probability * 100
            icon = "⚠️"
        else:
            label = "NORMAL"
            status_color = "#e8f5e9" # Light Green
            text_color = "#2e7d32"   # Dark Green
            confidence = (1 - probability) * 100
            icon = "✅"

        # --- RESULT DASHBOARD ---
        
        # Row 1: Metrics
        col_res1, col_res2 = st.columns([1, 2])
        
        with col_res1:
            st.image(display_img, caption="Citra X-Ray Asli", use_container_width=True, channels="RGB")
        
        with col_res2:
            st.markdown(f"""
            <div class="result-card" style="background-color: {status_color}; border: 2px solid {text_color};">
                <h2 style="color: {text_color}; margin:0;">{icon} {label}</h2>
                <p style="font-size: 18px; margin-top: 10px;">Confidence Score</p>
                <h1 style="color: {text_color}; font-size: 48px; margin:0;">{confidence:.2f}%</h1>
                <p style="margin-top:10px; font-style:italic;">Probabilitas TBC Raw: {probability:.4f}</p>
            </div>
            """, unsafe_allow_html=True)
            
            # Progress Bar Custom
            st.write("Tingkat Keyakinan Model:")
            bar_color = "red" if label == "TUBERCULOSIS" else "green"
            st.progress(int(confidence), text=f"Model yakin {confidence:.1f}% bahwa ini adalah {label}")

        # Row 2: Visualization Tabs
        st.markdown("### 🔍 Analisis Visual")
        tab1, tab2 = st.tabs(["🔥 Grad-CAM Heatmap", "ℹ️ Detail Teknis"])
        
        with tab1:
            try:
                with st.spinner("Membuat Visualisasi Grad-CAM..."):
                    heatmap = make_gradcam_heatmap(processed_input, model, LAST_CONV_LAYER)
                    gradcam_img = overlay_heatmap(display_img, heatmap)
                    
                    c1, c2 = st.columns(2)
                    with c1:
                        st.image(display_img, caption="Gambar Bersih (Tanpa Masking)", use_container_width=True)
                    with c2:
                        st.image(gradcam_img, caption="Overlay Area Infeksi (Grad-CAM)", use_container_width=True)
                        
                    st.info("💡 **Penjelasan:** Area berwarna **Merah/Kuning** adalah bagian paru-paru yang paling menarik perhatian AI saat mengambil keputusan.")
            except Exception as e:
                st.error(f"Gagal membuat Grad-CAM: {e}")

        with tab2:
            st.markdown("**Parameter Model:**")
            st.json({
                "Model Architecture": "EfficientNet-B0",
                "Input Size": "224x224",
                "Masking Strategy": "Top-Left & Top-Right Corners (30%)",
                "Threshold": threshold,
                "Prediction Mode": "Binary Classification (Sigmoid)"
            })

# --- FOOTER ---
st.markdown("---")
st.markdown("""
<div style="text-align: center; font-size: 12px; color: gray;">
    <p>⚠️ <strong>Disclaimer:</strong> Alat ini adalah asisten diagnostik berbasis AI dan bukan pengganti diagnosis dokter profesional.</p>
</div>
""", unsafe_allow_html=True)