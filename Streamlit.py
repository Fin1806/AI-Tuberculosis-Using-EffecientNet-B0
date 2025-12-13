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
# CUSTOM CSS
# =========================
st.markdown("""
<style>
    /* Main container styling */
    .main {
        padding-top: 2rem;
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 2rem 0 3rem 0;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
    }
    
    .header-title {
        color: white;
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .header-subtitle {
        color: #f0f0f0;
        font-size: 1.2rem;
        font-weight: 400;
    }
    
    /* Upload section */
    .upload-section {
        background: #f8f9fa;
        padding: 2rem;
        border-radius: 12px;
        border: 2px dashed #cbd5e0;
        margin: 2rem 0;
        text-align: center;
    }
    
    /* Result cards */
    .result-card {
        padding: 2rem;
        border-radius: 12px;
        margin: 1rem 0;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    
    .result-positive {
        background: linear-gradient(135deg, #f093fb 0%, #f5576c 100%);
        color: white;
    }
    
    .result-negative {
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        color: white;
    }
    
    .result-title {
        font-size: 2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .confidence-value {
        font-size: 2.5rem;
        font-weight: 800;
        margin: 1rem 0;
    }
    
    /* Info boxes */
    .info-box {
        background: #e6f3ff;
        border-left: 4px solid #3182ce;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    .warning-box {
        background: #fff5e6;
        border-left: 4px solid #ed8936;
        padding: 1rem;
        border-radius: 8px;
        margin: 1rem 0;
    }
    
    /* Image display */
    .image-container {
        border-radius: 12px;
        overflow: hidden;
        box-shadow: 0 8px 16px rgba(0,0,0,0.15);
        margin: 2rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        font-weight: 600;
        padding: 0.75rem 2rem;
        border-radius: 8px;
        border: none;
        font-size: 1.1rem;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 6px 12px rgba(0,0,0,0.2);
    }
</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_model():
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

model = load_model()

# =========================
# UI HEADER
# =========================
st.set_page_config(
    page_title="TB Detection System",
    page_icon="🫁",
    layout="centered"
)

st.markdown("""
<div class="header-container">
    <div class="header-title">🫁 Tuberculosis Detection</div>
    <div class="header-subtitle">AI-Powered Chest X-Ray Analysis System</div>
</div>
""", unsafe_allow_html=True)

# =========================
# INFORMATION SECTION
# =========================
with st.expander("ℹ️ About This Tool", expanded=False):
    st.markdown("""
    This tool uses an **EfficientNet deep learning model** to analyze chest X-ray images 
    and detect potential signs of tuberculosis.
    
    **How to use:**
    1. Upload a chest X-ray image (JPG, JPEG, or PNG format)
    2. Wait for the AI to analyze the image
    3. Review the prediction and confidence score
    
    **Important:** This tool is for educational/screening purposes only and should not 
    replace professional medical diagnosis. Always consult with a healthcare provider 
    for proper medical advice.
    """)

# =========================
# IMAGE UPLOAD
# =========================
st.markdown("### 📤 Upload Chest X-Ray Image")
uploaded_file = st.file_uploader(
    "Choose an image file",
    type=["jpg", "jpeg", "png"],
    help="Upload a clear chest X-ray image for analysis"
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
# PREDICTION AND RESULTS
# =========================
if uploaded_file is not None:
    # Display uploaded image
    image = Image.open(uploaded_file).convert("RGB")
    
    st.markdown("### 📸 Uploaded Image")
    col1, col2, col3 = st.columns([1, 3, 1])
    with col2:
        st.image(image, use_container_width=True)
    
    st.markdown("---")
    
    # Process and predict
    processed_image = preprocess_image(image)
    
    with st.spinner("🔬 Analyzing X-ray image... Please wait..."):
        pred = model.predict(processed_image, verbose=0)[0][0]
    
    # Display results
    st.markdown("### 📊 Analysis Results")
    
    if pred >= 0.5:
        label = "Tuberculosis Detected"
        confidence = pred * 100
        
        st.markdown(f"""
        <div class="result-card result-positive">
            <div class="result-title">⚠️ {label}</div>
            <div class="confidence-value">{confidence:.1f}% Confidence</div>
            <p style="font-size: 1.1rem;">The model has detected signs consistent with tuberculosis.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="warning-box">
            <strong>⚕️ Next Steps:</strong><br>
            • Consult a healthcare professional immediately<br>
            • Additional tests may be required for confirmation<br>
            • Early detection improves treatment outcomes
        </div>
        """, unsafe_allow_html=True)
        
    else:
        label = "Normal"
        confidence = (1 - pred) * 100
        
        st.markdown(f"""
        <div class="result-card result-negative">
            <div class="result-title">✅ {label}</div>
            <div class="confidence-value">{confidence:.1f}% Confidence</div>
            <p style="font-size: 1.1rem;">No signs of tuberculosis detected in this X-ray.</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("""
        <div class="info-box">
            <strong>💡 Important Note:</strong><br>
            A normal result does not guarantee absence of disease. If you have symptoms 
            or concerns, please consult with a healthcare provider for proper evaluation.
        </div>
        """, unsafe_allow_html=True)
    
    # Confidence bar
    st.markdown("### 📈 Confidence Level")
    st.progress(float(confidence / 100))
    
    # Technical details
    with st.expander("🔍 Technical Details"):
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Raw Prediction Score", f"{pred:.4f}")
            st.metric("Image Size", f"{IMG_SIZE[0]}x{IMG_SIZE[1]}")
        with col2:
            st.metric("Model Type", "EfficientNet")
            st.metric("Classification", label)
    
    # Option to analyze another image
    st.markdown("---")
    if st.button("🔄 Analyze Another Image"):
        st.rerun()

else:
    # Show placeholder when no image is uploaded
    st.markdown("""
    <div class="upload-section">
        <h3>👆 Please upload a chest X-ray image to begin analysis</h3>
        <p>Supported formats: JPG, JPEG, PNG</p>
    </div>
    """, unsafe_allow_html=True)

# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 2rem 0;">
    <small>
    This AI diagnostic tool is for educational and screening purposes only.<br>
    Always seek professional medical advice for health concerns.
    </small>
</div>
""", unsafe_allow_html=True)