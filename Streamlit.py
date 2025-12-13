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
        background-color: #f7f9fc;
    }
    
    /* Header styling */
    .header-container {
        text-align: center;
        padding: 2rem 0 3rem 0;
        background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 10px 20px rgba(0,0,0,0.15);
        backdrop-filter: blur(5px);
    }
    
    .header-title {
        color: white;
        font-size: 3.2rem;
        font-weight: 800;
        margin-bottom: 0.5rem;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.4);
    }
    
    .header-subtitle {
        color: #e0f7fa;
        font-size: 1.1rem;
        font-weight: 400;
    }
    
    /* Upload section */
    .upload-section {
        background: #ffffff;
        padding: 3rem;
        border-radius: 12px;
        border: 3px dashed #bbdefb;
        margin: 2rem 0;
        text-align: center;
        box-shadow: 0 4px 15px rgba(0,0,0,0.05);
    }
    
    /* Text color fix for dark mode contrast */
    .upload-section h3 {
        color: #333333 !important;
    }
    .upload-section p {
        color: #6c757d !important;
    }

    /* Result cards - More distinct color coding */
    .result-card {
        padding: 2rem;
        border-radius: 12px;
        margin: 1.5rem 0;
        box-shadow: 0 6px 15px rgba(0,0,0,0.1);
        transition: transform 0.3s ease;
    }
    .result-card:hover {
        transform: translateY(-5px);
    }
    
    .result-positive {
        background: linear-gradient(135deg, #ff6b6b 0%, #ee5253 100%);
        color: white;
    }
    
    .result-negative {
        background: linear-gradient(135deg, #48c774 0%, #43a047 100%);
        color: white;
    }
    
    .result-title {
        font-size: 2.2rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
    }
    
    .confidence-value {
        font-size: 3rem;
        font-weight: 900;
        margin: 1rem 0;
    }
    
    /* Button styling */
    .stButton > button {
        width: 100%;
        background: linear-gradient(135deg, #1e90ff 0%, #5352ed 100%);
        color: white;
        font-weight: 700;
        padding: 0.75rem 2rem;
        border-radius: 10px;
        border: none;
        font-size: 1.1rem;
        transition: all 0.3s ease;
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #5352ed 0%, #1e90ff 100%);
        transform: translateY(-3px);
        box-shadow: 0 8px 15px rgba(0,0,0,0.3);
    }

    /* Streamlit Metric Styling */
    [data-testid="stMetricValue"] {
        font-size: 1.5rem;
        font-weight: 700;
    }
    [data-testid="stMetricLabel"] {
        font-weight: 500;
        font-size: 0.9rem;
    }

</style>
""", unsafe_allow_html=True)

# =========================
# LOAD MODEL (CACHE)
# =========================
@st.cache_resource
def load_ml_model():
    """Load the pre-trained EfficientNet model."""
    try:
        model = tf.keras.models.load_model(MODEL_PATH)
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_ml_model()

# =========================
# UI HEADER & CONFIG
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
with st.container():
    with st.expander("ℹ️ About This Tool & Disclaimer", expanded=False):
        st.markdown("""
        This tool uses an **EfficientNet deep learning model** to analyze chest X-ray images 
        and detect potential signs of tuberculosis (TB).
        
        ### ⚠️ Important Medical Disclaimer
        * **This tool is for educational/screening purposes only.**
        * **It is NOT a substitute for professional medical diagnosis, advice, or treatment.**
        * **Always consult with a qualified healthcare provider** for proper diagnosis, especially if you have symptoms or a positive result.
        """)

# =========================
# IMAGE PROCESSING FUNCTION
# =========================
def preprocess_image(img):
    """Resizes and preprocesses the image for the EfficientNet model."""
    try:
        img = np.array(img)
        img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR) 
        img = cv2.resize(img, IMG_SIZE)
        img = preprocess_input(img.astype("float32"))
        img = np.expand_dims(img, axis=0)
        return img
    except Exception as e:
        st.error(f"Error processing image: {e}")
        return None

# =========================
# IMAGE UPLOAD SECTION
# =========================
st.markdown("---")

with st.container():
    st.markdown("### 📤 Upload Chest X-Ray Image")
    uploaded_file = st.file_uploader(
        "Choose an image file (JPG, JPEG, or PNG)",
        type=["jpg", "jpeg", "png"],
        help="Upload a clear Posteroanterior (PA) or Anteroposterior (AP) chest X-ray image for analysis."
    )

# =========================
# PREDICTION AND RESULTS
# =========================
if uploaded_file is not None:
    if model is None:
        st.error("The model could not be loaded. Please check the model path.")
    else:
        # --- Display uploaded image ---
        image = Image.open(uploaded_file).convert("RGB")
        
        st.markdown("---")
        st.markdown("### 📸 Uploaded Image Preview")
        col_img, col_space = st.columns([1, 4])
        with col_img:
            st.image(image, caption='X-Ray Image', use_container_width=True)

        # --- Process and predict ---
        processed_image = preprocess_image(image)
        
        if processed_image is not None:
            # Use a button to trigger analysis
            if st.button("🚀 Run AI Analysis"):
                
                # Predict
                with st.spinner("🔬 Analyzing X-ray image... Please wait..."):
                    try:
                        pred = model.predict(processed_image, verbose=0)[0][0]
                    except Exception as e:
                        st.error(f"Prediction failed: {e}")
                        pred = None

                if pred is not None:
                    # --- Display results ---
                    st.markdown("---")
                    st.markdown("### 📊 Analysis Results")
                    
                    if pred >= 0.5:
                        # Positive Result
                        label = "Tuberculosis Detected"
                        confidence = pred * 100
                        
                        st.markdown(f"""
                        <div class="result-card result-positive">
                            <div class="result-title">🚨 {label}</div>
                            <div class="confidence-value">{confidence:.1f}% Confidence</div>
                            <p style="font-size: 1.1rem; font-weight: 500;">
                                The model has detected signs highly consistent with Tuberculosis.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.warning("""
                        **⚠️ Immediate Recommended Action:**
                        1.  **Consult a healthcare professional** (e.g., doctor or specialist) immediately.
                        2.  Share this X-ray and the AI's result with them.
                        3.  A definitive diagnosis requires additional clinical tests.
                        """)
                        
                    else:
                        # Negative Result
                        label = "Normal (No TB Signs Detected)"
                        confidence = (1 - pred) * 100
                        
                        st.markdown(f"""
                        <div class="result-card result-negative">
                            <div class="result-title">✨ {label}</div>
                            <div class="confidence-value">{confidence:.1f}% Confidence</div>
                            <p style="font-size: 1.1rem; font-weight: 500;">
                                The model indicates a high probability of a normal chest X-ray.
                            </p>
                        </div>
                        """, unsafe_allow_html=True)
                        
                        st.info("""
                        **✅ Important Note:**
                        * A 'Normal' result is reassuring but **does not exclude disease**.
                        * If symptoms persist (e.g., persistent cough, fever, weight loss), please seek medical consultation.
                        """)
                    
                    # --- Confidence bar and Technical Details ---
                    st.markdown("### 📈 Confidence Breakdown")
                    st.progress(float(confidence / 100), text=f"Model Certainty: {confidence:.1f}%")
                    
                    with st.expander("🔍 Technical Details of the Analysis"):
                        col1, col2, col3 = st.columns(3)
                        
                        col1.metric("Classification", label.split('(')[0].strip())
                        col2.metric("Image Size (HxW)", f"{IMG_SIZE[0]}x{IMG_SIZE[1]}")
                        col3.metric("Raw Score (TB Prob)", f"{pred:.4f}")
                        
                        st.caption("Raw score is the output of the final sigmoid layer, representing the probability of TB.")
                        
                    # Option to analyze another image
                    st.markdown("---")
                    if st.button("🔄 Analyze Another Image"):
                        st.rerun()

# --- Placeholder for No Upload ---
else:
    # Show a friendly, interactive prompt when no image is uploaded
    st.markdown("""
    <div class="upload-section">
        <h3>💡 Get Started</h3>
        <p>Please upload a clear chest X-ray image (e.g., in JPG or PNG format) using the browser button above to initiate the deep learning analysis.</p>
        <p style="margin-top: 1.5rem; font-style: italic; color: #6c757d !important;">
            (The AI model is based on EfficientNet architecture, pre-trained on medical imaging data.)
        </p>
    </div>
    """, unsafe_allow_html=True)
    
    # Optional: Suggest an example to try (text removed as requested)
    with st.expander("🖼️ Example X-Ray Image (TB Positive)", expanded=False):
        # Image place holder for illustrative purposes
        st.markdown("Use this space to display a sample positive X-ray image for testing.")
        
# =========================
# FOOTER
# =========================
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #666; padding: 1rem 0;">
    <small>
    Built with Streamlit and TensorFlow. AI is a screening aid, not a definitive diagnosis.
    </small>
</div>
""", unsafe_allow_html=True)