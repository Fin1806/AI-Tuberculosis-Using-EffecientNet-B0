import streamlit as st
import tensorflow as tf
import numpy as np
import cv2
import matplotlib.cm as cm
from PIL import Image
from tensorflow.keras.applications.efficientnet import preprocess_input

# --- 1. CONFIGURATION & SETUP ---
st.set_page_config(page_title="TB Detector AI + Grad-CAM", page_icon="🫁", layout="wide")

# Disable scientific notation for clarity
np.set_printoptions(suppress=True)

# Define Image Size & Layer
IMG_SIZE = (224, 224)
# 'top_activation' is the standard last convolutional layer in EfficientNetB0
# If you encounter an error, try changing this to 'block7a_project_conv' or 'top_conv'
LAST_CONV_LAYER = 'top_activation' 

# --- 2. MODEL LOADING ---
@st.cache_resource
def load_tb_model():
    """
    Loads the trained model. Uses st.cache_resource to avoid reloading
    the model every time the user interacts with the app.
    """
    try:
        # Load the 'v6 normal' model as requested
        model = tf.keras.models.load_model('v6_efficientnetb0_tb_best.keras')
        return model
    except Exception as e:
        st.error(f"Error loading model: {e}")
        return None

model = load_tb_model()

# --- 3. PREPROCESSING FUNCTION ---
def process_image(image_file):
    """
    1. Resizes image to 224x224.
    2. Applies Custom Masking (Top Left/Right corners).
    3. Applies EfficientNet preprocessing.
    """
    # Convert to RGB and Resize
    img = Image.open(image_file).convert('RGB')
    img = img.resize(IMG_SIZE)
    img_array = np.array(img).astype(np.float32)

    # --- CUSTOM MASKING ---
    # Create a writable copy to avoid read-only errors
    img_array_masked = img_array.copy()
    height, width, _ = img_array_masked.shape
    
    # Define mask size (25% of image)
    mask_h = int(height * 0.25) 
    mask_w = int(width * 0.25)
    
    # Masking Top Left Corner
    img_array_masked[0:mask_h, 0:mask_w, :] = 0.0
    # Masking Top Right Corner
    img_array_masked[0:mask_h, width-mask_w:width, :] = 0.0
    
    # EfficientNet Preprocessing
    preprocessed_img = preprocess_input(img_array_masked)
    # Add batch dimension (1, 224, 224, 3)
    img_batch = np.expand_dims(preprocessed_img, axis=0)
    
    return img_batch, img_array_masked.astype(np.uint8)

# --- 4. GRAD-CAM FUNCTIONS ---
def make_gradcam_heatmap(img_array, model, last_conv_layer_name, pred_index=None):
    """
    Generates the Grad-CAM heatmap for explainability.
    """
    # Create a model that maps the input image to the activations of the last conv layer
    # and the output predictions
    grad_model = tf.keras.models.Model(
        [model.inputs], [model.get_layer(last_conv_layer_name).output, model.output]
    )

    # Record operations for automatic differentiation
    with tf.GradientTape() as tape:
        last_conv_layer_output, preds = grad_model(img_array)
        if pred_index is None:
            pred_index = tf.argmax(preds[0])
        class_channel = preds[:, pred_index]

    # This is the gradient of the output neuron (top predicted or chosen)
    # with regard to the output feature map of the last conv layer
    grads = tape.gradient(class_channel, last_conv_layer_output)
    
    # Vector of weights: mean intensity of the gradient over a specific feature map channel
    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))

    # Multiply each channel in the feature map array by "how important this channel is"
    last_conv_layer_output = last_conv_layer_output[0]
    heatmap = last_conv_layer_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)

    # Normalize the heatmap between 0 & 1
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    return heatmap.numpy()

def overlay_heatmap(img_display, heatmap, alpha=0.4):
    """
    Overlays the heatmap onto the original image.
    """
    # Resize heatmap to match the original image size
    heatmap_resized = cv2.resize(heatmap, (img_display.shape[1], img_display.shape[0]))
    
    # Rescale heatmap to a range 0-255
    heatmap_uint8 = np.uint8(255 * heatmap_resized)
    
    # Use jet colormap to colorize heatmap
    heatmap_color = cv2.applyColorMap(heatmap_uint8, cv2.COLORMAP_JET)
    heatmap_color = cv2.cvtColor(heatmap_color, cv2.COLOR_BGR2RGB)
    
    # Superimpose the heatmap on original image
    superimposed_img = cv2.addWeighted(img_display, 1 - alpha, heatmap_color, alpha, 0)
    
    return superimposed_img

# --- 5. STREAMLIT UI ---
st.title("🫁 TB Detector AI + Grad-CAM")
st.markdown("Detect Tuberculosis and visualize the infected lung areas using **Explainable AI (Grad-CAM)**.")

# Layout: Upload on the left, Results on the right
col_upload, col_result = st.columns([1, 2])

with col_upload:
    st.subheader("Input Image")
    uploaded_file = st.file_uploader("Upload Chest X-Ray (JPG/PNG)", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        st.image(uploaded_file, caption="Original Uploaded Image", use_container_width=True)

if uploaded_file is not None and model is not None:
    # Process the image
    processed_input, display_img = process_image(uploaded_file)
    
    # Make Prediction
    prediction = model.predict(processed_input)
    probability = prediction[0][0]
    
    # Threshold logic (0.5 is standard)
    threshold = 0.5
    if probability > threshold:
        label = "TUBERCULOSIS"
        confidence = probability * 100
        color = "red"
    else:
        label = "NORMAL"
        confidence = (1 - probability) * 100
        color = "green"

    with col_result:
        st.subheader("Prediction Result")
        
        # Displaey Status with Color
        st.markdown(f"### Status: :{color}[{label}]")
        
        # Display Confidence Bar
        st.progress(int(probability * 100))
        st.caption(f"TB Probability Score: {probability:.4f} (Threshold: {threshold})")
        
        # --- GENERATE GRAD-CAM ---
        try:
            with st.spinner("Generating Grad-CAM Heatmap..."):
                # Generate Heatmap
                heatmap = make_gradcam_heatmap(processed_input, model, LAST_CONV_LAYER)
                
                # Overlay Heatmap
                gradcam_img = overlay_heatmap(display_img, heatmap)
                
                # Display Visualizations Side-by-Side
                st.subheader("AI Visualization (Grad-CAM)")
                st.write("The **Red/Yellow** areas indicate the regions most important to the AI's decision.")
                
                gc_col1, gc_col2 = st.columns(2)
                with gc_col1:
                    st.image(display_img, caption="Processed Input (Masked)", use_container_width=True)
                with gc_col2:
                    st.image(gradcam_img, caption="Grad-CAM Overlay", use_container_width=True)
                    
        except Exception as e:
            st.error(f"Failed to generate Grad-CAM: {e}")
            st.info("Tip: Ensure the layer name 'top_activation' matches your model architecture.")

else:
    if not uploaded_file:
        st.info("Please upload a Chest X-Ray image to begin analysis.")