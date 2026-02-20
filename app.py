import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image

# -----------------------------
# Page Configuration
# -----------------------------
st.set_page_config(
    page_title="AI-Based Crop Disease Detection",
    page_icon="🌿",
    layout="centered"
)

st.title("🌱 AI-Based Crop Disease Detection")
st.write("Upload a tomato leaf image to detect disease.")

# -----------------------------
# Sidebar Info
# -----------------------------
with st.sidebar:
    st.header("📌 Project Information")
    st.write("""
    This application uses **MobileNetV2 (Transfer Learning)**  
    trained on a Tomato Leaf Disease dataset.

    **Classes Detected:**
    • Early Blight  
    • Late Blight  
    • Healthy  

    Built using:
    • TensorFlow  
    • Streamlit  
    """)

# -----------------------------
# Load Model (Cached)
# -----------------------------
@st.cache_resource
def load_model():
    return tf.keras.models.load_model("model/tomato_disease_model.keras")

model = load_model()

# -----------------------------
# Class Labels
# -----------------------------
class_names = {
    0: "Early Blight",
    1: "Late Blight",
    2: "Healthy"
}

# -----------------------------
# File Upload
# -----------------------------
uploaded_file = st.file_uploader(
    "Choose an image...",
    type=["jpg", "jpeg", "png"]
)

if uploaded_file is not None:

    image = Image.open(uploaded_file)
    st.image(image, caption="Uploaded Image", use_container_width=True)

    # -----------------------------
    # Preprocess Image
    # -----------------------------
    img = image.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # -----------------------------
    # Prediction with Spinner
    # -----------------------------
    with st.spinner("🔍 Analyzing leaf image..."):
        predictions = model.predict(img_array)

    predicted_index = np.argmax(predictions)
    predicted_class = class_names[predicted_index]
    confidence = float(np.max(predictions)) * 100

    # -----------------------------
    # Display Prediction Result
    # -----------------------------
    st.markdown("## 🌿 Prediction Result")

    if predicted_class == "Healthy":
        st.success(f"Plant Status: {predicted_class}")
    else:
        st.error(f"Disease Detected: {predicted_class}")

    st.write(f"Confidence: {confidence:.2f}%")

    # -----------------------------
    # Display Class Probabilities
    # -----------------------------
    st.markdown("### 📊 Class Probabilities")

    for i, prob in enumerate(predictions[0]):
        st.write(f"{class_names[i]}: {prob*100:.2f}%")
        st.progress(float(prob))