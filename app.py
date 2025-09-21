import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import gdown
import os

# -------------------------------
# Page Configuration
# -------------------------------
st.set_page_config(
    page_title="🧠 Brain Tumor Detection",
    page_icon="🧠",
    layout="centered",
    initial_sidebar_state="expanded"
)

# -------------------------------
# Custom CSS for Dark Theme & Styling
# -------------------------------
st.markdown("""
<style>
/* Background */
body {
    background-color: #121212;
    color: #ffffff;
}

/* Header */
h1 {
    color: #00bfff;
    text-align: center;
}

/* Description text */
p {
    color: #cccccc;
    text-align: center;
    font-size: 18px;
}

/* File uploader style */
.stFileUploader>div>div>div>input {
    border: 2px dashed #00bfff;
    padding: 10px;
    border-radius: 10px;
    background-color: #1e1e1e;
    color: white;
}

/* Prediction box */
.stSuccess>div>div {
    background-color: #00bfff50;
    border-radius: 10px;
    padding: 10px;
    font-weight: bold;
    font-size: 20px;
    text-align: center;
}

/* Image style */
.stImage>div>div>img {
    border-radius: 10px;
    box-shadow: 5px 5px 15px #000000;
}

/* Sidebar styling */
.css-1d391kg {
    background-color: #1e1e1e;
    color: #ffffff;
}

/* Center content */
.main > div {
    display: flex;
    flex-direction: column;
    align-items: center;
}
</style>
""", unsafe_allow_html=True)

# -------------------------------
# Sidebar Instructions
# -------------------------------
st.sidebar.title("🧠 Brain Tumor Detector")
st.sidebar.write("""
This AI tool detects **brain tumors** from MRI scans.
1. Upload an MRI image (jpg, jpeg, png).  
2. Wait for the AI prediction.  
3. See if a tumor is detected.
""")

# -------------------------------
# Header Section
# -------------------------------
st.markdown("""
<div style="background: linear-gradient(90deg, #1f77b4, #00bfff); padding: 20px; border-radius: 10px; text-align:center;">
<h1>🧠 Brain Tumor Detection</h1>
<p>Upload an MRI image and let the AI predict if a tumor is present</p>
</div>
""", unsafe_allow_html=True)

# -------------------------------
# Load Model from Google Drive
# -------------------------------
model_url = 'https://drive.google.com/uc?id=1pgu7O2OqH-QWxVBjb6KvmNu1F1UbGqKf'  # Replace with your ID
model_path = 'model_binary.keras'

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

model = load_model(model_path)

# -------------------------------
# Image Upload
# -------------------------------
uploaded_file = st.file_uploader("Upload MRI Image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded MRI', use_container_width=True)

    # Preprocess Image
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)[0][0]
    if pred > 0.5:
        result = "Tumor Detected"
    else:
        result = "No Tumor Detected"

    st.success(f"Prediction: {result} (Confidence: {pred:.2f})")
