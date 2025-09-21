import streamlit as st
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import numpy as np
from PIL import Image
import gdown
import os

st.title("🧠 Brain Tumor Detection (Binary)")

# Download model from Google Drive if not exists
model_url = 'https://drive.google.com/uc?id=1pgu7O2OqH-QWxVBjb6KvmNu1F1UbGqKf'  # Replace with your ID
model_path = 'model_binary.keras'

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Load model
model = load_model(model_path)

# Upload image
uploaded_file = st.file_uploader("Upload an MRI image", type=["jpg","jpeg","png"])

if uploaded_file is not None:
    img = Image.open(uploaded_file)
    st.image(img, caption='Uploaded MRI', use_column_width=True)

    # Preprocess image
    img = img.resize((224, 224))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    # Predict
    pred = model.predict(img_array)[0][0]
    if pred > 0.5:
        result = "Tumor Detected"
    else:
        result = "No Tumor



