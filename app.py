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
