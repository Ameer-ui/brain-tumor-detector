import gdown
import os

# Download model from Google Drive if not exists
model_url = 'https://drive.google.com/uc?id=1pgu7O2OqH-QWxVBjb6KvmNu1F1UbGqKf'  # Replace with your ID
model_path = 'model_binary.keras'

if not os.path.exists(model_path):
    gdown.download(model_url, model_path, quiet=False)

# Now load the model
model = load_model(model_path)
