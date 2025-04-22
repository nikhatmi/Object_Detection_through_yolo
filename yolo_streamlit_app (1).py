import streamlit as st
from PIL import Image
from ultralytics import YOLO
import tempfile
import os
from pathlib import Path
st.set_page_config(page_title="YOLO Object Detection", layout="centered")
st.title("🔍 Object Detection using YOLOv8")
st.markdown("Upload an image and run YOLOv8 to see detected objects with bounding boxes.")

# Upload image
uploaded_file = st.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

# Load model
@st.cache_resource
def load_model():
    return YOLO("mymodel.pt")  # Change to your custom .pt model if needed

model = load_model()

# Confidence slider
conf_threshold = 0.1

if uploaded_file is not None:
    # Display uploaded image
    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Save image to a temporary file
    with tempfile.NamedTemporaryFile(delete=False, suffix=".jpg") as tmp_file:
        tmp_file.write(uploaded_file.read())
        tmp_img_path = tmp_file.name

    # Run YOLO prediction
    with st.spinner("Detecting objects..."):
        results = model.predict(source=tmp_img_path, conf=conf_threshold, save=True)
        result_path = Path(results[0].save_dir) / os.path.basename(tmp_img_path)

    # Display detected image
    st.image(str(result_path), caption="Detected Objects", use_column_width=True)
