import streamlit as st
import cv2
import torch
from PIL import Image
import numpy as np
from ultralytics import YOLO

# Load YOLOv8 model (pre-trained or custom)
model = YOLO("yolov8n.pt")  # You can replace with a fine-tuned model (e.g., with phones)

# Streamlit UI
st.title("Proctoring App")
st.markdown("Real-time face and phone detection using YOLO and webcam.")

# Start webcam
run = st.checkbox("Start Webcam")
FRAME_WINDOW = st.image([])

if run:
    cap = cv2.VideoCapture(0)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.warning("Failed to grab frame")
            break

        # Convert frame to RGB (OpenCV uses BGR)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # YOLO inference
        results = model(rgb_frame, verbose=False)[0]

        # Draw results
        annotated_frame = results.plot()

        # Extract detected classes
        detected_classes = [model.model.names[int(cls)] for cls in results.boxes.cls]

        # Display alerts
        if "person" not in detected_classes:
            st.warning("⚠️ No face detected!")
        if "cell phone" in detected_classes:
            st.error("📵 Phone detected!")

        # Show frame in Streamlit
        FRAME_WINDOW.image(annotated_frame)

else:
    st.info(" Click the checkbox to start webcam")

