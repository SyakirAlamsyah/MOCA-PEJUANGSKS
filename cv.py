import cv2
import streamlit as st
from ultralytics import YOLO

st.title("Smart Dashboard MOCA: Monitoring K3")
FRAME_WINDOW = st.image([])

model = YOLO("runs/detect/train/weights/best.engine", task="detect")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

classcon = ['no head cap', 'no mark', 'lab coat', 'no goggle']

while True:
    ret, frame = cap.read()
    if not ret: break
    
    results = model.predict(source=frame, show=False, verbose=False)
    
    for box in results[0].boxes:
        class_name = results[0].names[int(box.cls[0])]
        if class_name in classcon: 
            st.warning(f"Deteksi FSM: BUZZER MENYALA - {class_name}")

    # Konversi format warna OpenCV (BGR) ke Web (RGB)
    annotated_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(annotated_frame)