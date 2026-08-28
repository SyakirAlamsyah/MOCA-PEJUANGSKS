import cv2
import streamlit as st
from ultralytics import YOLO

st.title("Smart Dashboard MOCA: Monitoring K3")
FRAME_WINDOW = st.image([])

# Eksekusi model ringan dengan TensorRT
model = YOLO("runs/detect/train/weights/best.engine", task="detect")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

# Set resolusi awal pembacaan kamera
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

classcon = ['no head cap', 'no mark', 'lab coat', 'no goggle']

while True:
    ret, frame = cap.read()
    if not ret: break
    
    results = model.predict(source=frame, show=False, verbose=False)
    
    # 1. OPTIMASI: Gambar Bounding Box secara Manual tes
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        
        # Evaluasi Logika FSM
        if class_name in classcon: 
            st.warning(f"Deteksi FSM: BUZZER MENYALA - {class_name}")
            
        # Gambar kotak dengan fungsi murni OpenCV (Beban CPU sangat ringan)
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    # 2. OPTIMASI: Turunkan Resolusi Sebelum Dirender UI Web
    frame_resized = cv2.resize(frame, (320, 240))
    annotated_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    # Kirim ke Dashboard Streamlit
    FRAME_WINDOW.image(annotated_frame)