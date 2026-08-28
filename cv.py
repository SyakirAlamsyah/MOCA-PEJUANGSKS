import cv2
import streamlit as st
from ultralytics import YOLO

st.title("Smart Dashboard MOCA: Monitoring K3")
FRAME_WINDOW = st.image([])

model = YOLO("runs/detect/train/weights/best.engine", task="detect")
cap = cv2.VideoCapture(0, cv2.CAP_V4L2)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 320)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 240)

classcon = ['no head cap', 'no mark', 'lab coat', 'no goggle']

frame_count = 0
skip_interval = 3 

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame_count += 1
    
    # Lewati proses deteksi dan UI jika bukan kelipatan frame ke-3
    if frame_count % skip_interval != 0:
        continue
    
    results = model.predict(source=frame, show=False, verbose=False)
    
    for box in results[0].boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        class_id = int(box.cls[0])
        class_name = results[0].names[class_id]
        
        if class_name in classcon: 
            st.warning(f"Deteksi FSM: BUZZER MENYALA - {class_name}")
            
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    frame_resized = cv2.resize(frame, (320, 240))
    annotated_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2RGB)
    
    FRAME_WINDOW.image(annotated_frame)