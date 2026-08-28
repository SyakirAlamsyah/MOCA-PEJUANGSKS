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
<<<<<<< HEAD
            class_id = int(box.cls[0])
            
            class_name = results[0].names[class_id]

            #FSM nya kaya gni
            if class_name == classcon[0]: 
                print("Deteksi: BUZZER MENYALA " + classcon[0])
            elif class_name == classcon[1]: 
                print("Deteksi: BUZZER MENYALA " + classcon[1])
            elif class_name == classcon[2]: 
                print("Deteksi: BUZZER MENYALA " + classcon[2])
            elif class_name == classcon[3]: 
                print("Deteksi: BUZZER MENYALA " + classcon[3])

#     annotated_frame = results[0].plot()

#     cv2.imshow("Deteksi Atribut YOLO", annotated_frame)

#     if cv2.waitKey(1) & 0xFF == ord('q'):
#         break

# # Bersihkan memori setelah selesai
# cap.release()
# cv2.destroyAllWindows()
=======
        class_name = results[0].names[int(box.cls[0])]
        if class_name in classcon: 
            st.warning(f"Deteksi FSM: BUZZER MENYALA - {class_name}")

    # Konversi format warna OpenCV (BGR) ke Web (RGB)
    annotated_frame = cv2.cvtColor(results[0].plot(), cv2.COLOR_BGR2RGB)
    FRAME_WINDOW.image(annotated_frame)
>>>>>>> 3d39928b802005e932e7691e16e5399668e1335c
