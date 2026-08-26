import cv2
from ultralytics import YOLO

def gstreamer_pipeline(capture_width=1280, capture_height=720, display_width=1280, display_height=720, framerate=30, flip_method=0):
    return (
        "nvarguscamerasrc ! "
        "video/x-raw(memory:NVMM), "
        f"width=(int){capture_width}, height=(int){capture_height}, "
        f"framerate=(fraction){framerate}/1 ! "
        f"nvvidconv flip-method={flip_method} ! "
        f"video/x-raw, width=(int){display_width}, height=(int){display_height}, format=(string)BGRx ! "
        "videoconvert ! "
        "video/x-raw, format=(string)BGR ! appsink"
    )


# Path Modelnya
model = YOLO("runs/detect/train/weights/best.pt")

#  Setup Computer Vision (Kamera)
cap = cv2.VideoCapture(gstreamer_pipeline(flip_method=0), cv2.CAP_GSTREAMER)

#Setup Kondisi
classcon = ['no head cap', 'no mark', 'lab coat', 'no goggle']

if not cap.isOpened():
    print("Error: Tidak dapat membuka kamera. Pastikan kamera tidak dipakai aplikasi lain.")
    exit()

print("Kamera berhasil dibuka! Tekan tombol 'q' pada jendela gambar untuk keluar.")

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Gagal mengambil frame gambar dari kamera.")
        break

    # Jalankan YOLO pada frame yang sedang aktif
    results = model.predict(source=frame, show=False, verbose=False)



    # Set Kondisi Deteksi Objek
    for box in results[0].boxes:
            class_id = int(box.cls[0])
            
            class_name = results[0].names[class_id]

            #FSM nya kaya gni
            if class_name == classcon[0]: 
                print(f"Deteksi: BUZZER MENYALA {classcon[0]}")
            elif class_name == classcon[1]: 
                print(f"Deteksi: BUZZER MENYALA {classcon[1]}")
            elif class_name == classcon[2]: 
                print(f"Deteksi: BUZZER MENYALA {classcon[2]}")
            elif class_name == classcon[3]: 
                print(f"Deteksi: BUZZER MENYALA {classcon[3]}")

    annotated_frame = results[0].plot()

    cv2.imshow("Deteksi Atribut YOLO", annotated_frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Bersihkan memori setelah selesai
cap.release()
cv2.destroyAllWindows()
