import cv2
from datetime import datetime
from ultralytics import YOLO

class MOCAEngine:
    def __init__(self):
        # Inisialisasi model YOLO aktif
        self.model = YOLO("runs/detect/train/weights/best.engine", task="detect")
        self.classcon = ['no head cap', 'no mark', 'lab coat', 'no goggle']

    def process_fsm(self, detected_classes):
        """Mengevaluasi K3 secara langsung berdasarkan daftar objek dari YOLO"""
        waktu = datetime.now().strftime("%H:%M")
        
        if 'no head cap' in detected_classes or 'no goggle' in detected_classes:
            return "#CF2C30", "Akses Ditolak", "Atribut yang digunakan tidak lengkap", waktu
        elif 'lab coat' in detected_classes:
            return "#006C49", "Akses Diterima", "Atribut lengkap, pengguna dapat akses", waktu
        else:
            return "#EDCE23", "Peringatan", "Status atribut tidak valid", waktu

    def draw_annotations(self, frame, results):
        """Menggambar kotak manual dan mengumpulkan nama kelas objek (sangat ringan)"""
        detected_classes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_id = int(box.cls[0])
            class_name = results[0].names[class_id]
            detected_classes.append(class_name)
            
            # Gambar visual manual dengan OpenCV
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return frame, detected_classes