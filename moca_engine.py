import cv2
import pickle
from datetime import datetime, date
import face_recognition
from ultralytics import YOLO

class MOCAEngine:
    def __init__(self):
        self.model = YOLO("runs/detect/train/weights/best.engine", task="detect")
        
        # PARAMETER FILTER DINAMIS: Masukkan ID kelas yang ingin diaktifkan
        # Contoh ID: 0 (person), 1 (lab coat), 2 (goggle), dst.
        self.active_class_ids = [3, 4, 5] 
        
        # Load database wajah 
        with open("encodings.pkl", "rb") as f:
            data = pickle.load(f)
            self.known_encodings = data["encodings"]
            self.known_names = data["names"]
            
        self.present_today = set()
        self.current_date = date.today()

    def check_attendance(self, frame):
        # ... (Logika face recognition tetap sama) ...
        pass

    def process_frame(self, frame, recognized_names):
        """Mengeksekusi YOLO dengan filter ID kelas dan meneruskannya ke FSM"""
        
        # 1. Eksekusi YOLO dengan parameter 'classes' aktif
        results = self.model.predict(
            source=frame, 
            classes=self.active_class_ids, 
            show=False, 
            verbose=False
        )
        
        detected_classes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = results[0].names[int(box.cls[0])]
            detected_classes.append(class_name)
            
            # Gambar visual manual dengan OpenCV
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(frame, class_name, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        # 2. Teruskan hasil filter ke logika FSM
        color, title, desc, waktu = self.process_fsm(detected_classes, recognized_names)
        
        return frame, color, title, desc, waktu
    def process_fsm(self, detected_classes, recognized_names):
        waktu = datetime.now().strftime("%H:%M")
        
        # Evaluasi Kelengkapan APD
        has_labcoat = 'lab coat' in detected_classes
        has_goggles = 'goggle' in detected_classes
        
        jumlah_apd = sum([has_labcoat, has_goggles])
        is_member = any(name != "Orang Asing" for name in recognized_names)
        nama = recognized_names[0] if recognized_names else "Orang Asing"

        # Aturan FSM Berdasarkan Rincian Baru
        if jumlah_apd == 0:
            return "#CF2C30", "Akses Ditolak", "Tidak menggunakan APD", waktu
            
        elif jumlah_apd == 1:
            if not is_member:
                return "#EDCE23", "Peringatan", "Atribut Tidak lengkap", waktu
            else:
                return "#EDCE23", "Peringatan", f"Atributnya tidak lengkap, {nama} perlu izin", waktu
                
        elif jumlah_apd == 2:
            if not is_member:
                return "#EDCE23", "Peringatan", "Seseorang minta akses lab, perlu izin", waktu
            else:
                return "#006C49", "Akses Diterima", f"Atribut lengkap, {nama} dapat akses", waktu
                
        return "#EDCE23", "Menunggu", "Menganalisis sistem...", waktu

    def draw_annotations(self, frame, results):
        detected_classes = []
        for box in results[0].boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            class_name = results[0].names[int(box.cls[0])]
            detected_classes.append(class_name)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        return frame, detected_classes