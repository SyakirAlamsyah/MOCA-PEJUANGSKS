import cv2
import pickle
from datetime import datetime, date
import face_recognition
from ultralytics import YOLO

class MOCAEngine:
    def __init__(self):
        self.model = YOLO("models/model.engine", task="detect")
        
        # Load database wajah dari file .pkl
        with open("encodings.pkl", "rb") as f:
            data = pickle.load(f)   
            self.known_encodings = data["encodings"]
            self.known_names = data["names"]
            
        self.present_today = set()
        self.current_date = date.today()

    def check_attendance(self, frame):
        # Reset otomatis jika berganti hari
        if date.today() != self.current_date:
            self.present_today.clear()
            self.current_date = date.today()

        # OPTIMASI 1: Susutkan frame menjadi 1/4 ukuran asli
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Konversi warna pada frame yang sudah disusutkan
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Eksekusi deteksi pada frame kecil (jauh lebih ringan untuk CPU)
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)

        recognized_names = []
        for face_encoding in face_encodings:
            matches = face_recognition.compare_faces(self.known_encodings, face_encoding, tolerance=0.5)
            if True in matches:
                name = self.known_names[matches.index(True)]
                self.present_today.add(name)
                recognized_names.append(name)
            else:
                recognized_names.append("Orang Asing")
                
        return recognized_names, len(self.present_today)

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