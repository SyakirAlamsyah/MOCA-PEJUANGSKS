import cv2
import pickle
import time 
from datetime import datetime, date
import face_recognition
from ultralytics import YOLO

class MOCAEngine:
    def __init__(self):
        self.model = YOLO("models/model.engine", task="detect")
        
        # PARAMETER AKTIF (Contoh: 0=person, 1=lab coat, 2=goggle)
        # Sistem akan otomatis menganggap ID selain 'person' sebagai APD yang wajib dipakai
        self.active_class_ids = [0, 1, 2] 
        
        with open("encodings.pkl", "rb") as f:
            data = pickle.load(f)
            self.known_encodings = data["encodings"]
            self.known_names = data["names"]
            
        self.present_today = set()
        self.current_date = date.today()
        
        # Variabel Memori FSM
        self.last_face_time = 0      
        self.tolerance_delay = 5     
        self.is_checking_apd = False 
        self.fail_count = 0          

    def check_attendance(self, frame):
        if date.today() != self.current_date:
            self.present_today.clear()
            self.current_date = date.today()

        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
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

    def get_detections(self, frame):
        results = self.model.predict(source=frame, show=False, verbose=False)
        detected_classes = []
        
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            if class_id not in self.active_class_ids:
                continue
                
            class_name = results[0].names[class_id]
            confidence = float(box.conf[0]) * 100
            detected_classes.append(class_name)
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label_text = f"{class_name} {confidence:.1f}%"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return frame, detected_classes

    def process_fsm(self, detected_classes, recognized_names):
        waktu = datetime.now().strftime("%H:%M")
        current_time = time.time()
        
        nama = recognized_names[0] if recognized_names else None
        is_member = nama and nama != "Orang Asing"

        if not nama:
            self.is_checking_apd = False
            self.fail_count = 0
            return "#EDCE23", "Menunggu", "Tidak ada orang terdeteksi", waktu

        # --- FASE 1: VERIFIKASI WAJAH & JEDA ---
        if not self.is_checking_apd:
            if current_time - self.last_face_time > self.tolerance_delay:
                self.last_face_time = current_time
            
            if current_time - self.last_face_time < self.tolerance_delay:
                sisa_waktu = int(self.tolerance_delay - (current_time - self.last_face_time))
                if is_member:
                    return "#006C49", "Dikenali", f"Terdeteksi anggota lab: {nama}. Siapkan APD anda ({sisa_waktu}s)", waktu
                else:
                    return "#EDCE23", "Peringatan", f"Terdeteksi bukan anggota. Siapkan APD anda ({sisa_waktu}s)", waktu
            
            self.is_checking_apd = True

        # --- FASE 2: EVALUASI APD DINAMIS ---
        # 1. Ekstrak daftar APD yang sedang diwajibkan (semua ID aktif selain 'person')
        active_labels = [self.model.names[i] for i in self.active_class_ids]
        required_apds = [label for label in active_labels if label.lower() != 'person']
        
        total_wajib = len(required_apds)
        apd_terdeteksi = sum([1 for apd in required_apds if apd in detected_classes])

        if total_wajib == 0:
            status = ("#EDCE23", "Menunggu", "Tidak ada parameter APD yang diaktifkan")
            
        elif apd_terdeteksi == 0:
            status = ("#CF2C30", "Akses Ditolak", "Tidak menggunakan APD sama sekali")
            
        elif apd_terdeteksi < total_wajib:
            if not is_member:
                status = ("#EDCE23", "Peringatan", "Atribut tidak lengkap")
            else:
                status = ("#EDCE23", "Peringatan", f"Atributnya tidak lengkap, {nama} perlu izin")
                
        elif apd_terdeteksi == total_wajib:
            if not is_member:
                status = ("#EDCE23", "Peringatan", "Seseorang minta akses lab, perlu izin")
            else:
                self.is_checking_apd = False 
                return "#006C49", "Akses Diterima", f"Atribut lengkap, {nama} dapat akses", waktu

        # --- FASE 3: LOOP TOLERANSI BERULANG ---
        self.fail_count += 1
        if self.fail_count > 2:
            self.fail_count = 0
            self.is_checking_apd = False 
            
        return status[0], status[1], status[2], waktu