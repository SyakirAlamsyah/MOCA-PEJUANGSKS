import cv2
import pickle
import time 
from datetime import datetime, date
import face_recognition
from ultralytics import YOLO

class MOCAEngine:
    def __init__(self):
        self.model = YOLO("models/model3.engine", task="detect")
        
        self.active_positive_ids = [1, 3] 
        self.active_negative_ids = [4, 6] 
        
        with open("encodings.pkl", "rb") as f:
            data = pickle.load(f)
            self.known_encodings = data["encodings"]
            self.known_names = data["names"]
            
        self.present_today = set()
        self.current_date = date.today()
        
        # Memori Penahan Wajah
        self.active_name = None
        self.last_seen_time = 0
        
        # Memori FSM 3 Fase
        self.fsm_stage = 0                # 0 = Kosong, 1 = Sapaan, 2 = Timer 10s
        self.last_apd_combination = None  
        self.stage_timer_start = 0        # Timer universal

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
        
        # PERBAIKAN: Hapus referensi '+ self.person_id'
        allowed_ids = self.active_positive_ids + self.active_negative_ids
        
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            
            if class_id not in allowed_ids:
                continue
                
            class_name = results[0].names[class_id]
            confidence = float(box.conf[0]) * 100
            detected_classes.append(class_name)
            
            # Warnai kotak merah untuk negatif, hijau untuk positif
            box_color = (0, 0, 255) if class_id in self.active_negative_ids else (0, 255, 0)
            
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), box_color, 2)
            label_text = f"{class_name} {confidence:.1f}%"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, box_color, 2)
            
        return frame, detected_classes

    def process_fsm(self, detected_classes, recognized_names):
        waktu = datetime.now().strftime("%H:%M")
        current_time = time.time()
        
        # 1. UPDATE MEMORI WAJAH 
        if len(recognized_names) > 0:
            if self.active_name != recognized_names[0]:
                self.active_name = recognized_names[0]
                self.fsm_stage = 1 
                self.stage_timer_start = current_time
                
            self.last_seen_time = current_time

        wajah_hilang = (current_time - self.last_seen_time) > 4

        # 2. RESET ALAMI JIKA KAMERA KOSONG
        if wajah_hilang:
            self.active_name = None
            self.fsm_stage = 0
            self.last_apd_combination = None
            return "#EDCE23", "Menunggu", "Tidak ada wajah terdeteksi", waktu
            
        nama = self.active_name
        is_member = nama != "Orang Asing"

        active_pos_labels = [self.model.names[i] for i in self.active_positive_ids]
        active_neg_labels = [self.model.names[i] for i in self.active_negative_ids]
        
        total_wajib = len(active_pos_labels)
        total_negatif = len(active_neg_labels)
        
        apd_terdeteksi = sum([1 for apd in active_pos_labels if apd in detected_classes])
        pelanggaran_terdeteksi = sum([1 for neg in active_neg_labels if neg in detected_classes])
        
        current_apd_combination = (apd_terdeteksi, pelanggaran_terdeteksi)

        # --- FASE 1: SAPAAN (Ditahan 2.5 Detik untuk UI Dashboard) ---
        if self.fsm_stage == 1:
            if current_time - self.stage_timer_start < 2.5:
                pesan = f"Terdeteksi anggota lab: {nama}. Siapkan APD anda" if is_member else "Terdeteksi bukan anggota. Siapkan APD anda"
                warna = "#006C49" if is_member else "#EDCE23"
                status_title = "Dikenali" if is_member else "Peringatan"
                return warna, status_title, pesan, waktu
            else:
                self.fsm_stage = 2
                self.last_apd_combination = current_apd_combination
                self.stage_timer_start = current_time 

        # --- FASE 2: EVALUASI APD & TIMER 10 DETIK ---
        if self.fsm_stage == 2:
            if current_apd_combination != self.last_apd_combination:
                self.last_apd_combination = current_apd_combination
                self.stage_timer_start = current_time
            
            sisa_waktu = 10 - int(current_time - self.stage_timer_start)

            # 1. Jika APD Lengkap -> Akses Diterima & Jeda 10 Detik
            if apd_terdeteksi == total_wajib and total_wajib > 0 and is_member:
                status_pesan = f"Atribut lengkap, {nama} dapat akses"
                
                # --- TAMBAHAN BARU: Delay 10 Detik ---
                time.sleep(10)
                
                self.fsm_stage = 0 
                self.active_name = None 
                self.last_seen_time = 0 
                return "#006C49", "Akses Diterima", status_pesan, waktu
                
            # 2. Jika Melanggar/Tidak Lengkap -> Tampilkan Hitung Mundur
            if total_negatif > 0 and pelanggaran_terdeteksi == total_negatif:
                status = ("#CF2C30", "Akses Ditolak", f"Tidak menggunakan APD ({sisa_waktu}s)")
            elif apd_terdeteksi == total_wajib and total_wajib > 0 and not is_member:
                status = ("#EDCE23", "Peringatan", f"Seseorang minta akses lab, perlu izin ({sisa_waktu}s)")
            else:
                pesan_peringatan = f"Atributnya tidak lengkap, {nama} perlu izin ({sisa_waktu}s)" if is_member else f"Atribut tidak lengkap ({sisa_waktu}s)"
                status = ("#EDCE23", "Peringatan", pesan_peringatan)

            # 3. Jika timer 10 detik habis tanpa kelengkapan APD -> Reset
            if sisa_waktu <= 0:
                self.fsm_stage = 0         
                self.active_name = None    
                self.last_seen_time = 0    
                return "#EDCE23", "Menunggu", "Waktu habis, memuat ulang...", waktu

            return status[0], status[1], status[2], waktu