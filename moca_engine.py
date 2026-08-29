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
        
        # --- MEMORI FSM BARU ---
        self.active_name = None
        self.last_seen_time = 0
        
        # Pelacak Perubahan APD
        self.last_apd_state = tuple()
        self.last_apd_change_time = 0
        self.tolerance_delay = 10

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
        
        # 1. UPDATE MEMORI WAJAH & RESET TIMER JIKA ORANG BARU
        if len(recognized_names) > 0:
            nama_terdeteksi = recognized_names[0]
            if self.active_name != nama_terdeteksi:
                # Wajah baru terkunci! Mulai hitung mundur 10 detik dari sekarang
                self.active_name = nama_terdeteksi
                self.last_apd_change_time = current_time
                self.last_apd_state = tuple()
            self.last_seen_time = current_time

        # 2. CEK GRACE PERIOD (Toleransi jika wajah hilang sesaat)
        wajah_hilang = (current_time - self.last_seen_time) > 4

        # Jika wajah benar-benar pergi ATAU belum ada wajah sama sekali
        if wajah_hilang or not self.active_name:
            self.active_name = None
            self.last_apd_change_time = 0
            return "#EDCE23", "Menunggu", "Tidak ada wajah terdeteksi", waktu
            
        nama = self.active_name
        is_member = nama != "Orang Asing"

        # 3. EVALUASI APD (LANGSUNG DIEKSEKUSI)
        active_pos_labels = [self.model.names[i] for i in self.active_positive_ids]
        active_neg_labels = [self.model.names[i] for i in self.active_negative_ids]
        
        total_wajib = len(active_pos_labels)
        total_negatif = len(active_neg_labels)
        
        # Pilah APD yang terdeteksi
        apd_positif_terdeteksi = [apd for apd in active_pos_labels if apd in detected_classes]
        apd_negatif_terdeteksi = [neg for neg in active_neg_labels if neg in detected_classes]
        
        # 4. CEK PERUBAHAN APD UNTUK MERESET WAKTU
        # Jadikan tuple agar bisa dibandingkan (contoh: dari pakai kacamata lalu pakai jas lab)
        current_apd_state = tuple(sorted(apd_positif_terdeteksi + apd_negatif_terdeteksi))
        
        if current_apd_state != self.last_apd_state:
            # Jika ada APD yang dipakai/dilepas, beri tambahan waktu 10 detik lagi
            self.last_apd_state = current_apd_state
            self.last_apd_change_time = current_time
            
        # 5. CEK TIMEOUT (Jika 10 detik diam tanpa progres APD)
        if (current_time - self.last_apd_change_time) > self.tolerance_delay:
            self.active_name = None
            self.last_apd_state = tuple()
            self.last_apd_change_time = 0
            return "#EDCE23", "Menunggu", "Waktu habis, silakan ulangi deteksi wajah", waktu

        # 6. PENENTUAN STATUS FSM
        jumlah_pos = len(apd_positif_terdeteksi)
        jumlah_neg = len(apd_negatif_terdeteksi)

        if total_negatif > 0 and jumlah_neg == total_negatif:
            status = ("#CF2C30", "Akses Ditolak", "Tidak menggunakan APD")
            
        elif jumlah_pos == total_wajib and total_wajib > 0:
            if not is_member:
                status = ("#EDCE23", "Peringatan", "Seseorang minta akses lab, perlu izin")
            else:
                # Lengkap! Reset sistem agar siap menerima orang berikutnya
                self.active_name = None
                self.last_apd_state = tuple()
                self.last_apd_change_time = 0
                return "#006C49", "Akses Diterima", f"Atribut lengkap, {nama} dapat akses", waktu
                
        else:
            if not is_member:
                status = ("#EDCE23", "Peringatan", "Atribut tidak lengkap")
            else:
                status = ("#EDCE23", "Peringatan", f"Atributnya tidak lengkap, {nama} perlu izin")
            
        return status[0], status[1], status[2], waktu