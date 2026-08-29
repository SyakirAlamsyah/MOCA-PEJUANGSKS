import cv2
import pickle
import time 
from datetime import datetime, date
import face_recognition
from ultralytics import YOLO

class MOCAEngine:
    def __init__(self):
        self.model = YOLO("models/model3.engine", task="detect")
        
        # 1. PISAHKAN PARAMETER ID KELAS 
        self.active_positive_ids = [1, 3] 
        self.active_negative_ids = [4, 6] 
        
        with open("encodings.pkl", "rb") as f:
            data = pickle.load(f)
            self.known_encodings = data["encodings"]
            self.known_names = data["names"]
            
        self.present_today = set()
        self.current_date = date.today()
        
        self.last_face_time = 0      
        self.tolerance_delay = 10    # DURASI DIPERBARUI: Jeda 10 detik 
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
        
        # Kehadiran manusia dipicu oleh deteksi wajah
        ada_orang = len(recognized_names) > 0
        nama = recognized_names[0] if ada_orang else None

        # RESET ALAMI: Jika kamera tidak menangkap wajah sama sekali
        if not ada_orang:
            self.is_checking_apd = False
            self.last_face_time = 0 
            return "#EDCE23", "Menunggu", "Tidak ada wajah terdeteksi", waktu
            
        is_member = nama != "Orang Asing"

        # --- FASE 1: VERIFIKASI WAJAH & JEDA 10 DETIK ---
        if not self.is_checking_apd:
            if self.last_face_time == 0:
                self.last_face_time = current_time
            
            sisa_waktu = int(self.tolerance_delay - (current_time - self.last_face_time))
            
            # Tahan di fase ini selama waktu belum habis
            if sisa_waktu > 0:
                pesan = f"Terdeteksi anggota lab: {nama}. Siapkan APD anda ({sisa_waktu}s)" if is_member else f"Terdeteksi bukan anggota. Siapkan APD anda ({sisa_waktu}s)"
                warna = "#006C49" if is_member else "#EDCE23"
                status_title = "Dikenali" if is_member else "Peringatan"
                return warna, status_title, pesan, waktu
            
            # Waktu habis, maju permanen ke Fase 2
            self.is_checking_apd = True

        # --- FASE 2: EVALUASI POSITIF VS NEGATIF (TERKUNCI) ---
        active_pos_labels = [self.model.names[i] for i in self.active_positive_ids]
        active_neg_labels = [self.model.names[i] for i in self.active_negative_ids]
        
        total_wajib = len(active_pos_labels)
        total_negatif = len(active_neg_labels)
        
        apd_terdeteksi = sum([1 for apd in active_pos_labels if apd in detected_classes])
        pelanggaran_terdeteksi = sum([1 for neg in active_neg_labels if neg in detected_classes])

        if total_negatif > 0 and pelanggaran_terdeteksi == total_negatif:
            status = ("#CF2C30", "Akses Ditolak", "Tidak menggunakan APD")
            
        elif apd_terdeteksi == total_wajib and total_wajib > 0:
            if not is_member:
                status = ("#EDCE23", "Peringatan", "Seseorang minta akses lab, perlu izin")
            else:
                # Berhasil memakai APD lengkap, reset timer untuk orang berikutnya
                self.is_checking_apd = False 
                self.last_face_time = 0 
                return "#006C49", "Akses Diterima", f"Atribut lengkap, {nama} dapat akses", waktu
                
        else:
            if not is_member:
                status = ("#EDCE23", "Peringatan", "Atribut tidak lengkap")
            else:
                status = ("#EDCE23", "Peringatan", f"Atributnya tidak lengkap, {nama} perlu izin")

        # FASE 3 DIHAPUS. Sistem akan terus mengunci evaluasi APD di layar (Akses Ditolak/Peringatan)
        # sampai pengguna melengkapi APD (Akses Diterima) atau pergi (Kamera Kosong).
            
        return status[0], status[1], status[2], waktu