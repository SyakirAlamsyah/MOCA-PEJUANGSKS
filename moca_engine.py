import cv2
import pickle
import time 
from datetime import datetime, date
import face_recognition
from ultralytics import YOLO
class MOCAEngine:
    def __init__(self):
        self.model = YOLO("models/model.engine", task="detect")
        self.active_class_ids = [1, 2, 4, 5, 6, 7, 9] 
        
        with open("encodings.pkl", "rb") as f:
            data = pickle.load(f)
            self.known_encodings = data["encodings"]
            self.known_names = data["names"]
            
        self.present_today = set()
        self.current_date = date.today()
        
        self.last_face_time = 0      
        self.tolerance_delay = 5     
        self.is_checking_apd = False 
        self.fail_count = 0           

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

    def process_frame(self, frame, recognized_names):
        """Mengeksekusi YOLO dengan filter ID kelas dan meneruskannya ke FSM"""
        
        # 1. Eksekusi YOLO dengan parameter 'classes' aktif
        results = self.model.predict(
            source=frame, 
            classes=self.active_class_ids, 
            show=False, 
            verbose=False,
            )
        
        detected_classes = []
        for box in results[0].boxes:
            class_name = results[0].names[int(box.cls[0])]
            detected_classes.append(class_name)

        annotated_frame = results[0].plot(labels=True, conf=True)

        color, title, desc, waktu = self.process_fsm(detected_classes, recognized_names)

        return annotated_frame, color, title, desc, waktu
    
    def process_fsm(self, detected_classes, recognized_names):
        waktu = datetime.now().strftime("%H:%M")
        current_time = time.time()
        
        # Pastikan ada orang di depan kamera
        nama = recognized_names[0] if recognized_names else None
        is_member = nama and nama != "Orang Asing"

        # --- LOGIKA KOSONG (IDLE) ---
        if not nama:
            self.is_checking_apd = False
            self.fail_count = 0
            return "#EDCE23", "Menunggu", "Tidak ada orang terdeteksi", waktu

        # --- FASE 1: VERIFIKASI WAJAH & JEDA TOLERANSI ---
        if not self.is_checking_apd:
            # Jika ini wajah baru, mulai perhitungan mundur
            if current_time - self.last_face_time > self.tolerance_delay:
                self.last_face_time = current_time
            
            # Cek apakah jeda 5 detik sudah terlewati
            if current_time - self.last_face_time < self.tolerance_delay:
                sisa_waktu = int(self.tolerance_delay - (current_time - self.last_face_time))
                if is_member:
                    return "#006C49", "Dikenali", f"Terdeteksi anggota lab: {nama}. Siapkan APD anda ({sisa_waktu}s)", waktu
                else:
                    return "#EDCE23", "Peringatan", f"Terdeteksi bukan anggota. Siapkan APD anda ({sisa_waktu}s)", waktu
            
            # Jeda selesai, aktifkan fase cek APD
            self.is_checking_apd = True

        # --- FASE 2: EVALUASI APD MUTLAK (Sesuai Rincian Baru) ---
        has_labcoat = 'lab coat' in detected_classes
        has_goggles = 'goggle' in detected_classes
        jumlah_apd = sum([has_labcoat, has_goggles])

        # Aturan 1: Ditolak Mutlak (Tidak ada APD sama sekali)
        if jumlah_apd == 0:
            status = ("#CF2C30", "Akses Ditolak", "Tidak menggunakan APD")
        
        # Aturan 2: Peringatan (APD Tidak Lengkap)
        elif jumlah_apd == 1:
            if not is_member:
                status = ("#EDCE23", "Peringatan", "Atribut tidak lengkap")
            else:
                status = ("#EDCE23", "Peringatan", f"Atributnya tidak lengkap, {nama} perlu izin")
                
        # Aturan 3: Peringatan Orang Asing (APD Lengkap tapi bukan anggota)
        elif jumlah_apd == 2 and not is_member:
            status = ("#EDCE23", "Peringatan", "Seseorang minta akses lab, perlu izin")
            
        # Aturan 4: Akses Diterima (APD Lengkap & Anggota)
        elif jumlah_apd == 2 and is_member:
            self.is_checking_apd = False # Reset sistem untuk orang berikutnya
            return "#006C49", "Akses Diterima", f"Atribut lengkap, {nama} dapat akses", waktu
            
        else:
             status = ("#EDCE23", "Menunggu", "Menganalisis...")

        # --- FASE 3: LOOP TOLERANSI BERULANG (Pengingat 2 kali) ---
        # Jika masuk ke blok ini, berarti APD gagal (Status 1, 2, atau 3)
        self.fail_count += 1
        
        # Tunggu 2 siklus peringatan (misal YOLO jalan per 3 frame, berarti ini sekitar 6 frame/1 detik kemudian)
        if self.fail_count > 2:
            self.fail_count = 0
            self.is_checking_apd = False # Kembalikan FSM ke awal (Verifikasi Wajah)
            
        return status[0], status[1], status[2], waktu

    def get_detections(self, frame):
        """Eksekusi YOLO, filter ID aktif, dan gambar label secara manual"""
        # Eksekusi prediksi mentah
        results = self.model.predict(source=frame, show=False, verbose=False)
        detected_classes = []
        
        for box in results[0].boxes:
            class_id = int(box.cls[0])
            
            # GERBANG FILTER: Jika ID objek tidak ada di list aktif, abaikan
            if class_id not in self.active_class_ids:
                continue
                
            class_name = results[0].names[class_id]
            confidence = float(box.conf[0]) * 100
            detected_classes.append(class_name)
            
            # 1. Gambar Bounding Box
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # 2. Gambar Teks Label dan Persentase Akurasi
            label_text = f"{class_name} {confidence:.1f}%"
            cv2.putText(frame, label_text, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
            
        return frame, detected_classes