import face_recognition
import pickle
import os

known_encodings = []
known_names = []

# Ganti dengan nama file fotomu
foto_anggota = {"Syakir": "photos/syakir.jpeg"}

for nama, file_foto in foto_anggota.items():
    if os.path.exists(file_foto):
        image = face_recognition.load_image_file(file_foto)
        encoding = face_recognition.face_encodings(image)[0]
        known_encodings.append(encoding)
        known_names.append(nama)
        print(f"Vektor wajah {nama} berhasil diekstrak!")

# Simpan ke dalam file biner
with open("encodings.pkl", "wb") as f:
    pickle.dump({"encodings": known_encodings, "names": known_names}, f)