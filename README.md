# MOCA (Prototype) - Edge AI Lab Safety System Using NVIDIA Jetson Nano

### Projek dan Repositori ini

MOCA adalah sebuah sistem pemantauan Keselamatan dan Kesehatan Kerja (K3) cerdas berbasis **Edge Computing** yang dirancang untuk memonitor lingkungan laboratorium secara *real-time*. MOCA menggunakan algoritma deteksi objek **YOLOv8** dan logika sekuensial **Finite State Machine (FSM)** untuk mengevaluasi kelengkapan Alat Pelindung Diri (APD) sebelum dan selama bekerja di lab serta mendeteksi perilaku beresiko. Prototype sistem ini beroperasi sepenuhnya secara lokal (Edge AI) menggunakan perangkat NVIDIA Jetson Nano.

Repositori ini memuat seluruh basis kode dan konfigurasi perangkat lunak untuk purwarupa MOCA. Proyek ini mengintegrasikan deteksi objek YOLO dengan logika Finite State Machine (FSM) yang **dioptimalkan untuk perangkat berdaya rendah**. Di dalam repositori ini, Anda akan menemukan konfigurasi Docker untuk isolasi environment, modul-modul skrip Python, file model AI, serta dokumentasi teknis komprehensif yang mencakup panduan instalasi, arsitektur sistem, dan optimalisasi deployment secara spesifik pada arsitektur NVIDIA Jetson Nano.

Projek ini awalnya dibangun khusus untuk ajang Research Innovation Challenge ERIC 2026. Saat anda membaca ini, sistem ini **masih berupa prototype yang belum sempurna** dan selanjutnya akan ada update perkembangan untuk lebih baik lagi. 


<details>
<summary style="font-size: 1.17em; font-weight: 500; cursor: pointer;">Abstrak</summary>
Penerapan Keselamatan dan Kesehatan Kerja (K3) di lingkungan laboratorium merupakan aspek krusial dalam mencegah kecelakaan kerja. Namun, pelaksanaan pengawasan K3 secara konvensional sering kali terbatas oleh ketelitian dan rentang perhatian manusia yang harus terus-menerus mengamati aktivitas laboratorium. Keterbatasan ini dapat diatasi dengan pemanfaatan Artificial Intelligence (AI) berbasis Edge Computing untuk melakukan pemantauan keselamatan secara real-time tanpa bergantung sepenuhnya pada pengawasan manusia. Proposal ini bertujuan untuk mengembangkan sistem Monitoring Camera (MOCA) untuk  keselamatan laboratorium yang memanfaatkan model deteksi objek You Only Look Once (YOLO) untuk mengidentifikasi keberadaan alat pelindung diri serta perilaku berisiko dari rekaman video kamera secara langsung. Hasil deteksi tersebut kemudian diproses menggunakan pendekatan Finite State Machine (FSM) untuk menentukan status keselamatan berdasarkan urutan kondisi yang terdeteksi, sehingga sistem dapat memberikan peringatan secara kontekstual dan konsisten. Sistem MOCA dapat diaplikasikan dalam dua sistem utama, yaitu sistem pengecekan sebelum masuk laboratorium dan pengawasan prosedur kerja selama dalam laboratorium. Kombinasi YOLO dan FSM dipilih karena mampu berjalan efisien pada perangkat edge dengan sumber daya komputasi terbatas, sekaligus menjaga akurasi deteksi dan kecepatan pemrosesan yang dibutuhkan untuk pengawasan real time. Pengujian dilakukan pada skenario laboratorium dengan berbagai kondisi pencahayaan dan aktivitas untuk mengevaluasi akurasi deteksi objek serta ketepatan transisi status pada FSM. Proposal ini bertujuan untuk menunjukkan bahwa sistem MOCA mampu mendeteksi pelanggaran K3 secara real-time dengan akurasi yang baik dan waktu respons cepat, sehingga berpotensi menjadi solusi pengawasan keselamatan laboratorium yang lebih efisien, konsisten, dan hemat sumber daya dibandingkan pengawasan manual.
</details>

<details>
<summary style="font-size: 1.17em; font-weight: 500; cursor: pointer;">Latar Belakang</summary>
<p>Pengawasan Keselamatan dan Kesehatan Kerja (K3) di laboratorium konvensional saat ini masih sangat bergantung pada pengamatan visual manusia yang rentan terhadap kelelahan, kelalaian, dan titik buta (*human error*). MOCA hadir sebagai solusi pengawas digital otonom yang memadukan *Computer Vision* dan logika kecerdasan buatan untuk mendeteksi pelanggaran APD secara *real-time*.</p> 

Alat ini dikembangkan khusus untuk ajang **Resarch Innovation Challenge ERIC 2026**. Saya mengembangkan alat ini dengan dua rekan saya lainnya.
</details>

<details>
<summary style="font-size: 1.17em; font-weight: 500; cursor: pointer;">Manfaat</summary>
Pengembangan sistem MOCA diharapkan dapat memberikan manfaat secara teoritis maupun praktis. Usulan ini berkontribusi terhadap kemajuan ilmu pengetahuan di bidang computer vision dan kecerdasan buatan, sekaligus menjadi rujukan akademis terkait integrasi model You Only Look Once (YOLO) dan algoritma Finite State Machine (FSM) untuk pemantauan Keselamatan dan Kesehatan Kerja (K3) secara real-time. Implementasi purwarupa ini memberikan solusi mitigasi preventif bagi pengelola laboratorium dan laboran dala mengatasi keterbatasan pengwasan visual manusia. Melalui otomasi pengawasan yang berkesinambungan, MOCA tidak hanya meminimalisir risiko kecelakaan kerja maupun kerugian material, tetapi juga berfungsi sebagai instrumen pendisiplinan efektif untuk membentuk budaya sadar keselamatan yang kuat di lingkungan perguruan tinggi.
</details>

<details>
<summary style="font-size: 1.17em; font-weight: 500; cursor: pointer;">Proposal</summary>
Coming soon
</details>


## Fitur Utama

*   **Pemrosesan Edge AI Lokal:** Inferensi AI berjalan 100% secara *offline* di perangkat, meminimalisir latensi dan menjaga privasi data tanpa bergantung pada *cloud*.
*   **Logika FSM Cerdas:** Mengeliminasi alarm palsu (*false positive*) dengan melacak transisi status APD sebelum menetapkan kondisi AMAN, PERINGATAN, atau BAHAYA.
*   **Integrasi Hardware:** Peringatan visual interaktif menggunakan antarmuka GPIO untuk mengontrol indikator LED (Merah, Kuning, Hijau) dan *Buzzer*.
*   **Smart Dashboard UI:** Pemantauan langsung dan pencatatan *log* pelanggaran K3 terintegrasi menggunakan antarmuka berbasis Streamlit.

## Spesifikasi Hardware dan Wiring

### Hardware
Berikut adalah perangkat keras utama yang dibutuhkan untuk menjalankan purwarupa ini secara optimal:

| Komponen | Spesifikasi / Detail |
| :--- | :--- |
| **SBC (Edge Device)** | NVIDIA Jetson Nano Developer Kit B01 (RAM 4GB LPDDR4, GPU 128-core Maxwell) |
| **Kamera Input** | Logitech C270 HD 720p USB Webcam |
| **Penyimpanan** | SanDisk Ultra microSDXC A1 64GB (Class 10/UHS-I) |
| **Catu Daya** | Power Adaptor 5V 4A DC Jack |


### 🔌 Konfigurasi Wiring (GPIO Jetson Nano)
.....
aaaaaa

## ⚙️ Konfigurasi Software dan Instalasi

### Spesfikasi dan Konfigurasi Software yang Digunakan
| Software | Spesifikasi / Detail |
| :--- | :--- |
| **SDK** | JetPack 4.6 |
| **PYTHON** | 3.8 |
| **YOLO** | YOLOv8 |
| **Docker Image** | [ultralytics:latest-jetson-jetpack4](https://hub.docker.com/layers/ultralytics/ultralytics/latest-jetson-jetpack4/images/sha256-cf57cd9b92e4b7f902af1cae96196cc875490540a07fc50ea16a81721f357732) |

Anda bisa melihat konfigurasi docker lebih detail pada [Dockerfile](https://github.com/SyakirAlamsyah/MOCA-PEJUANGSKS/blob/master/Dockerfile) atau pada snippets berikut ini
```
FROM ultralytics/ultralytics:latest-jetson-jetpack4
RUN apt-get update && apt-get install -y build-essential cmake
RUN pip3 install numpy==1.23.5 streamlit face_recognition
WORKDIR /workspace
```
Mengapa memakai docker? Cek [bagian ini](#keterbatasan-jetson-nano-dan-mengapa-pakai-docker).

### Cara Instalasi

## Lebih banyak terkait projek ini
### Kenapa Jetson Nano?
Berbeda dengan sistem deteksi modern yang sangat bergantung pada *Cloud Computing* atau *server* besar berspesifikasi tinggi, MOCA sengaja dirancang untuk dieksekusi 100% secara lokal pada perangkat **NVIDIA Jetson Nano Developer Kit B01**. Pemilihan arsitektur desentralisasi (*Edge Computing*) ini memberikan tiga keuntungan krusial:
1. **Zero Latency:** Respons instan dalam hitungan milidetik karena sistem tidak perlu menunggu proses transmisi video bolak-balik melalui jaringan internet.
2. **Privasi Absolut:** Video aktivitas lab tidak pernah dikirim keluar dari perangkat, mencegah risiko kebocoran data.
3. **Efisiensi Energi (Resource-Efficient):** Beroperasi dengan konsumsi daya di bawah 10 Watt, sangat sejalan dengan prinsip *Sustainable Intelligence*.

### Keterbatasan Jetson Nano
Menjalankan sistem deteksi AI modern di atas perangkat lawas seperti Jetson Nano (dengan OS bawaan Ubuntu 18.04 & Python 3.6) menghadirkan tantangan bottleneck memori dan termal yang ekstrem. Dengan itu dilakukan optimalisasi rekayasa perangkat lunak melalui:
*   **Algorithmic Efficiency (YOLO + FSM):** Alih-alih menggunakan algoritma pelacakan visual yang menguras CPU, kami menggunakan Finite State Machine (FSM) berkompleksitas matematika konstan untuk mengevaluasi status K3 (Aman, Peringatan, Bahaya) secara sekuensial. Logika ini secara efektif mengeliminasi alarm palsu akibat kedipan kamera tanpa memicu panas berlebih (thermal throttling).
*   **Pakai Docker?:** Seluruh sistem dibungkus menggunakan Docker untuk memastikan isolasi environment (menjalankan Python 3.8) secara mulus tanpa merusak pustaka bawaan sistem host. Ini juga dilakukan karena Jetpack Jetson Nano terbatas pada versi Python yang sudah ada di Jetpack nya (3.6). Python versi lebih tinggi tidak didukung oleh Jetson Nano dan Jetpack nya, sehingga memperbaharui versi akan menjadi kurang optimal. **Agar tetap bisa menggunakan YOLO dan memanfaatkan kekuatan GPU Jetson Nano dengan optimal, maka digunakan docker**.
*   **TensorRT Acceleration:** Mengonversi model standar YOLOv8 (.pt) menjadi format mesin TensorRT (.engine) guna memaksimalkan utilitas dari 128-core GPU Maxwell bawaan, mendongkrak performa FPS secara drastis pada perangkat edge.





