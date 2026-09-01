# MOCA (Prototype) - Edge AI Lab Safety System Using NVIDIA Jetson Nano

### Projek dan Repositori ini

MOCA adalah sebuah sistem pemantauan Keselamatan dan Kesehatan Kerja (K3) cerdas berbasis **Edge Computing** yang dirancang untuk mencegah kecelakaan di lingkungan laboratorium secara *real-time*. MOCA mengintegrasikan algoritma deteksi objek **YOLOv8** dan logika sekuensial **Finite State Machine (FSM)** untuk mengevaluasi kelengkapan Alat Pelindung Diri (APD) sebelum dan selama bekerja serta mendeteksi perilaku beresiko. Prototype sistem ini beroperasi sepenuhnya secara lokal (Edge AI) menggunakan perangkat NVIDIA Jetson Nano.

Repositori ini berisi program dan file-file yang menjadi dasar perangkat lunak dari prototype MOCA. Mulai dari konfigurasi docker, kode python, dan model yang digunakan. Saya juga akan menambahkan beberapa hal lain seperti cara instalasi, bagaimana menggunakan YOLO di Jetson Nano, penjelasan setiap script python, dan lainnya berkaitan dengan teknis penggunaan Jetson Nano.

### Latar Belakang
Pengawasan Keselamatan dan Kesehatan Kerja (K3) di laboratorium konvensional saat ini masih sangat bergantung pada pengamatan visual manusia yang rentan terhadap kelelahan, kelalaian, dan titik buta (*human error*). MOCA hadir sebagai solusi pengawas digital otonom yang memadukan *Computer Vision* dan logika kecerdasan buatan untuk mendeteksi pelanggaran APD secara *real-time*. 

Alat ini dikembangkan khusus untuk ajang **Resarch Innovation Challenge ERIC 2026**. Saya mengembangkan alat ini dengan dua rekan saya lainnya.