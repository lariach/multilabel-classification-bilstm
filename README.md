# Klasifikasi Multilabel Berita menggunakan BiLSTM

Repositori ini berisi implementasi kode dan data untuk penelitian skripsi mengenai klasifikasi teks multilabel pada berita menggunakan model **Bidirectional Long Short-Term Memory (BiLSTM)**.

---

## 📁 Struktur Folder

Penyusunan folder telah diurutkan berdasarkan alur kerja penelitian (pipeline) untuk memudahkan navigasi:

* **`01_scraping/`**: Script Python untuk pengambilan data berita
* **`02_data/`**: 
    * `raw/`: Dataset mentah hasil scraping.
    * `labeled/`: Dataset yang telah melalui proses pelabelan manual (Dataset 1 & Dataset 2).
    * `metadata/`: File pendukung pemrosesan teks.
* **`03_experiments/`**: Koleksi 4 file Jupyter Notebook (.ipynb) utama yang mencakup seluruh proses dari preprocessing hingga evaluasi.
* **`04_models/`**: Wadah penyimpanan model (Hanya berisi `.gitkeep`).
  > **Catatan:** Folder ini dikosongkan untuk efisiensi ruang. File model akan ter-generate otomatis saat notebook dijalankan.
* **`05_app/`**: Implementasi aplikasi demo interaktif berbasis Streamlit.
* **`06_src/`**: File `.py` untuk kebutuhan lampiran naskah skripsi.
* **`99_archived/`**: Dokumentasi eksperimen tambahan (model BERT) yang tidak masuk dalam naskah utama.

---

## 🚀 Ringkasan Eksperimen

Penelitian ini mengevaluasi model BiLSTM melalui 4 skenario utama guna menguji ketangguhan model:

| No | Eksperimen | Dataset | Metode Validasi | Deskripsi |
|:---:|---|---|---|---|
| 1 | **Experiment 1** | Dataset 1 | Hold-out (80:20) | Baseline model pada dataset original. |
| 2 | **Experiment 2** | Dataset 2 | Hold-out (80:20) | Pengujian pada dataset variasi ke-2. |
| 3 | **Experiment 3** | Dataset 2 | K-Fold Cross Val | Uji stabilitas model pada Dataset 2. |
| 4 | **Experiment 4** | Dataset 1 | K-Fold Cross Val | Uji stabilitas model pada Dataset 1. |

---

## 🛠️ Cara Penggunaan

### 1. Instalasi
Pastikan Anda telah menginstal Python 3.x, kemudian instal library yang dibutuhkan melalui terminal:
```bash
pip install -r requirements.txt
```

### 2. Menjalankan Eksperimen
Buka folder 03_experiments/ dan jalankan notebook menggunakan Jupyter atau VS Code secara berurutan. Setiap notebook mencakup proses instalasi library, pembersihan data, pelatihan, hingga evaluasi metrik.

### 3. Menjalankan Demo Aplikasi
Untuk menjalankan antarmuka prediksi teks secara interaktif, gunakan perintah berikut:
```bash
streamlit run 05_app/demo.py
```

---

## 📊 Hasil Penelitian

Berdasarkan pengujian yang telah dilakukan terhadap empat skenario eksperimen, berikut adalah ringkasan performa model terbaik:

| Metrik Evaluasi | Skor Akurasi |
| :--- | :---: |
| **Model Terbaik** | Eksperimen [Isi No] |
| **F1-Score** | 0.XX |
| **Hamming Loss** | 0.XX |
| **Precision** | 0.XX |
| **Recall** | 0.XX |

> **Catatan:** Hasil detail untuk setiap *fold* pada K-Fold Cross Validation dapat dilihat langsung di dalam Notebook pada folder `03_experiments/`.

---

## ✒️ Identitas Penulis

<table border="0">
  <tr>
    <td><strong>Nama</strong></td>
    <td>: Chairal Octavyanz Tanjung</td>
  </tr>
  <tr>
    <td><strong>NPM</strong></td>
    <td>: 140810210030</td>
  </tr>
  <tr>
    <td><strong>Program Studi</strong></td>
    <td>: S1-Teknik Informatika</td>
  </tr>
  <tr>
    <td><strong>Fakultas</strong></td>
    <td>: Fakultas Matematika dan Ilmu Pengetahuan Alam</td>
  </tr>
  <tr>
    <td><strong>Universitas</strong></td>
    <td>: Universitas Padjadjaran</td>
  </tr>
</table>

---
*Proyek ini dikembangkan sebagai bagian dari Tugas Akhir (Skripsi) untuk memperoleh gelar Sarjana Komputer.*