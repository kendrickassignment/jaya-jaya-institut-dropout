# Proyek Akhir: Menyelesaikan Permasalahan Perusahaan Edutech

| Info | Detail |
|------|--------|
| **Nama** | Kendrick Filbert |
| **Email** | kendrickfilbert@gmail.com |
| **ID Dicoding** | kendrickfff |

---

## Business Understanding

**Jaya Jaya Institut** adalah institusi pendidikan perguruan tinggi yang telah berdiri sejak tahun 2000 dan memiliki reputasi sangat baik dalam mencetak lulusan berkualitas. Namun, institusi ini menghadapi tantangan serius dengan tingginya persentase mahasiswa yang tidak menyelesaikan pendidikannya (*dropout*) — mencapai **32.1%** dari total mahasiswa.

Tingginya angka dropout berdampak negatif pada:
- **Reputasi institusi** di mata calon mahasiswa dan stakeholder
- **Pendapatan institusi** karena kehilangan mahasiswa aktif
- **Kesempatan mahasiswa** yang kehilangan akses pendidikan tinggi

Oleh karena itu, Jaya Jaya Institut membutuhkan **sistem deteksi dini** berbasis data untuk mengidentifikasi mahasiswa berisiko *sebelum* mereka benar-benar dropout, sehingga intervensi tepat waktu dapat diberikan.

### Permasalahan Bisnis

1. **Apa faktor-faktor utama** yang paling memengaruhi seorang mahasiswa untuk melakukan *dropout*?
2. **Bagaimana memprediksi** secara akurat apakah seorang mahasiswa akan *dropout* berdasarkan data akademik dan status demografis-finansial mereka?

### Cakupan Proyek

| Tahap | Aktivitas |
|-------|-----------|
| Data Understanding & EDA | Analisis mendalam terhadap 37 fitur dataset, visualisasi distribusi, korelasi, dan pola dropout |
| Data Preparation | Filtering, encoding, feature selection berbasis korelasi dan domain knowledge |
| Modeling | LightGBM dengan RandomizedSearchCV (50 iterasi, StratifiedKFold 5-fold) |
| Evaluation | Accuracy, Recall, Precision, F1, ROC-AUC, Precision-Recall Curve, Feature Importance |
| Dashboard | Looker Studio — monitoring performa dan risiko mahasiswa secara visual |
| Deployment | Streamlit Community Cloud — prototipe prediksi interaktif dengan fitur batch prediction |

---

## Persiapan

**Sumber data:** `data.csv` (Dataset Performa & Status Mahasiswa Jaya Jaya Institut, 4.424 baris, 37 kolom)

**Setup environment:**

```bash
# Clone atau download proyek ini
git clone <repo-url>
cd jaya-jaya-dropout-prediction

# Install semua dependensi
pip install -r requirements.txt
```

### Struktur Proyek

```
├── data.csv                  # Dataset utama
├── notebook.ipynb            # Analisis & pemodelan lengkap
├── app.py                    # Aplikasi Streamlit
├── model_lgb.pkl             # Model LightGBM tersimpan
├── model_features.json       # Daftar fitur model
├── pd.py                     # Konversi CSV untuk Looker Studio
├── requirements.txt          # Daftar library
└── README.md                 # Dokumentasi proyek
```

---

## Business Dashboard

Dashboard dibuat menggunakan **Looker Studio** untuk memvisualisasikan data mahasiswa dan menyoroti metrik penting seperti:
- Distribusi status mahasiswa (Graduate / Dropout / Enrolled)
- Analisis faktor finansial (status SPP, beasiswa, utang)
- Perbandingan performa akademik semester 1 vs semester 2
- Tren dropout berdasarkan program studi, usia, dan gender

**Tautan Dashboard:**  
👉 [Buka Dashboard Looker Studio](https://lookerstudio.google.com/) *(masukkan link dashboard aktif di sini)*

> **Catatan:** Untuk upload data ke Looker Studio, jalankan `pd.py` terlebih dahulu:
> ```bash
> python pd.py
> ```
> File `data_looker_studio.csv` akan dihasilkan dan siap diupload.

---

## Menjalankan Sistem Machine Learning

### Prasyarat

Pastikan `model_lgb.pkl` sudah ada (dihasilkan dari menjalankan seluruh sel di `notebook.ipynb`).

### Jalankan Secara Lokal

```bash
streamlit run app.py
```

Aplikasi akan berjalan di `http://localhost:8501`

### Fitur Aplikasi

| Fitur | Deskripsi |
|-------|-----------|
| **Prediksi Individual** | Input data satu mahasiswa, dapatkan probabilitas dropout + level risiko + gauge chart |
| **Analisis Faktor Risiko** | Identifikasi otomatis faktor-faktor spesifik yang meningkatkan risiko mahasiswa tersebut |
| **Radar Chart** | Perbandingan profil mahasiswa vs rata-rata Graduate dan Dropout |
| **Rekomendasi Tindakan** | Saran intervensi yang dipersonalisasi berdasarkan level risiko |
| **Prediksi Batch** | Upload CSV berisi banyak mahasiswa, prediksi massal sekaligus |
| **Download Hasil** | Export hasil prediksi batch ke file CSV untuk tindak lanjut |

**Tautan Aplikasi Streamlit (Cloud):**  
👉 [Buka Aplikasi Prediksi Dropout](https://jayajaya-dropout-prediction.streamlit.app/) *(masukkan link aktif setelah deploy)*

---

## Conclusion

Berdasarkan seluruh proses analisis dan pemodelan yang dilakukan:

### Jawaban atas Permasalahan Bisnis

**1. Faktor-faktor dominan yang memengaruhi dropout:**

| Ranking | Faktor | Kategori | Kekuatan Prediksi |
|---------|--------|----------|------------------|
| 1 | Jumlah SKS lulus Semester 1 & 2 | Akademik | ⭐⭐⭐⭐⭐ Sangat Tinggi |
| 2 | Nilai rata-rata Semester 1 & 2 | Akademik | ⭐⭐⭐⭐⭐ Sangat Tinggi |
| 3 | Status pembayaran SPP (Tuition fees) | Finansial | ⭐⭐⭐⭐ Tinggi |
| 4 | Status Debitur/Utang | Finansial | ⭐⭐⭐ Sedang |
| 5 | Usia saat enrollment | Demografis | ⭐⭐⭐ Sedang |
| 6 | Status beasiswa | Finansial | ⭐⭐ Moderat |
| 7 | Nilai masuk (Admission grade) | Akademik | ⭐⭐ Moderat |

**2. Performa Model Machine Learning:**

Mahasiswa yang menunggak SPP memiliki *dropout rate* **>70%**, jauh di atas rata-rata 32.1%. Sementara itu, mahasiswa yang lulus kurang dari 3 SKS di semester pertama hampir pasti akan *dropout*.

Model LightGBM yang telah di-*tuning* berhasil mencapai:
- **Accuracy**: >87%
- **Recall (Dropout)**: >85% — artinya 85% dari mahasiswa yang sebenarnya dropout berhasil terdeteksi
- **ROC-AUC**: >0.92 — model sangat baik dalam membedakan Graduate vs Dropout

---

## Rekomendasi Action Items

### 🚨 Prioritas Tinggi (Immediate Action)

1. **Sistem Peringatan Dini Akademik**  
   Implementasikan sistem monitoring otomatis yang memflag mahasiswa dengan SKS lulus <3 di pertengahan semester pertama. Jadwalkan sesi konseling akademik wajib dalam 2 minggu setelah flag muncul.

2. **Intervensi Finansial Proaktif**  
   Buat program "Mahasiswa Berisiko Finansial" yang menawarkan skema cicilan SPP tanpa denda dan beasiswa parsial bagi mahasiswa yang terdeteksi menunggak sebelum mereka memutuskan berhenti kuliah.

### ⚠️ Prioritas Menengah (Short-Term)

3. **Program Mentoring Berbasis Data**  
   Gunakan output model setiap awal semester untuk membuat daftar 100 mahasiswa prioritas yang mendapat pendampingan intensif dari dosen wali dan mahasiswa senior berprestasi.

4. **Integrasi Dashboard Real-Time**  
   Hubungkan dashboard Looker Studio dengan sistem informasi akademik kampus agar data selalu terbarui secara otomatis, memungkinkan manajemen memantau tren dropout setiap saat.

### 📅 Prioritas Normal (Long-Term)

5. **Re-training Model Berkala**  
   Lakukan re-training model setiap akhir tahun ajaran menggunakan data terbaru untuk menjaga akurasi prediksi dan menangkap perubahan pola perilaku mahasiswa dari waktu ke waktu.

6. **Evaluasi Program Intervensi**  
   Ukur dampak setiap program intervensi terhadap penurunan angka dropout. Gunakan A/B testing untuk mengoptimalkan jenis intervensi yang paling efektif per segmen mahasiswa.
