# 🩺 Sistem Diagnosis Diabetes Mellitus Berbasis Web Menggunakan Naive Bayes, SVM, KNN, dan Random Forest

![Python](https://img.shields.io/badge/Python-3.9%2B-blue.svg)
![Streamlit](https://img.shields.io/badge/Framework-Streamlit-ff4b4b.svg)
![License](https://img.shields.io/badge/Lisensi-MIT-green.svg)
![Status](https://img.shields.io/badge/Status-Selesai-success.svg)

---

## 📘 Deskripsi Proyek
Proyek ini merupakan **perbandingan empat algoritma machine learning** — **Naive Bayes**, **Support Vector Machine (SVM)**, **K-Nearest Neighbor (KNN)**, dan **Random Forest** — untuk **diagnosis penyakit Diabetes Mellitus** berbasis web menggunakan **Streamlit**.  

Tujuan utama dari sistem ini adalah membantu tenaga medis dalam mendiagnosis diabetes secara **lebih cepat, efisien, dan akurat** berdasarkan data pasien seperti **usia, kadar glukosa, tekanan darah, ketebalan kulit, insulin, dan BMI**.

---

## 🎯 Tujuan
- Membangun **sistem diagnosis diabetes berbasis web**.  
- Membandingkan performa empat algoritma klasifikasi populer.  
- Menentukan algoritma dengan **akurasi terbaik** untuk diagnosis penyakit diabetes.

---

## ⚙️ Fitur Utama
- Upload dan preprocessing dataset diabetes.  
- Pelatihan model menggunakan:
  - 🧮 Naive Bayes  
  - 🧠 Support Vector Machine (SVM)  
  - 👥 K-Nearest Neighbor (KNN)  
  - 🌲 Random Forest  
- Visualisasi **Confusion Matrix** dan metrik evaluasi (akurasi, presisi, recall, F1-score).  
- Prediksi penyakit secara real-time melalui **antarmuka Streamlit**.  

---

## 🧠 Metodologi
1. **Pengumpulan Data**  
   Dataset diambil dari *RSUD Syarifah Ambami Rato Ebu Bangkalan* yang terdiri dari 120 data pasien dengan 6 parameter utama.

2. **Preprocessing Data**  
   - Menangani missing value  
   - Menyeimbangkan data menggunakan **SMOTE (oversampling)**  
   - Pembagian data menjadi **data training dan testing (90:10)**  

3. **Pemodelan**  
   Menerapkan empat algoritma klasifikasi: Naive Bayes, SVM, KNN, dan Random Forest menggunakan **Python (scikit-learn)**.

4. **Evaluasi Model**  
   Mengukur akurasi, recall, precision, dan f-measure menggunakan **Excel**, **Orange**, dan **Python**.

---

## 📊 Hasil Perbandingan
| Algoritma | Akurasi (Excel) | Akurasi (Orange) | Akurasi (Python) |
|------------|----------------|------------------|------------------|
| Naive Bayes | 83% | 66% | 67% |
| SVM | — | 69% | **81%** |
| KNN | — | **73%** | 63% |
| Random Forest | — | 65% | 71% |

✅ **Algoritma terbaik:** **Support Vector Machine (SVM)** dengan akurasi **81%**

---

## 🌐 Aplikasi Web
Aplikasi web ini dibuat menggunakan **Streamlit** dan memungkinkan pengguna untuk:
- Menginput data pasien (usia, glukosa, tekanan darah, ketebalan kulit, insulin, BMI).  
- Memilih metode diagnosis (Naive Bayes, KNN, SVM, Random Forest).  
- Menampilkan hasil prediksi secara langsung (Diabetes / Tidak Diabetes).  

---

## 💻 Teknologi yang Digunakan
| Kategori | Teknologi |
|-----------|------------|
| Bahasa Pemrograman | Python 3.x |
| Framework | Streamlit |
| Machine Learning | scikit-learn |
| Pengolahan Data | Pandas, NumPy |
| Visualisasi | Matplotlib |
| Penyeimbangan Data | Imbalanced-learn (SMOTE) |
| Penyimpanan Model | Pickle |
