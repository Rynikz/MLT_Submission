# Laporan Proyek Machine Learning - Wahyu Adji Agus Saputra
---
## Domain Proyek

Proyek ini berfokus pada domain **analisis prediktif** di sektor **biaya pendidikan internasional**. Tujuan utamanya adalah untuk mengidentifikasi faktor-faktor yang secara signifikan memengaruhi total biaya pendidikan dan membangun sebuah model *machine learning* yang andal untuk mengestimasi biaya tersebut. Secara spesifik, proyek ini akan memprediksi biaya kuliah (`Tuition_USD`) berdasarkan berbagai atribut terkait yang tersedia dalam dataset.

---
## Latar Belakang

Biaya pendidikan, terutama di tingkat internasional, merupakan salah satu pertimbangan paling krusial bagi calon siswa. Variabilitas biaya ini sangat tinggi, dipengaruhi oleh berbagai faktor seperti negara, kota, jenis institusi, dan program studi yang dipilih. Kemampuan untuk memprediksi estimasi biaya secara akurat menjadi alat yang sangat penting untuk perencanaan keuangan yang matang. Proyek ini bertujuan untuk menganalisis hubungan antara faktor-faktor tersebut dengan biaya kuliah (`Tuition_USD`) menggunakan dataset yang relevan.

---
## Tujuan Proyek

1.  Menganalisis dataset untuk mengidentifikasi pola dan tren yang berkaitan dengan biaya kuliah.
2.  Mengidentifikasi faktor-faktor kunci yang paling berpengaruh terhadap biaya kuliah.
3.  Membangun dan mengevaluasi model prediktif yang mampu mengestimasi biaya kuliah secara akurat.

---
## Business Understanding

### Pernyataan Masalah (Problem Statements)
1.  Bagaimana cara memprediksi biaya kuliah (`Tuition_USD`) secara akurat berdasarkan atribut yang tersedia dalam dataset?
2.  Faktor-faktor apa saja yang paling signifikan dalam memengaruhi variasi biaya kuliah internasional?
3.  Model *machine learning* manakah yang memberikan performa terbaik untuk tugas prediksi ini?

### Tujuan (Goals)
1.  Mengembangkan sebuah model prediktif yang mampu mengestimasi `Tuition_USD` dengan tingkat kesalahan (MAE) yang rendah.
2.  Menganalisis fitur-fitur kunci untuk memahami kontribusi terbesarnya terhadap perbedaan biaya.
3.  Mengevaluasi beberapa algoritma regresi dan memilih model dengan performa terbaik berdasarkan metrik evaluasi seperti *Mean Absolute Error (MAE)* dan *R2 Score*.

### Pernyataan Solusi (Solution Statements)
Solusi yang diimplementasikan dalam proyek ini mencakup beberapa tahapan utama:
1.  **Analisis Data Eksploratif (EDA):** Memahami distribusi data, mengidentifikasi outlier, menganalisis korelasi, dan memvisualisasikan hubungan antara fitur dengan variabel target.
2.  **Pra-pemrosesan Data Komprehensif:**
    * Menangani fitur dengan kardinalitas sangat tinggi (`City`, `University`, `Program`) dengan cara menghapusnya untuk menyederhanakan model.
    * Menerapkan **One-Hot Encoding** untuk fitur kategorikal yang tersisa (`Country`, `Level`).
    * Membagi data menjadi set pelatihan dan pengujian.
    * Melakukan **Standarisasi** pada fitur-fitur numerik (`Duration_Years`, `Rent_USD`, dll.) menggunakan `StandardScaler`.
3.  **Pengembangan Model Machine Learning:** Membuat, melatih, dan mengevaluasi lima model regresi berbeda: Linear Regression, KNN Regressor, SVR, Random Forest Regressor, dan Gradient Boosting Regressor.
4.  **Evaluasi Model:** Menggunakan metrik MAE, MSE, RMSE, dan R2 Score untuk membandingkan performa setiap model pada data uji secara objektif.

---
## Data Understanding

Dataset yang digunakan pada proyek ini adalah **International Education Costs - Unveiling Global Study Expenses**, yang diperoleh dari platform Kaggle.
* **Tautan Sumber Dataset:** [https://www.kaggle.com/datasets/adilshamim8/cost-of-international-education](https://www.kaggle.com/datasets/adilshamim8/cost-of-international-education)

Dataset awal terdiri dari **907 baris dan 12 kolom**.

### Deskripsi Variabel
Berikut adalah deskripsi dari semua fitur yang ada pada dataset awal:
* **`Country`**: Negara tempat universitas berada (Kategorikal).
* **`City`**: Kota tempat universitas berada (Kategorikal).
* **`University`**: Nama universitas (Kategorikal).
* **`Program`**: Nama program studi (Kategorikal).
* **`Level`**: Jenjang pendidikan, misal: Bachelor, Master (Kategorikal).
* **`Tuition_USD`**: Biaya kuliah dalam USD (Numerik, **Variabel Target**).
* **`Duration_Years`**: Durasi program dalam tahun (Numerik).
* **`Living_Cost_Index`**: Indeks biaya hidup di kota/negara terkait (Numerik).
* **`Rent_USD`**: Perkiraan biaya sewa bulanan dalam USD (Numerik).
* **`Visa_Fee_USD`**: Biaya aplikasi visa dalam USD (Numerik).
* **`Insurance_USD`**: Perkiraan biaya asuransi tahunan dalam USD (Numerik).
* **`Exchange_Rate`**: Nilai tukar mata uang lokal terhadap USD (Numerik).

### Ringkasan Temuan EDA
> Berdasarkan analisis data eksploratif, ditemukan beberapa wawasan penting mengenai kondisi dan karakteristik data.

* **Kondisi Data Awal:** Hasil pengecekan menunjukkan bahwa **tidak ditemukan nilai yang hilang (*missing values*)** maupun data duplikat pada keseluruhan dataset.
* **Distribusi Target (`Tuition_USD`):** Distribusi variabel target menunjukkan **kemiringan positif (skewness 0.71)**. Nilai ini mengindikasikan bahwa sebagian besar data biaya kuliah terkonsentrasi pada nilai yang lebih rendah, dengan beberapa nilai ekstrem yang jauh lebih tinggi.
* **Fitur Kategorikal vs. Target:** Analisis visual menunjukkan fitur `Level` (jenjang pendidikan) dan `Country` (negara) memiliki pengaruh yang jelas terhadap median dan sebaran biaya kuliah `Tuition_USD`.
* **Korelasi Fitur Numerik:** Heatmap korelasi menyoroti bahwa `Rent_USD` (biaya sewa) dan `Living_Cost_Index` (indeks biaya hidup) menunjukkan korelasi positif yang cukup kuat dengan `Tuition_USD`. Ini mengindikasikan bahwa biaya kuliah cenderung lebih tinggi di lokasi dengan biaya hidup dan sewa yang lebih tinggi.

---
## Data Preparation

Tahapan persiapan data dilakukan secara sistematis untuk memastikan data siap digunakan untuk pemodelan.
1.  **Penghapusan Kolom Kardinalitas Tinggi:** Kolom `City`, `University`, dan `Program` dihapus. Ketiga kolom ini memiliki ribuan nilai unik yang jika dipertahankan akan menghasilkan dimensi data yang sangat besar setelah proses *encoding*, berisiko menyebabkan *curse of dimensionality*.
2.  **One-Hot Encoding Fitur Kategorikal:** Fitur `Country` dan `Level` diubah menjadi representasi numerik biner menggunakan `pd.get_dummies(drop_first=True)` untuk menghindari multikolinearitas.
3.  **Pemisahan Fitur (X) dan Target (y):** Dataset dipisahkan menjadi `X` (kumpulan fitur) dan `y` (variabel target, `Tuition_USD`).
4.  **Pembagian Data Latih dan Uji:** Data dibagi menjadi set pelatihan (80%) dan set pengujian (20%) dengan `random_state=42` untuk memastikan hasil yang dapat direproduksi.
5.  **Standarisasi Fitur Numerik:** Fitur-fitur numerik pada `X_train` dan `X_test` distandarisasi menggunakan `StandardScaler`. *Scaler* ini dilatih (*fit*) **hanya** pada data latih (`X_train`) dan kemudian digunakan untuk mentransformasi `X_train` dan `X_test` guna mencegah kebocoran data (*data leakage*).

---
## Modeling

Lima model regresi *machine learning* dikembangkan untuk memprediksi biaya kuliah. Setiap model dipilih untuk merepresentasikan pendekatan yang berbeda dalam pemodelan.

1.  **Linear Regression:**
    * **Cara Kerja:** Model ini bekerja dengan mencari hubungan linier terbaik antara fitur-fitur input dan variabel target. Tujuannya adalah menemukan garis (atau hyperplane) yang meminimalkan jumlah selisih kuadrat antara nilai aktual dan nilai prediksi.
    * **Parameter:** Tidak ada parameter utama yang disetel secara manual pada proyek ini.

2.  **K-Nearest Neighbors (KNN) Regressor:**
    * **Cara Kerja:** KNN adalah algoritma non-parametrik yang memprediksi nilai target dari sebuah data baru berdasarkan rata-rata nilai target dari 'k' tetangga terdekatnya di ruang fitur.
    * **Parameter:** `n_neighbors=7`, artinya prediksi didasarkan pada 7 data terdekat dari set pelatihan.

3.  **Support Vector Regression (SVR):**
    * **Cara Kerja:** SVR bertujuan untuk menemukan hyperplane yang paling optimal dengan memaksimalkan margin atau jarak antara hyperplane dan titik data terdekat (disebut *support vectors*). Berbeda dengan regresi biasa yang mencoba meminimalkan eror, SVR mencoba agar eror tidak melebihi batas tertentu.
    * **Parameter:** `kernel='rbf'`, `C=100`, `gamma='scale'`. Parameter ini mendefinisikan penggunaan kernel non-linier dan mengatur trade-off antara kompleksitas model dan toleransi eror.

4.  **Random Forest Regressor:**
    * **Cara Kerja:** Ini adalah model *ensemble* yang membangun banyak *decision tree* secara independen pada berbagai sub-sampel data. Prediksi akhir adalah rata-rata dari prediksi semua pohon, yang membuatnya lebih kuat dan tahan terhadap *overfitting* dibandingkan satu *decision tree*.
    * **Parameter:** `n_estimators=100` (jumlah pohon), `max_depth=10` (kedalaman maksimum setiap pohon).

5.  **Gradient Boosting Regressor:**
    * **Cara Kerja:** Model *ensemble* ini juga membangun banyak *decision tree*, tetapi secara sekuensial. Setiap pohon baru dibangun untuk memperbaiki kesalahan (residu) dari pohon sebelumnya, sehingga model secara bertahap belajar dari kesalahan dan meningkatkan akurasinya.
    * **Parameter:** `n_estimators=100` (jumlah pohon), `learning_rate=0.1` (laju pembelajaran), `max_depth=5` (kedalaman maksimum setiap pohon).

---
## Evaluation

Performa kelima model dievaluasi pada data uji menggunakan metrik standar regresi: MAE, MSE, RMSE, dan R2 Score.

**Tabel Perbandingan Performa Model**
> Tabel berikut menampilkan hasil evaluasi yang diurutkan berdasarkan *Mean Absolute Error (MAE)* terendah.

| Model | MAE | MSE | RMSE | R2 Score |
| :--- | :--- | :--- | :--- | :--- |
| **Gradient Boosting Regressor** | **2,103.46** | 10,616,859.39 | 3,258.35 | 0.9593 |
| Random Forest Regressor | 2,164.71 | **10,406,009.93** | **3,225.83** | **0.9601** |
| KNN Regressor | 2,925.22 | 25,668,206.43 | 5,066.38 | 0.9016 |
| Linear Regression | 3,501.91 | 38,487,687.86 | 6,203.84 | 0.8525 |
| SVR | 10,853.95 | 191,969,992.83 | 13,855.32 | 0.2641 |

### Analisis Performa Model
* **Model Terbaik:** **Random Forest Regressor** dan **Gradient Boosting Regressor** menunjukkan performa terbaik dan sangat kompetitif.
    * **Random Forest** unggul dengan **R2 Score tertinggi (0.9601)** dan **RMSE terendah (3,225.83 USD)**. Ini berarti model ini mampu menjelaskan sekitar 96.01% variabilitas dalam data biaya kuliah.
    * **Gradient Boosting** memiliki **MAE terendah (2,103.46 USD)**, yang berarti rata-rata kesalahan prediksi model ini adalah sekitar $2,103.
* **Performa Model Lain:** Model KNN menunjukkan performa yang cukup baik, sedangkan Linear Regression dan SVR menunjukkan performa yang jauh lebih rendah. R2 Score SVR yang hanya 0.2641 menunjukkan ketidakcocokannya dengan sifat data yang kemungkinan besar sangat non-linier dan kompleks.

---
## Kesimpulan

Proyek ini berhasil menjawab ketiga problem statement yang telah dirumuskan:
1.  Model prediktif untuk biaya kuliah berhasil dikembangkan. **Random Forest Regressor** dan **Gradient Boosting Regressor** muncul sebagai model dengan performa terbaik. Random Forest mencapai R2 Score **0.9601**, sementara Gradient Boosting mencapai MAE terendah sebesar **2,103.46 USD**. Keduanya sangat efektif untuk mengestimasi `Tuition_USD`.

2.  Berdasarkan analisis data eksploratif (EDA), faktor-faktor yang paling signifikan memengaruhi biaya kuliah adalah **biaya sewa (`Rent_USD`)** dan **indeks biaya hidup (`Living_Cost_Index`)**. Kedua faktor ini memiliki korelasi positif yang kuat dengan `Tuition_USD`, mengindikasikan bahwa biaya pendidikan cenderung lebih tinggi di kota-kota dengan biaya hidup yang mahal.

3.  Analisis menegaskan bahwa data biaya pendidikan memiliki sifat non-linier. Hal ini terlihat dari performa model *ensemble* berbasis pohon (Random Forest, Gradient Boosting) yang jauh mengungguli model linier (Linear Regression, SVR).

**Rekomendasi Langkah Selanjutnya:**
* **Hyperparameter Tuning:** Melakukan *tuning* parameter secara sistematis untuk model Random Forest dan Gradient Boosting menggunakan teknik seperti Grid Search atau Randomized Search untuk optimasi lebih lanjut.
* **Feature Engineering:** Mengeksplorasi cara yang lebih canggih untuk menangani fitur kardinalitas tinggi yang sebelumnya dihapus (misalnya, menggunakan *target encoding* atau *embedding*).
* **Analisis Feature Importance:** Mengekstrak dan menganalisis `feature_importances_` dari model Random Forest atau Gradient Boosting untuk mendapatkan konfirmasi kuantitatif mengenai fitur mana yang paling berpengaruh pada prediksi.
