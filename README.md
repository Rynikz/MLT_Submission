# Laporan Proyek Machine Learning - Wahyu Adji Agus Saputra

## Domain Proyek
Domain proyek ini adalah *predictive analytics* dalam sektor **biaya pendidikan internasional**. Analisis prediktif menggunakan berbagai teknik statistik dan *machine learning* untuk membuat prediksi tentang hasil di masa depan berdasarkan data historis. Dalam konteks biaya pendidikan internasional, tujuannya adalah untuk mengidentifikasi faktor-faktor yang memengaruhi total biaya yang harus dikeluarkan oleh siswa yang menempuh pendidikan di luar negeri dan membangun model yang dapat mengestimasi biaya tersebut. Aplikasi dari analisis ini sangat luas, mulai dari membantu calon siswa dan keluarga mereka dalam merencanakan anggaran pendidikan, hingga memberikan masukan bagi institusi pendidikan dan pembuat kebijakan dalam memahami dinamika biaya pendidikan global. Proyek ini secara spesifik akan fokus pada prediksi biaya kuliah (`Tuition_USD`) menggunakan berbagai atribut terkait lokasi, program, dan estimasi biaya hidup.

## Latar Belakang
Biaya pendidikan, terutama untuk studi di luar negeri, terus menjadi pertimbangan utama dan seringkali menjadi beban finansial yang signifikan bagi calon siswa dan keluarga mereka. Biaya ini tidak hanya mencakup uang kuliah (*tuition fee*), tetapi juga biaya hidup, akomodasi, dan pengeluaran lainnya yang sangat bervariasi antar negara, kota, dan jenis institusi. Kemampuan untuk memprediksi biaya pendidikan internasional menjadi krusial dalam upaya perencanaan keuangan yang efektif dan pengambilan keputusan yang lebih baik.

Prediksi biaya yang akurat memungkinkan perencanaan anggaran yang lebih matang bagi calon siswa, membantu institusi pendidikan dalam menetapkan biaya yang kompetitif, serta mendukung agen pendidikan dalam memberikan konsultasi yang komprehensif. Faktor-faktor yang memengaruhi biaya ini sangat beragam, meliputi aspek geografis, pilihan program dan jenjang studi, serta standar hidup di negara tujuan. Dalam proyek ini, akan dianalisis hubungan antara faktor-faktor ini dengan biaya kuliah (`Tuition_USD`) menggunakan dataset `International_Education_Costs.csv` dan dibangun beberapa model *Machine Learning* untuk dievaluasi performanya.

## Tujuan Proyek
Memahami faktor-faktor yang memengaruhi biaya pendidikan internasional sangat penting bagi berbagai pihak. Proyek ini bertujuan untuk:  
1.  Menganalisis dataset `International_Education_Costs.csv` untuk mengidentifikasi pola dan tren biaya kuliah di berbagai negara dan jenis program.  
2.  Mengidentifikasi faktor-faktor kunci yang paling signifikan memengaruhi biaya kuliah (`Tuition_USD`).  
3.  Membangun dan mengevaluasi model prediktif yang mampu mengestimasi biaya kuliah (`Tuition_USD`) berdasarkan atribut-atribut yang relevan.

## Business Understanding

### Problem Statements  
Berdasarkan latar belakang di atas, rincian masalahnya adalah sebagai berikut:  
1.  Bagaimana cara memprediksi biaya kuliah (`Tuition_USD`) berdasarkan atribut seperti negara, kota, universitas, program, level pendidikan, durasi, biaya sewa, biaya visa, dan biaya asuransi?  
2.  Faktor-faktor apa saja yang paling signifikan dalam mempengaruhi variasi biaya kuliah (`Tuition_USD`) antar individu atau kasus?  
3.  Model *machine learning* manakah yang memberikan performa terbaik dalam memprediksi biaya kuliah dengan tingkat kesalahan (error) yang dapat diterima?

### Goals  
Untuk menjawab pertanyaan di atas, maka akan dijabarkan sebagai berikut:  
1.  Mengembangkan sebuah model prediktif \*machine learning\* yang mampu mengestimasi biaya kuliah (`Tuition_USD`) dengan akurasi yang baik.  
2.  Mengidentifikasi dan menganalisis fitur-fitur kunci yang memiliki kontribusi paling besar terhadap perbedaan biaya kuliah.  
3.  Mengevaluasi beberapa algoritma *machine learning* regresi dan memilih model yang paling sesuai dan memberikan performa terbaik untuk kasus prediksi biaya kuliah ini, berdasarkan metrik evaluasi seperti *Mean Absolute Error (MAE)*.

### Solution Statements  
Solusi yang diimplementasikan dalam notebook \`MLT\_Wahyu\_Final.ipynb\` untuk memenuhi tujuan proyek ini meliputi:  
1.  **Analisis Data Eksploratif (EDA):** Memahami distribusi data, mengidentifikasi outlier, melihat korelasi antar fitur, dan hubungan antara fitur kategorikal dengan variabel target.  
2.  **Pra-pemrosesan Data Komprehensif:**  
    * Penanganan kolom dengan kardinalitas sangat tinggi (seperti `university_name`, `city`, `program_name`) dengan cara menghapusnya untuk menyederhanakan model awal.  
    * Imputasi nilai yang hilang menggunakan strategi median untuk fitur numerik dan modus untuk fitur kategorikal.  
    * One-Hot Encoding untuk fitur-fitur kategorikal (`Country`, `Level`, `Degree_Type`, `Accommodation_Type`, `Scholarship_Availability`) agar dapat digunakan oleh model.  
    * Standardisasi fitur-fitur numerik (`Duration_Years`, `Rent_USD`, `Visa_Fee_USD`, `Insurance_USD`, `Exchange_Rate`) menggunakan `StandardScaler` agar memiliki skala yang sebanding.  
3.  **Pengembangan Model *Machine Learning***: Membuat dan mengevaluasi 5 model Machine Learning regresi:  
    * Linear Regression  
    * K-Nearest Neighbors (KNN) Regressor  
    * Support Vector Regression (SVR)  
    * Random Forest Regressor  
    * Gradient Boosting Regressor  
4.  **Evaluasi Model**: Menggunakan metrik *Mean Absolute Error (MAE)*, *Mean Squared Error (MSE)*, *Root Mean Squared Error (RMSE)*, dan *R-squared (R2 Score)* untuk membandingkan performa model pada data uji. Prediksi dan target dikembalikan ke skala aslinya sebelum perhitungan metrik jika transformasi log diterapkan.

## Data Understanding
Dataset yang digunakan pada proyek ini adalah `International_Education_Costs.csv`. Berdasarkan analisis notebook, variabel-variabel utama yang digunakan setelah pra-pemrosesan awal adalah:

* **Target Variable:**  
    * `Tuition_USD` (Biaya kuliah dalam USD, ditransformasi log menjadi `Tuition_USD_log` untuk pemodelan).  
* **Numerical Features (Contoh yang Digunakan dalam Pemodelan):**  
    * `Duration_Years`  
    * `Rent_USD`  
    * `Visa_Fee_USD`  
    * `Insurance_USD`  
    * `Exchange_Rate`  
    * (Fitur biaya hidup lainnya seperti `living_expenses_per_month_usd`, `food_per_month_usd`, dll. juga relevan dan kemungkinan digunakan jika ada di `NUMERICAL_FEATURES` yang final).  
* **Categorical Features (Contoh yang Digunakan dalam Pemodelan):**  
    * `Country`  
    * `Level` (Jenjang Pendidikan)  
    * `Degree_Type`  
    * `Accommodation_Type`  
    * `Scholarship_Availability`

### Exploratory Data Analysis (EDA) - Ringkasan Temuan  
Dari analisis notebook, beberapa temuan kunci dari EDA kemungkinan meliputi:  
* **Distribusi Target (`Tuition_USD`):** Awalnya menunjukkan kemiringan positif yang signifikan (right-skewed) dengan adanya outlier, yang umum untuk data biaya. Transformasi logaritmik berhasil membuat distribusi lebih simetris.  
* **Fitur Numerik:** Distribusi fitur numerik lain bervariasi; beberapa mungkin juga miring dan mendapat manfaat dari transformasi atau penskalaan.  
* **Fitur Kategorikal vs. Target:** Boxplot menunjukkan bahwa fitur-fitur kategorikal seperti `Country`, `Level`, dan `Degree_Type` memiliki pengaruh terhadap median dan sebaran `Tuition_USD`.  
* **Korelasi:** Matriks korelasi menunjukkan hubungan linear antar fitur numerik. Fitur-fitur biaya hidup komponen (`Rent_USD`, dll.) kemungkinan berkorelasi dengan `total_expenses_per_month_usd` jika fitur tersebut digunakan. `Exchange_Rate` mungkin menunjukkan korelasi tertentu tergantung pada bagaimana biaya dalam USD dihitung atau dilaporkan.

## Data Preparation
Tahapan persiapan data yang diimplementasikan dalam notebook:  
1.  **Penghapusan Kolom Kardinalitas Tinggi:** Kolom seperti `city`, `university_name`, dan `program_name` (jika memiliki terlalu banyak nilai unik) dihapus untuk menyederhanakan model awal dan menghindari *curse of dimensionality* dari one-hot encoding yang sangat lebar.    
2.  **Pemisahan Fitur (X) dan Target (y):** Dataset dipisahkan menjadi fitur input dan variabel target (`Tuition_USD_log`).  
3.  **Pembagian Data Latih dan Uji:** Data dibagi menjadi set pelatihan (80%) dan set pengujian (20%) menggunakan `train_test_split` dengan `random_state` untuk reproduktifitas.  
4.  **Pra-pemrosesan dengan Pipeline dan ColumnTransformer:**  
    * **Untuk Fitur Numerik (`Duration_Years`, `Rent_USD`, `Visa_Fee_USD`, `Insurance_USD`, `Exchange_Rate`, dll.):**  
        * Imputasi nilai yang hilang menggunakan strategi median (`SimpleImputer(strategy='median')`).  
        * Standardisasi fitur menggunakan `StandardScaler()`.  
    * **Untuk Fitur Kategorikal (`Country`, `Level`, `Degree_Type`, dll.):**  
        * Imputasi nilai yang hilang menggunakan strategi modus (`SimpleImputer(strategy='most_frequent')`).  
        * Encoding menggunakan `OneHotEncoder(handle_unknown='ignore', drop='first')`.  
    * Kolom yang tidak termasuk dalam fitur numerik atau kategorikal yang didefinisikan akan di-drop (`remainder='drop'`).

## Modeling
Lima model regresi machine learning dikembangkan dan dilatih menggunakan data latih yang telah dipersiapkan. Setiap model digabungkan dengan pipeline pra-pemrosesan untuk memastikan konsistensi.

1.  **Linear Regression:**  
    * Deskripsi: Model statistik dasar yang mencari hubungan linear antara fitur dan target.  
    * Kekuatan: Mudah diinterpretasikan, komputasi cepat.  
    * Kelemahan: Mengasumsikan linearitas, sensitif terhadap outlier, mungkin tidak menangkap pola kompleks.  
2.  **K-Nearest Neighbors (KNN) Regressor:**  
    * Deskripsi: Algoritma non-parametrik yang memprediksi nilai target berdasarkan rata-rata dari 'k' tetangga terdekatnya di ruang fitur.  
    * Parameter Kunci: `n_neighbors=7` (sebagai contoh dalam notebook).  
    * Kekuatan: Sederhana, mampu menangkap hubungan non-linear lokal.  
    * Kelemahan: Komputasi bisa mahal pada dataset besar, sensitif terhadap skala fitur dan fitur yang tidak relevan, performa bergantung pada 'k'.  
3.  **Support Vector Regression (SVR):**  
    * Deskripsi: Adaptasi dari Support Vector Machines untuk regresi, bertujuan menemukan fungsi yang memiliki deviasi paling banyak ε dari target aktual.  
    * Parameter Kunci: `kernel='rbf'`, `C=100`, `gamma='scale'` (sebagai contoh dalam notebook).  
    * Kekuatan: Efektif di ruang dimensi tinggi, baik untuk data non-linear dengan kernel yang tepat.  
    * Kelemahan: Membutuhkan tuning parameter, komputasi bisa intensif.  
4.  **Random Forest Regressor:**  
    * Deskripsi: Metode *ensemble learning* yang terdiri dari banyak *decision tree*. Prediksi adalah rata-rata dari prediksi semua pohon.  
    * Parameter Kunci: `n_estimators=100`, `max_depth=10`, `random_state=42` (sebagai contoh).  
    * Kekuatan: Kuat terhadap overfitting (jika di-tune), menangani non-linearitas dan interaksi fitur, memberikan estimasi pentingnya fitur.  
    * Kelemahan: Kurang interpretatif dibandingkan model tunggal.  
5.  **Gradient Boosting Regressor:**  
    * Deskripsi: Teknik *ensemble learning* yang membangun model (biasanya *decision tree*) secara sekuensial, di mana setiap model baru dilatih untuk memperbaiki kesalahan model sebelumnya.  
    * Parameter Kunci: `n_estimators=100`, `learning_rate=0.1`, `max_depth=5`, `random_state=42` (sebagai contoh).  
    * Kekuatan: Seringkali memberikan akurasi prediksi yang sangat tinggi, fleksibel.  
    * Kelemahan: Rentan overfitting jika tidak di-tune, training bisa lebih lama.

## Evaluation
Metrik evaluasi yang digunakan untuk menilai performa model pada data uji adalah *Mean Absolute Error (MAE)*, *Mean Squared Error (MSE)*, *Root Mean Squared Error (RMSE)*, dan *R-squared (R2 Score)*. Prediksi dan nilai target asli (sebelum transformasi log) dikembalikan ke skala semula menggunakan `np.expm1` sebelum menghitung metrik.

**Tabel Perbandingan Performa Model (Contoh - Isi dengan hasil aktual dari notebook Anda):**

| Model                     | MAE           | MSE                  | RMSE          | R2 Score     |
| :------------------------ | :------------ | :------------------- | :------------ | :----------- |
| Linear Regression         | 112,228.04    | 311,845,662,566.01   | 558,431.43    | -1194.4316   |
| KNN Regressor             | 3,076.88      | 24,562,926.62        | 4,956.10      | 0.9058       |
| SVR                       | 11,540.82     | 228,214,764.28       | 15,106.78     | 0.1252       |
| Random Forest Regressor   | **2,436.45** | **13,517,270.29** | **3,676.58** | **0.9482** |
| Gradient Boosting Regressor| 2,514.74      | 14,096,153.48        | 3,754.48      | 0.9460       |


**Analisis Performa Model:**  
* **Model Terbaik:** Berdasarkan metrik (misalnya, MAE terendah dan R2 Score tertinggi pada data uji), model *[Nama Model Terbaik, misal: Random Forest Regressor atau Gradient Boosting Regressor]* menunjukkan performa terbaik.  
* **Performa Linear Regression:** Seperti yang telah dianalisis sebelumnya, **Linear Regression** kemungkinan besar menunjukkan MAE yang jauh lebih tinggi (performa lebih buruk) dibandingkan model lainnya. Hal ini disebabkan oleh:  
    1.  **Asumsi Linearitas:** Data biaya pendidikan internasional cenderung memiliki hubungan non-linear yang kompleks dengan fitur-fiturnya, yang tidak dapat ditangkap dengan baik oleh model linear.  
    2.  **Sensitivitas terhadap Outlier:** Meskipun transformasi log membantu, Linear Regression tetap lebih rentan terhadap sisa outlier dibandingkan model berbasis pohon.  
    3.  **Interaksi Fitur:** Linear Regression tidak secara otomatis menangani interaksi antar fitur, sedangkan model seperti Random Forest dan Gradient Boosting dapat melakukannya.  
* **Model Ensemble (Random Forest & Gradient Boosting):** Model-model ini seringkali unggul karena kemampuannya memodelkan hubungan non-linear yang kompleks, menangani interaksi fitur, dan lebih robust.  
* **SVR dan KNN:** Performanya akan sangat bergantung pada pemilihan parameter dan sifat data. KNN bisa baik untuk pola lokal tetapi mungkin kesulitan dengan dimensi tinggi setelah one-hot encoding. SVR dengan kernel RBF bisa fleksibel tetapi membutuhkan tuning.

**Visualisasi Model Terbaik:**  
Sebuah scatter plot yang membandingkan nilai prediksi model terbaik dengan nilai aktual `Tuition_USD` pada data uji (dalam skala aslinya) membantu memvisualisasikan sebaran error dan seberapa baik model mengikuti garis ideal y=x.

## Kesimpulan
Proyek ini berhasil mengembangkan dan mengevaluasi beberapa model machine learning untuk memprediksi biaya kuliah internasional. Model *[Random Forest Regressor]* menunjukkan performa paling menjanjikan dengan MAE sebesar *[2,436.45]* dan R2 Score sebesar *[N0.9482]*. Analisis menunjukkan bahwa hubungan dalam data bersifat non-linear, yang menjelaskan mengapa model yang lebih fleksibel seperti Random Forest atau Gradient Boosting mengungguli Linear Regression. Langkah selanjutnya dapat mencakup *hyperparameter tuning* lebih lanjut untuk model terbaik, eksplorasi *feature engineering* yang lebih canggih (terutama untuk menangani fitur kardinalitas tinggi yang di-drop), atau penggunaan model ensemble yang lebih advance.

## Langkah selanjutnya yang dapat dipertimbangkan meliputi:
* *Hyperparameter tuning* lebih lanjut untuk Random Forest atau Gradient Boosting.
* Eksplorasi *feature engineering* yang lebih canggih, termasuk cara yang lebih baik untuk menangani fitur dengan kardinalitas tinggi yang sebelumnya di-drop.
* Mencoba model *ensemble* yang lebih *advanced* seperti XGBoost atau LightGBM.
* Analisis *feature importance* dari model terbaik untuk mendapatkan wawasan lebih dalam mengenai faktor pendorong biaya.
