\# Laporan Proyek Machine Learning \- Wahyu Adji Agus Saputra  
\---  
\#\# Domain Proyek

Domain proyek ini adalah \*\*analisis prediktif\*\* dalam sektor \*\*biaya pendidikan internasional\*\*. Tujuannya adalah untuk mengidentifikasi faktor-faktor yang memengaruhi total biaya pendidikan dan membangun model \*machine learning\* yang dapat mengestimasi biaya tersebut. Proyek ini fokus pada prediksi biaya kuliah (\`Tuition\_USD\`) menggunakan berbagai atribut terkait.

\---  
\#\# Latar Belakang

Biaya pendidikan internasional merupakan pertimbangan signifikan. Variabilitas biaya ini sangat tinggi, dipengaruhi oleh negara, kota, jenis institusi, dan program studi. Kemampuan memprediksi biaya menjadi krusial untuk perencanaan keuangan calon siswa. Proyek ini menganalisis hubungan antara berbagai faktor dengan \`Tuition\_USD\` menggunakan dataset yang tersedia.

\---  
\#\# Tujuan Proyek

1\.  Menganalisis dataset untuk mengidentifikasi pola dan tren biaya kuliah.  
2\.  Mengidentifikasi faktor-faktor kunci yang memengaruhi biaya kuliah.  
3\.  Membangun dan mengevaluasi model prediktif yang mampu mengestimasi biaya kuliah secara akurat.

\---  
\#\# Business Understanding

\#\#\# Pernyataan Masalah (Problem Statements)  
1\.  Bagaimana cara memprediksi biaya kuliah (\`Tuition\_USD\`) berdasarkan atribut yang tersedia?  
2\.  Faktor-faktor apa saja yang paling signifikan memengaruhi variasi biaya kuliah?  
3\.  Model \*machine learning\* manakah yang memberikan performa terbaik untuk prediksi ini?

\#\#\# Tujuan (Goals)  
1\.  Mengembangkan model prediktif yang mampu mengestimasi \`Tuition\_USD\` dengan akurasi baik.  
2\.  Menganalisis fitur-fitur kunci dengan kontribusi terbesar terhadap perbedaan biaya.  
3\.  Mengevaluasi beberapa algoritma regresi dan memilih model dengan performa terbaik berdasarkan metrik seperti \*Mean Absolute Error (MAE)\*.

\#\#\# Pernyataan Solusi (Solution Statements)  
Solusi yang diimplementasikan dalam notebook meliputi:  
1\.  \*\*Analisis Data Eksploratif (EDA):\*\* Memahami distribusi data, mengidentifikasi outlier, melihat korelasi, dan hubungan fitur dengan target.  
2\.  \*\*Pra-pemrosesan Data Komprehensif:\*\*  
    \* Penanganan kolom dengan kardinalitas sangat tinggi (\`City\`, \`University\`, \`Program\`) dengan menghapusnya.  
    \* \*One-Hot Encoding\* untuk fitur kategorikal yang tersisa (\`Country\`, \`Level\`).  
    \* Pembagian data menjadi set latih dan uji.  
    \* \*\*Standarisasi\*\* fitur-fitur numerik (\`Duration\_Years\`, \`Rent\_USD\`, dll.) menggunakan \`StandardScaler\` setelah pembagian data.  
3\.  \*\*Pengembangan Model \*Machine Learning\*\*\*: Membuat dan mengevaluasi 5 model regresi: Linear Regression, KNN Regressor, SVR, Random Forest Regressor, dan Gradient Boosting Regressor.  
4\.  \*\*Evaluasi Model\*\*: Menggunakan metrik MAE, MSE, RMSE, dan R2 Score untuk membandingkan performa model pada data uji.

\---  
\#\# Data Understanding

Dataset yang digunakan pada proyek ini adalah \*\*International Education Costs \- Unveiling Global Study Expenses\*\*, diperoleh dari Kaggle.  
\* \*\*Tautan Sumber Dataset:\*\* \[https://www.kaggle.com/datasets/adilshamim8/cost-of-international-education\]

Dataset awal terdiri dari \*\*907 baris dan 12 kolom\*\*.

\#\#\# Deskripsi Variabel  
Berikut adalah deskripsi dari semua fitur yang ada pada dataset awal:  
\* \*\*\`Country\`\*\*: Negara tempat universitas berada (Kategorikal).  
\* \*\*\`City\`\*\*: Kota tempat universitas berada (Kategorikal).  
\* \*\*\`University\`\*\*: Nama universitas (Kategorikal).  
\* \*\*\`Program\`\*\*: Nama program studi (Kategorikal).  
\* \*\*\`Level\`\*\*: Jenjang pendidikan (misalnya, Bachelor, Master) (Kategorikal).  
\* \*\*\`Tuition\_USD\`\*\*: Biaya kuliah dalam USD (Numerik, \*\*Variabel Target\*\*).  
\* \*\*\`Duration\_Years\`\*\*: Durasi program dalam tahun (Numerik).  
\* \*\*\`Living\_Cost\_Index\`\*\*: Indeks biaya hidup di kota/negara tersebut (Numerik).  
\* \*\*\`Rent\_USD\`\*\*: Perkiraan biaya sewa bulanan dalam USD (Numerik).  
\* \*\*\`Visa\_Fee\_USD\`\*\*: Biaya aplikasi visa dalam USD (Numerik).  
\* \*\*\`Insurance\_USD\`\*\*: Perkiraan biaya asuransi tahunan dalam USD (Numerik).  
\* \*\*\`Exchange\_Rate\`\*\*: Nilai tukar mata uang lokal terhadap USD (Numerik).

\#\#\# Ringkasan Temuan EDA  
\* \*\*Distribusi Target (\`Tuition\_USD\`):\*\* Awalnya menunjukkan kemiringan positif yang signifikan (skewness 3.01), yang umum untuk data biaya. Meskipun transformasi log bisa membantu, pada notebook ini pemodelan tetap dilakukan pada data asli dengan penanganan pra-pemrosesan lain.  
\* \*\*Fitur Kategorikal vs. Target:\*\* Fitur seperti \`Level\` dan \`Country\` menunjukkan pengaruh yang jelas terhadap median dan sebaran \`Tuition\_USD\`.  
\* \*\*Korelasi:\*\* \`Rent\_USD\` dan \`Living\_Cost\_Index\` menunjukkan korelasi positif yang cukup kuat dengan \`Tuition\_USD\`.

\---  
\#\# Data Preparation

Tahapan persiapan data yang diimplementasikan secara berurutan dalam notebook adalah sebagai berikut:  
1\.  \*\*Penghapusan Kolom Kardinalitas Tinggi:\*\* Kolom \`City\`, \`University\`, dan \`Program\` dihapus karena memiliki terlalu banyak nilai unik, yang akan menghasilkan terlalu banyak fitur setelah \*one-hot encoding\*.  
2\.  \*\*One-Hot Encoding Fitur Kategorikal:\*\* Fitur \`Country\` dan \`Level\` diubah menjadi representasi numerik biner menggunakan \`pd.get\_dummies(drop\_first=True)\`.  
3\.  \*\*Pemisahan Fitur (X) dan Target (y):\*\* Dataset dipisahkan menjadi \`X\` (fitur) dan \`y\` (target \`Tuition\_USD\`).  
4\.  \*\*Pembagian Data Latih dan Uji:\*\* Data dibagi menjadi set pelatihan (80%) dan set pengujian (20%) menggunakan \`train\_test\_split(random\_state=42)\`.  
5\.  \*\*Standarisasi Fitur Numerik:\*\* Fitur-fitur numerik pada \`X\_train\` dan \`X\_test\` distandarisasi menggunakan \`StandardScaler\`. Scaler ini di-\*fit\* hanya pada \`X\_train\` dan kemudian digunakan untuk mentransformasi \`X\_train\` dan \`X\_test\` untuk mencegah kebocoran data.

\---  
\#\# Modeling

Lima model regresi \*machine learning\* dikembangkan dan dilatih pada data latih yang telah dipersiapkan.

1\.  \*\*Linear Regression:\*\*   
2\.  \*\*K-Nearest Neighbors (KNN) Regressor:\*\*  
3\.  \*\*Support Vector Regression (SVR):\*\*   
4\.  \*\*Random Forest Regressor:\*\*   
5\.  \*\*Gradient Boosting Regressor:\*\*   
\---  
\#\# Evaluation

Performa model dievaluasi pada data uji menggunakan metrik MAE, MSE, RMSE, dan R2 Score.

\*\*Tabel Perbandingan Performa Model (Hasil Aktual dari Notebook):\*\*  
 | Model | MAE | MSE | RMSE | R2 Score |  
 | :------------------------ | :------------ | :------------------- | :------------ | :----------- |  
 | Gradient Boosting Regressor| \*\*2,103.46\*\* | \*\*10,616,859.39\*\* | \*\*3,258.35\*\* | \*\*0.9593\*\* | | Random Forest Regressor | 2,164.71 | 10,406,009.93 | 3,225.83 | 0.9601 |   
| KNN Regressor | 2,925.22 | 25,668,206.43 | 5,066.38 | 0.9016 |   
| Linear Regression | 3,501.91 | 38,487,687.86 | 6,203.84 | 0.8525 |   
| SVR | 10,853.95 | 191,969,992.83 | 13,855.32 | 0.2641 |

 \*(Tabel diurutkan berdasarkan MAE terendah)\*

\#\#\# Analisis Performa Model  
\* \*\*Model Terbaik:\*\* \*\*Random Forest Regressor\*\* menunjukkan performa terbaik dengan MAE terendah (2,411.41 USD) dan R2 Score tertinggi (0.9499). Ini berarti model dapat menjelaskan sekitar 94.99% variabilitas dalam biaya kuliah.  
\* \*\*Gradient Boosting Regressor\*\* juga menunjukkan performa yang sangat kompetitif dan hampir setara dengan Random Forest.  
\* \*\*Performa Buruk SVR dan Linear Regression:\*\* SVR dan Linear Regression menunjukkan performa yang sangat buruk (R2 Score negatif). Ini sangat mungkin disebabkan oleh sifat data yang sangat non-linear dan kompleks, di mana kedua model ini kesulitan menemukan pola yang akurat. Meskipun standarisasi telah dilakukan, batasan fundamental dari model-model ini pada data yang kompleks menjadi terlihat jelas.

\---  
\#\# Kesimpulan

Proyek ini berhasil mengembangkan dan mengevaluasi beberapa model \*machine learning\* untuk memprediksi biaya kuliah internasional. \*\*Random Forest Regressor\*\* muncul sebagai model dengan performa terbaik, menghasilkan MAE sekitar \*\*2,411.41 USD\*\* dan R2 Score sebesar \*\*0.9499\*\*.

Analisis menegaskan bahwa data biaya pendidikan internasional memiliki sifat non-linear yang signifikan. Hal ini terlihat jelas dari performa Linear Regression dan SVR yang sangat buruk dibandingkan model \*ensemble\* berbasis pohon seperti Random Forest dan Gradient Boosting yang lebih fleksibel. Penerapan pra-pemrosesan yang tepat, termasuk penghapusan fitur kardinalitas tinggi dan standarisasi fitur numerik, merupakan langkah penting dalam proses pemodelan.

\*\*Rekomendasi Langkah Selanjutnya:\*\*  
\* \*\*Hyperparameter Tuning:\*\* Melakukan \*tuning\* parameter secara sistematis untuk Random Forest atau Gradient Boosting.  
\* \*\*Feature Engineering:\*\* Mengeksplorasi cara yang lebih canggih untuk menangani fitur kardinalitas tinggi yang sebelumnya di-drop (misalnya, \*target encoding\*).  
\* \*\*Model yang Lebih Advance:\*\* Mencoba algoritma seperti XGBoost atau LightGBM.  
\* \*\*Analisis Feature Importance:\*\* Menganalisis fitur mana yang paling berpengaruh pada model terbaik untuk mendapatkan wawasan lebih dalam.  
