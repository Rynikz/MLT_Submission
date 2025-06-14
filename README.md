# Laporan Proyek Machine Learning - Wahyu Adji Agus Saputra

## Project Overview

Di era digital saat ini, platform *streaming* film seperti Netflix, Disney+, dan lainnya menawarkan katalog konten yang sangat luas. MovieLens, sebagai layanan rekomendasi film, memiliki data yang mencakup puluhan ribu film. Jumlah pilihan yang sangat besar ini seringkali menimbulkan fenomena *information overload*, di mana pengguna kesulitan untuk menemukan konten yang sesuai dengan selera unik mereka. Pengguna bisa menghabiskan waktu berharga hanya untuk menjelajahi katalog, yang pada akhirnya dapat mengurangi kepuasan dan *engagement* mereka terhadap platform.

Proyek ini bertujuan untuk mengatasi masalah tersebut dengan membangun sebuah sistem rekomendasi film yang efektif. Sistem ini akan membantu pengguna menemukan film yang relevan dan menarik secara efisien, berdasarkan preferensi historis mereka dan juga pola perilaku dari pengguna lain yang memiliki selera serupa. Dengan menyediakan rekomendasi yang dipersonalisasi, kita dapat secara signifikan meningkatkan pengalaman pengguna, mendorong penemuan konten baru (*content discovery*), dan pada akhirnya meningkatkan retensi serta loyalitas pengguna.

**Referensi:**
Untuk mengakui penggunaan dataset dalam proyek ini, sitasi berikut digunakan sesuai dengan ketentuan dari penyedia data:
* F. Maxwell Harper and Joseph A. Konstan. 2015. The MovieLens Datasets: History and Context. *ACM Transactions on Interactive Intelligent Systems (TiiS) 5, 4*: 19:1–19:19. https://doi.org/10.1145/2827872

## Business Understanding

### Problem Statements

Berdasarkan latar belakang di atas, masalah yang ingin diselesaikan adalah:
-   Pengguna kesulitan menemukan film yang sesuai dengan selera pribadi mereka di tengah katalog yang berisi puluhan ribu judul film.
-   Platform streaming berisiko kehilangan *engagement* pengguna jika mereka tidak dapat menemukan konten yang menarik dengan cepat dan mudah.
-   Bagaimana cara memberikan rekomendasi yang tidak hanya akurat tetapi juga relevan secara personal untuk setiap dari ratusan ribu pengguna?

### Goals

Tujuan dari proyek ini adalah untuk menjawab permasalahan tersebut dengan:
-   Mengembangkan model machine learning yang dapat menghasilkan daftar film yang dipersonalisasi untuk setiap pengguna.
-   Membangun dua pendekatan sistem rekomendasi yang berbeda untuk memahami keunggulan dan keterbatasan dari masing-masing metode.
-   Mengevaluasi performa kedua model secara kuantitatif menggunakan metrik yang sesuai untuk memastikan kualitas dan akurasi rekomendasi.

### Solution statements
Untuk mencapai tujuan tersebut, dua pendekatan solusi akan dikembangkan dan dievaluasi:
1.  **Content-Based Filtering**: Mengajukan solusi rekomendasi dengan membuat model yang merekomendasikan film berdasarkan kemiripan atributnya (dalam kasus ini, genre). Jika pengguna menyukai sebuah film, sistem akan merekomendasikan film lain dengan genre serupa.
2.  **Collaborative Filtering**: Mengajukan solusi rekomendasi dengan membuat model yang merekomendasikan film berdasarkan pola rating dari pengguna lain yang memiliki selera serupa. Model ini mampu menemukan rekomendasi yang lebih beragam dan tidak terduga.

## Data Understanding
Dataset yang digunakan dalam proyek ini adalah **MovieLens 32M Dataset**. Dataset ini berisi 32,000,204 ratings pada 87,585 film oleh 200,948 pengguna. Data ini dibuat antara Januari 1995 dan Oktober 2023.

-   **Sumber Data**: Dataset dapat diunduh dari [MovieLens Latest Datasets](https://grouplens.org/datasets/movielens/32m/).

Dataset ini terdiri dari beberapa file, namun proyek ini fokus pada dua file utama: `movies.csv` dan `ratings.csv`.
-   **movies.csv**: Berisi informasi detail mengenai setiap film.
-   **ratings.csv**: Berisi informasi rating yang diberikan oleh pengguna terhadap film. Ukuran file ini sangat besar sehingga memerlukan teknik penanganan khusus.

**Variabel-variabel pada dataset:**
-   **movies.csv**:
    -   `movieId`: ID unik untuk setiap film.
    -   `title`: Judul film, termasuk tahun rilis dalam format `(YYYY)`.
    -   `genres`: Genre film, dipisahkan oleh karakter `|`.
-   **ratings.csv**:
    -   `userId`: ID unik untuk setiap pengguna.
    -   `movieId`: ID film yang diberi rating.
    -   `rating`: Rating yang diberikan pada skala 5 bintang, dengan kenaikan setengah bintang (0.5 - 5.0).
    -   `timestamp`: Waktu saat rating diberikan dalam format detik sejak 1 Januari 1970 (UTC).

Eksplorasi data awal menunjukkan bahwa data rating sangat besar dan *sparse* (tidak semua pengguna memberi rating untuk semua film). Hal ini menjadi justifikasi utama untuk melakukan filtering data secara agresif agar model dapat dilatih secara efisien.

## Data Preparation
Tahap persiapan data adalah langkah paling krusial dalam proyek ini karena ukuran data awal yang sangat besar (32 juta baris rating). Untuk mengatasi masalah keterbatasan memori (RAM), metode pemrosesan per-bagian (*chunking*) diterapkan.

**Alasan**: Metode ini diperlukan untuk memfilter dan membersihkan data tanpa harus memuat seluruh file ke dalam memori secara bersamaan, sehingga mencegah kegagalan program karena kehabisan RAM.

**Proses yang dilakukan secara berurutan:**
1.  **Pra-kalkulasi Jumlah Rating**: File `ratings.csv` dibaca per-bagian (chunk) untuk menghitung total rating yang diberikan oleh setiap pengguna dan yang diterima oleh setiap film.
2.  **Filtering Agresif**: Ditetapkan ambang batas yang sangat ketat:
    -   Pengguna aktif: Pengguna yang telah memberi **lebih dari 700 rating**.
    -   Film populer: Film yang telah menerima **lebih dari 500 rating**.
3.  **Filtering per-Bagian**: File `ratings.csv` dibaca ulang per-bagian. Setiap bagian langsung difilter berdasarkan kriteria di atas.
    ```python
    # Snippet logika filtering per-bagian
    filtered_chunks = []
    chunksize = 1000000
    for chunk in pd.read_csv(path_ratings, chunksize=chunksize):
        # Filter chunk berdasarkan pengguna aktif dan film populer
        filtered_chunk = chunk[(chunk['userId'].isin(active_users)) & (chunk['movieId'].isin(popular_movies))]
        filtered_chunks.append(filtered_chunk)
    ```
4.  **Penggabungan dan Pembersihan Akhir**: Bagian-bagian kecil yang sudah terfilter digabungkan menjadi satu DataFrame final. Hasilnya adalah DataFrame dengan **157,458** baris. Pada data yang sudah jauh lebih kecil inilah dilakukan pengecekan nilai `null` dan data duplikat.
5.  **Feature Engineering**: Pada data final yang bersih, dibuat fitur baru `year` dan `title_cleaned`.

## Modeling
Dua model sistem rekomendasi dikembangkan dengan pendekatan yang berbeda.

### Model 1: Content-Based Filtering
Model ini bekerja dengan merekomendasikan film yang memiliki kemiripan konten berdasarkan **genre**.

-   **Proses**: `TfidfVectorizer` digunakan untuk mengubah data teks genre menjadi vektor numerik. Selanjutnya, *Cosine Similarity* dihitung untuk menentukan skor kemiripan antara semua film.
-   **Hasil Rekomendasi (Input: 'Toy Story' dengan movieId=1):**

| movieId | title_cleaned        | genres                                      |
|:--------|:---------------------|:--------------------------------------------|
| 2355    | A Bug's Life      | Adventure\|Animation\|Children\|Comedy      |
| 3114    | Toy Story 2        | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| 4886    | Monsters, Inc.     | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| 58559   | WALL·E             | Adventure\|Animation\|Children\|Romance\|Sci-Fi |
| 6377    | Finding Nemo       | Adventure\|Animation\|Children\|Comedy      |
| 78499   | Toy Story 3        | Adventure\|Animation\|Children\|Comedy\|Fantasy |
| 89745   | Monsters University| Adventure\|Animation\|Comedy\|Fantasy     |
| 59501   | Kung Fu Panda     | Action\|Animation\|Children\|Comedy\|IMAX   |
| 62113   | The Incredibles    | Action\|Adventure\|Animation\|Children\|Comedy  |


### Model 2: Collaborative Filtering
Model ini merekomendasikan film berdasarkan kemiripan selera antar pengguna menggunakan algoritma **SVD (Singular Value Decomposition)** dari library `Surprise`.

-   **Proses**: Model SVD dilatih pada data interaksi `userId`, `movieId`, dan `rating` untuk mempelajari vektor laten (preferensi tersembunyi) dari pengguna dan film.
-   **Hasil Rekomendasi (Input: userId=50):**

  
| movieId | title_cleaned                | genres                               |
|:--------|:-----------------------------|:-------------------------------------|
| 318     | Shawshank Redemption, The  | Crime\|Drama                        |
| 858     | Godfather, The             | Crime\|Drama                        |
| 2959    | Fight Club                 | Action\|Crime\|Drama\|Thriller      |
| 1221    | Godfather: Part II, The    | Crime\|Drama                         |
| 4993    | Lord of the Rings: The Fellowship of the Ring, The | 'Adventure\|Fantasy |
| 5952    | Lord of the Rings: The Two Towers, The | 'Adventure\|Fantasy |
| 7153    | Lord of the Rings: The Return of the King, The | 'Action\|Adventure\|Drama\|Fantasy |
| 2028    | Saving Private Ryan        | Action\|Drama\|War                  |
| 2571    | Matrix, The                | Action\|Sci-Fi\|Thriller            |
| 58559   | WALL·E                     | Adventure\|Animation\|Children\|Romance\|Sci-Fi |

### Kelebihan dan Kekurangan Pendekatan
-   **Content-Based Filtering**:
    -   *Kelebihan*: Tidak memerlukan data pengguna lain (mengatasi masalah *user cold-start*) dan dapat merekomendasikan item yang tidak populer.
    -   *Kekurangan*: Terbatas pada fitur yang ada (kurang bisa memberikan rekomendasi yang mengejutkan) dan bisa menjadi terlalu spesifik (*over-specialization*).
-   **Collaborative Filtering**:
    -   *Kelebihan*: Mampu menemukan rekomendasi yang beragam dan mengejutkan (*serendipity*) tanpa memerlukan fitur item.
    -   *Kekurangan*: Mengalami masalah *item cold-start* (tidak bisa merekomendasikan item baru tanpa rating) dan membutuhkan data interaksi yang banyak.

## Evaluation
Performa model diukur secara kuantitatif menggunakan metrik yang sesuai untuk setiap pendekatan.

### Evaluasi Collaborative Filtering
-   **Metrik**: **RMSE (Root Mean Squared Error)**. Metrik ini mengukur rata-rata besarnya selisih (error) antara rating yang diprediksi oleh model dan rating aktual yang diberikan pengguna.
-   **Formula**:
    $$RMSE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(y_i - \hat{y}_i)^2}$$
    Di mana $n$ adalah jumlah rating, $y_i$ adalah rating aktual, dan $\hat{y}_i$ adalah rating hasil prediksi.
-   **Hasil**: Model dievaluasi menggunakan 5-fold cross-validation. Hasil rata-rata RMSE yang didapatkan adalah **0.8192**. Nilai ini menunjukkan bahwa secara rata-rata, prediksi rating dari model memiliki selisih sekitar 0.82 poin dari rating sebenarnya, sebuah hasil yang sangat baik dan akurat.

### Evaluasi Content-Based Filtering
-   **Metrik**: **Precision@10**. Metrik ini mengukur seberapa relevan 10 rekomendasi teratas yang diberikan. Metrik ini menghitung persentase item yang direkomendasikan yang ternyata benar-benar disukai oleh pengguna.
-   **Formula**:
    $$Precision@k = \frac{\text{Jumlah Rekomendasi Relevan yang Ditemukan}}{k}$$
-   **Hasil**: Setelah melalui pengujian dengan menyembunyikan sebagian data yang disukai pengguna dan mencoba "menebaknya" kembali, model mencapai **rata-rata Precision@10 sebesar 0.1540**. Ini berarti, dari 10 film yang direkomendasikan, sekitar **15.40%** terbukti relevan. Dalam konteks sistem rekomendasi dengan puluhan ribu pilihan film, skor ini terbilang bagus dan menunjukkan bahwa model memiliki kemampuan prediktif yang jauh lebih baik daripada acak.





