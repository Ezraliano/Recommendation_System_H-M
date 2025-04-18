# Laporan Proyek Machine Learning - Ezraliano Sachio Krisnadiva

## Project Overview
Sistem rekomendasi telah menjadi komponen penting dalam industri bisnis fashion untuk meningkatkan pengalaman pengguna dan mendorong penjualan. Proyek ini bertujuan untuk membangun sistem rekomendasi berbasis konten untuk produk H&M, menggunakan dataset yang berisi informasi produk seperti harga, warna, kategori, bahan, dan deskripsi. Dengan memanfaatkan teknik machine learning, proyek ini mengembangkan model yang merekomendasikan produk serupa berdasarkan fitur produk, memberikan solusi yang relevan bagi pengguna yang mencari barang dengan karakteristik tertentu.

Mengapa Proyek Ini Penting?
Sistem rekomendasi merupakan elemen kunci dalam strategi bisnis fashion. Menurut penelitian oleh McKinsey, sistem rekomendasi yang efektif dapat meningkatkan pendapatan hingga 30% melalui personalisasi pengalaman belanja. Proyek ini relevan karena membantu H&M menawarkan rekomendasi produk yang sesuai dengan preferensi pelanggan, sehingga meningkatkan kepuasan dan loyalitas pelanggan.

Refrensi :
- "Recommender Systems Handbook" oleh Ricci, F., Rokach, L., & Shapira, B. (2015). Springer.
-  McKinsey & Company. (2013). How Retailers Can Keep Up with Consumers.


## Business Understanding

### Problem Statements
1. Kurangnya rekomendasi produk yang relevan: Pelanggan sering kesulitan menemukan produk yang sesuai dengan preferensi mereka berdasarkan warna, kategori, atau bahan, yang dapat mengurangi kepuasan belanja online.
2. Data produk yang kompleks: Dataset H&M memiliki berbagai fitur seperti deskripsi teks, harga, dan kategori, yang memerlukan pemrosesan khusus untuk menghasilkan rekomendasi yang bermakna.


### Goals
1. Membangun sistem rekomendasi berbasis konten yang dapat menyarankan produk serupa berdasarkan fitur seperti warna, kategori, merek, harga, deskripsi, dan bahan.
2. Menyediakan rekomendasi yang relevan dengan mempertimbangkan preferensi pelanggan, termasuk batasan harga, untuk meningkatkan pengalaman belanja dan konversi penjualan.


  ### Solution statements
   - Pendekatan 1: Content-Based Filtering dengan Cosine Similarity
Menggunakan fitur produk (kategori, warna, harga, deskripsi) yang diencode dan diubah menjadi vektor untuk menghitung kesamaan antar produk menggunakan cosine similarity. Pendekatan ini sederhana dan efektif untuk dataset dengan fitur yang kaya.

## Data Understanding
Dataset ini berasal pada situs kaggle : (https://www.kaggle.com/datasets/niharpatel03/h-and-m-product-dataset). Dataset ini berisi informasi produk H&M dengan total 9676 baris dan 14 kolom. Data ini mencakup informasi seperti ID produk, harga, warna, kategori, merek, deskripsi, dan bahan. Dataset memiliki beberapa missing values pada kolom colorshades, details, dan materials dan dataset ini memiliki duplikasi data pada kolom productId. Demi memahami dataset ini berikut merupakan penjelasan fitur/variabel yang ada pada dataset ini.

- productId: ID unik untuk setiap produk.
- price: Harga produk dalam dolar.
- colorName: Nama warna produk (misalnya, "Black", "Blue").
- mainCatCode: Kode kategori utama produk (misalnya, "Ladieswear", "Menswear").
- brandName: Nama merek produk.
- newArrival: Status apakah produk baru (True/False).
- details: Deskripsi teks tentang produk.
- materials: Informasi bahan produk.
- colorShades: Nuansa warna produk.
- isOnline, comingSoon, stockState: Status ketersediaan produk.
- url : merupakan kolom url
- productName : nama produk
- colors : merupakan kode warna

Untuk memahami data secara mendalam, saya melakukan proses Univariate Explorasi Data Analysis agar dapat memahami fitur dataset secara mendalam. Berikut hasil Univariate Explorasi Data Analysis yang dilakukan.
1. Distribusi Harga: 
Harga produk bervariasi dari $5 hingga $200, dengan sebagian besar produk berada di kisaran $20–$50 (histogram dan boxplot).
2. Distribusi Kategorikal:
Kolom seperti colorName memiliki banyak kategori unik (>50), sedangkan mainCatCode memiliki beberapa kategori dominan seperti "Ladieswear".
Insight yang didapat adalah data memiliki distribusi harga yang miring ke kanan, dan beberapa kolom seperti isOnline bersifat konstan, sehingga tidak relevan untuk model.

## Data Preparation
Dalam membuat model machine learning, diperlukan sebuah proses data preparation agar model machine learning dapat berjalan dengan optimal, berikut merupakan tahap data preparation.
1. Penanganan Missing Values
Proses :
- Kolom details dan materials diisi dengan string kosong ('') menggunakan fillna('').
- Kolom colorShades diisi dengan "Unknown" menggunakan fillna('Unknown').
- Verifikasi dilakukan dengan isnull().sum() untuk memastikan tidak ada nilai hilang.
Alasan :
- details dan materials adalah kolom teks untuk analisis TF-IDF, sehingga string kosong mencegah error saat ekstraksi fitur.
- colorShades adalah kategorikal; "Unknown" memberikan nilai default yang logis tanpa mengubah distribusi.
- Penanganan ini memastikan semua baris dapat digunakan tanpa menghapus data.

2. Penghapusan Duplikasi
Proses :
- Memeriksa duplikasi dengan hm1['productId'].duplicated().sum().
- Menghapus duplikasi berdasarkan productId menggunakan drop_duplicates(subset=['productId'], keep='first').
Alasan :
- Duplikasi productId dapat menyebabkan bias dalam rekomendasi karena produk yang sama muncul berulang.
- Menyimpan baris pertama memastikan data tetap representatif.

3. Penghapusan Kolom Tidak Relevan
Proses :
- Kolom Unnamed: 0, url, dan productName dimasukkan ke daftar cols_to_drop.
- Memeriksa kolom konstan (isOnline, comingSoon, stockState) dengan nunique() == 1, lalu menambahkannya ke cols_to_drop.
- Menghapus kolom dengan drop(columns=cols_to_drop, errors='ignore').
Alasan :
- Unnamed: 0 adalah indeks tambahan, url tidak relevan untuk kesamaan, dan productName redundan dengan details.
- Kolom konstan tidak memberikan informasi tambahan untuk rekomendasi.
- Penghapusan ini mengurangi dimensi data, mempercepat pemrosesan.

4. Encoding Variabel Kategorikal
Proses :
- Memilih kolom kategorikal: colorName, mainCatCode, brandName, colorShades, newArrival.
- Menggunakan OneHotEncoder(sparse_output=False, handle_unknown='ignore') untuk mengubah kategori menjadi kolom biner.
- Hasil encoding disimpan dalam encoded_cats_df dengan nama kolom dari encoder.get_feature_names_out().
- Menggabungkan hasil encoding dengan dataset dan menghapus kolom kategorikal asli.
Alasan :
- One-Hot Encoding cocok untuk data kategorikal tanpa urutan (misalnya, warna atau merek).
- sparse_output=False memudahkan penggabungan dengan DataFrame.
- handle_unknown='ignore' mencegah error jika kategori baru muncul di data uji.
- Penghapusan kolom asli mencegah redundansi.

5. Standarisasi Kolom Numerik
Proses :
- Menggunakan StandardScaler untuk menstandarisasi kolom price, menghasilkan price_scaled dengan rata-rata 0 dan standar deviasi 1.
- Menambahkan price_scaled ke dataset.
Alasan :
- Standarisasi memastikan fitur numerik (harga) memiliki skala seragam, penting untuk algoritma berbasis jarak seperti cosine similarity.
- Hanya price yang distandarisasi karena ini satu-satunya fitur numerik.

6. Feature Engineering
Proses :
- Membuat price_segment dengan pd.cut dan mengelompokannya menjadi (Murah (≤$30), Menengah ($30–$50), Premium (>$50))
- Menggunakan TfidfVectorizer(max_features=100, stop_words='english') untuk mengekstrak fitur dari details dan materials.
- Hasil TF-IDF disimpan dalam tfidf_details_df dan tfidf_materials_df, dengan nama kolom seperti details_cotton atau materials_denim.
- Menggabungkan fitur baru dengan dataset.
Alasan :
- price_segment menambah konteks harga untuk analisis.
- TF-IDF mengubah teks menjadi vektor numerik berdasarkan pentingnya kata, meningkatkan relevansi rekomendasi.
- max_features=100 membatasi dimensi untuk efisiensi; stop_words='english' menghapus kata umum (misalnya, "the").
- Fitur ini memperkaya representasi produk.

7. Penghapusan Kolom Redundan
Proses :
- Menghapus kolom colors dengan drop(columns=['colors'], errors='ignore').
Alasan :
- colors redundan dengan colorName, yang sudah diencode.

8. Penggabungan Fitur
Proses :
- Menggabungkan productId, price, price_scaled, fitur kategorikal (dari One-Hot Encoding), dan fitur TF-IDF menggunakan pd.concat.
- Memastikan indeks selaras dengan reset_index(drop=True).
Alasan :
- Penggabungan menciptakan dataset lengkap untuk modeling.
- Menyimpan productId dan price memungkinkan pelacakan dan filtering harga.


## Modeling
Model dari Sitem rekomendasi ini menggunakan pendekatan Content Based Filtering dengan cosine similarity, dikarenakan tidak adanya data pengguna dan rating sehingga jika menerapkan model collaborative maka sistem rekomendasi tidak dapat berjalan dengan maksimal. Untuk memahami proses modeling, berikut merupakan tahap dari proses modeling :

1. Seleksi Fitur
Proses :
- Memilih semua kolom dari hm1_encoded kecuali productId dan price (misalnya, price_scaled, fitur One-Hot Encoding, fitur TF-IDF).
- Mengubah fitur menjadi matriks NumPy dengan hm1_encoded[feature_cols].values.
Alasan :
- Fitur ini mencakup semua informasi relevan (harga, warna, kategori, teks) untuk menghitung kesamaan.
- productId dan price dikecualikan karena hanya digunakan untuk pelacakan dan filtering.

2. Perhitungan Cosine Similarity
Proses :
- Menggunakan cosine_similarity dari scikit-learn untuk menghitung kesamaan antar produk, menghasilkan matriks similarity_matrix.
- Setiap elemen dalam matriks menunjukkan skor kesamaan (0 hingga 1) antar pasangan produk.
Alasan :
- Cosine similarity mengukur sudut antara vektor fitur, efektif untuk data berdimensi tinggi seperti TF-IDF.
- Skor 1 menunjukkan produk identik, skor 0 menunjukkan tidak ada kesamaan.

3. Fungsi Rekomendasi
Proses :
- Membuat fungsi get_recommendations(product_id, num_recommendations=5, price_range=None). 
- Mencari indeks produk berdasarkan productId.
- Mengambil skor kesamaan dari similarity_matrix.
- Menyaring produk dalam price_range.
- Mengurutkan skor kesamaan, mengambil num_recommendations produk teratas.
- Mengembalikan informasi produk dari hm1_info (ID, harga, warna, kategori, merek, status baru).
Alasan :
- Fungsi ini dapat fleksibel memungkinkan rekomendasi dengan atau tanpa batasan harga.
- Menggunakan hm1_info memastikan output mudah dibaca oleh pengguna.

4. Contoh Output
Proses :
- Menguji fungsi dengan tiga productId pertama, menggunakan price_range=10.
- Menampilkan top 5 rekomendasi untuk setiap produk.
![Rekomendasi Top 5](Rekomendasi_Top_N.png)

## Evaluation
Untuk evaluasi dilakukan dengan secara kualitatif dengan menganalisis relevansi rekomendasi berdasarkan kategori, warna, harga, dan skor kesamaan. Berikut merupakan detail dari tahapan proses evaluasi yang telah dilakukan :
1. Metrik Evaluasi
- Relevansi Kategori: Persentase rekomendasi dengan mainCatCode sama dengan produk asli.
  - Formula: (jumlah rekomendasi dengan kategori sama) / total rekomendasi.
  - Cara Kerja: Mengukur apakah model memprioritaskan produk dalam kategori yang sama (misalnya, "Ladieswear").
- Relevansi Warna: Jumlah rekomendasi dengan warna serupa (dihitung dengan str.contains untuk pencocokan string).
  - Formula: (jumlah rekomendasi dengan warna serupa) / total rekomendasi.
  - Cara Kerja: Mengevaluasi apakah warna produk asli (misalnya, "Blue") muncul di rekomendasi.
- Relevansi Harga: Rentang harga rekomendasi dibandingkan dengan harga produk asli.
  - Formula: (min(harga rekomendasi), max(harga rekomendasi)).
  - Cara Kerja: Memastikan harga rekomendasi berada dalam price_range ($10).
- Distribusi Skor Kesamaan: Histogram skor kesamaan dari similarity_matrix.
  - Formula: Distribusi frekuensi skor kesamaan (0 hingga 1).
  - Cara Kerja: Menganalisis apakah model membedakan produk serupa (skor tinggi) dan tidak serupa (skor rendah).

2. Proses Evaluasi
- Pemilihan Sampel:
  - Memilih tiga produk dari kategori berbeda menggunakan hm1_info.groupby('mainCatCode').head(1).
  - Contoh: Produk dari "Ladieswear", "Menswear", dan "Kidswear".
- Analisis Rekomendasi:
  - Untuk setiap produk, menampilkan (Informasi produk asli (productId, price, colorName, mainCatCode, brandName, newArrival, details, materials)).
  - Top-5 rekomendasi dengan price_range=10, termasuk skor kesamaan.
- Perhitungan :
  - Jumlah rekomendasi dengan kategori sama.
  - Jumlah rekomendasi dengan warna serupa.
  - Rentang harga rekomendasi.
  - Rentang skor kesamaan.
- Distribusi Skor Kesamaan:
  - Membuat histogram dengan sns.histplot(similarity_matrix.flatten(), bins=50, kde=True) untuk melihat distribusi skor kesamaan.

3. Hasil Evaluasi
- Relevansi Kategori:
  - Rata-rata 3–4 dari 5 rekomendasi berada dalam kategori yang sama (misalnya, "Ladieswear").
  - Interpretasi: Fitur mainCatCode memiliki bobot tinggi dalam perhitungan kesamaan.
- Relevansi Warna:
  - Sekitar 2–3 dari 5 rekomendasi memiliki warna serupa (misalnya, "Blue" atau "Navy" untuk produk "Blue").
  - Interpretasi: Fitur colorName relevan, tetapi variasi warna menambah keragaman rekomendasi.
- Relevansi Harga:
  - Harga rekomendasi berada dalam kisaran ±$10 dari produk asli (misalnya, produk $30 menghasilkan rekomendasi $20–$40).
  - Interpretasi: Filter price_range berfungsi dengan baik.
- Distribusi Skor Kesamaan:
  - Histogram menunjukkan distribusi skor dari 0 hingga 1, dengan puncak di 0.2–0.4.
  - Interpretasi: Model membedakan produk serupa (skor >0.5) dan tidak serupa (skor <0.2), dengan sebagian besar produk memiliki kesamaan sedang.
- Kontribusi TF-IDF:
  - Fitur TF-IDF dari details dan materials meningkatkan relevansi untuk produk dengan kata kunci serupa (sebagai contoh, "cotton" atau "denim").
  - Contoh: Produk dengan materials="cotton" cenderung merekomendasikan produk berbahan serupa.

Kesimpulan :
1. Model berhasil memberikan rekomendasi yang relevan berdasarkan kategori, warna, dan harga.
2. Fitur TF-IDF dari details dan materials memperkaya kesamaan, terutama untuk produk dengan deskripsi atau bahan spesifik.
3. Filter price_range memastikan rekomendasi sesuai dengan anggaran pengguna.
4. Kelemahan: Model tidak mempertimbangkan preferensi pengguna (contoh, riwayat pembelian), yang dapat ditingkatkan dengan model collaborative.
5. Saran: Dapat menambahkan data preferensi pengguna dan rating agar model berbasis collaborative dapat dilaksanakan.







