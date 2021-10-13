# Laporan Proyek Machine Learning - Kurnivan Noer Yusvianto

## Domain Proyek
Domain proyek yang dipilih dalam proyek _machine learning_ adalah bisnis dengan judul proyek "Prediksi Harga Ponsel Untuk Menghindari Penipuan".

- Latar Belakang

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/72246401/137119365-bb1eebe2-6f00-4cea-a183-0f25567cc7ce.png">
</p>

Pengguna smartphone Indonesia juga bertumbuh dengan pesat. _Smartphone_ (Ponsel) sudah menjadi kebutuhan primer. Pesatnya pertumbuhan _smartphone_ menjadi fenomena yang tidak bisa dihindari, karena masyarakat membutuhkan informasi dan dipakai juga untuk mengakses internet[[1](https://techno.okezone.com/read/2014/05/13/57/984293/di-indonesia-smartphone-sudah-menjadi-kebutuhan-utama)]. Akibatnya banyak penipuan dengan menjual _smartphone_ yang diatas harga rata-rata.

Dampak dari menjual _smartphone_ yang diatas harga rata-rata menyebabkan pembeli tertipu dengan _smartphone_ bekas yang memiliki spesifikasi yang dibawah rata-rata. Survei harga _smartphone_ bekas di pasaran, belum bisa menentukan apakah _smartphone_ bekas layak mendapatkan harga segitu, Maka dari itu diperlukan pengecekan spesifikasi _smartphone_ agar mendapatkan informasi detail dan dapat digunakan untuk menentukan harga kisarannya[[2](https://review.bukalapak.com/gadget/7-hal-yang-harus-diperhatikan-sebelum-membeli-smartphone-bekas-2292)]. Salah satunya pada proyek ini, dimana akan dibuat sebuah model _machine learning_ untuk mengklasifikasikan kategori _smartphone_ dalam biaya rendah sampai biaya paling tinggi. Dengan adanya model _machine learning_ ini, pembeli dapat mengecek spesifikasi _smartphone_ dan memperkirakan apakah _smartphone_ termasuk dalam kategori biaya rendah, biaya sedang, biaya tinggi, dan biaya sangat tinggi. Implementasi model ini dapat dijalankan pada aplikasi web atau android maupun ios.

## Business Understanding

### Problem Statements
Berdasarkan latar belakang diatas, ada beberapa rincian masalah yang dapat diselesaikan pada proyek ini sebagai berikut.
- Bagaimana cara  melakukan pra-pemrosesan data agar dapat digunakan dengan baik pada model _machine learning_?
- Bagaimana cara membuat model _machine learning_ untuk mengklasifikasikan kisaran harga ponsel?

### Goals
Tujuan dari dibuatnya proyek ini sebagai berikut.
- Melakukan pra-pemrosesan data agar dapat digunakan dengan baik pada model _machine learning_.
- membuat model _machine learning_ untuk mengklasifikasikan kisaran harga ponsel yang memiliki tingkat akurasi > 80%.

### Solution statements
Solusi yang dapat dilakukan untuk memenuhi tujuan dari proyek ini sebagai berikut.
- Untuk pra-pemrosesan dapat dilakukan beberapa teknik sebagai berikut.
    - Melakukan _Categorical Encoding_ yaitu proses mengubah data kategori menjadi data numerik dengan **_One-Hot Encoding_**
    - Melakukan **_data splitting_** berupa membagi dataset menjadi 2, yaitu data latih _(train data)_ dengan rasio 80% dan data test _(test data)_ dengan rasio 20%.
    - Melakukan **standardisasi data** pada fitur numerik dengan **StandarScaler**.
- Untuk pembuatan model menggunakan algoritma **_Support Vector Machine (SVM)_** sebagai model _baseline_. Algoritma tersebut dipilih karena dimensi data tinggi, jumlah data hasil observasi banyak, dan cocok untuk kasus klasifikasi. **_Support Vector Machine (SVM)_** digunakan untuk mencari _hyperplane_ terbaik dengan memaksimalkan jarak antar kelas. _Hyperplane_ adalah sebuah fungsi yang dapat digunakan untuk pemisah antar kelas. Tujuan dari algoritma SVM adalah untuk menemukan _hyperplane_ terbaik dalam ruang berdimensi-N (ruang dengan N-jumlah fitur) yang berfungsi sebagai pemisah yang jelas bagi titik-titik data input. Untuk proyek kami menggunakan SVM Klasifikasi non Linear. Cara kerja **_Support Vector Machine (SVM)_** Klasifikasi non Linear sebagai berikut.
    - Memuat data.
    - Transformasikan data menjadi ruang baru sehingga batas linier dapat digunakan untuk memisahkan tupel.
    - Untuk pemisahan data menggunakan beberapa fungsi kernel berikut.
        - RBF (Radial Basis Function) atau Gaussian kernel
        ![gaussian-kernel](https://user-images.githubusercontent.com/72246401/137120936-2bec6b2b-0df2-4a3b-a94b-e95a7c93560b.png)
        - Polinomial
        ![polynomial-kernel](https://user-images.githubusercontent.com/72246401/137120934-9b86cf3e-ec68-4b4b-affd-52674ab3031d.png)
        - Sigmoid
        ![sigmoid-kernel](https://user-images.githubusercontent.com/72246401/137120935-4c1b263b-69f7-4709-9708-2283a2b3a833.png)
    - Proses pembelajaran:
        - Fase _training_:
            - Minimize: <img src="https://user-images.githubusercontent.com/72246401/137123514-dc434933-7f57-4f2f-87c4-d696478111ab.png" width="48">
            - Target: <img src="https://user-images.githubusercontent.com/72246401/137123516-74d6939e-37fb-486a-83c5-c284eb1e49f2.png" width="48">
        - Fase _testing_: <img src="https://user-images.githubusercontent.com/72246401/137123510-107b5ee1-a0ef-4a64-a420-457a7c4c504a.png" width="48">
        

## Data Understanding
Pada dataset yang kita gunakan [SBA Approval Loan](https://raw.githubusercontent.com/kurnivan-ny/predictive.analysis.io/main/small_business_loan_approval.csv) terdapat 2.102 baris (records atau jumlah pengamatan) dan 34 kolom.Kemudian kita hanya menggunakan 7 kolom saja, karena 7 kolom tersebut mengandung informasi yang dapat digunakan pada Predictive Analysis. Deskripsi variabel yang kita gunakan sebagai berikut.
1. NewExist (1 = Existing Business, 2 = New Business) menunjukkan apakah bisnis tersebut merupakan Existing Business (bisnis yang sudah ada) berada selama lebih dari 2 tahun atau New Business (bisnis baru) berada kurang dari atau sama dengan 2 tahun.
2. RevLineCr (Revolving line of credit) adalah jalur kredit bergulir dimana Y = Yes, N = No.
3. LowDoc (Y = Yes, N = No) berfungsi untuk memproses pinjaman lebih efisien. Program “LowDoc Loan” diterapkan di mana pinjaman di bawah $150.000 dapat diproses menggunakan aplikasi satu halaman. "Yes" menunjukkan pinjaman dengan aplikasi satu halaman, dan "No" menunjukkan pinjaman dengan lebih banyak informasi yang dilampirkan ke aplikasi.
4. Term berupa jangka waktu pinjaman dalam bulan.
5. GrAppv berupa jumlah kotor pinjaman yang disetujui oleh bank\
6. SBA_Appv jumlah pinjaman yang disetujui SBA yang dijamin
7. MIS_Status berupa Status pinjaman dicabut (Gagal Membayar) = CHGOFF, Dibayar penuh (Berhasil Membayar) =PIF

## Data Preparation
Data Preparation adalah proses mengubah atau mentransformasi fitur-fitur data ke dalam bentuk yang mudah diinterpretasikan dan diproses oleh algoritma machine learning. Beberapa hal yang bisa dilakukan sebagai berikut.

Ada beberapa tahapan yang umum dilakukan pada data preparation, antara lain data transform, dimensi reduction, imbalance data.
- Data transform ini kita pisahkan antara data numerik dan data kategori. Data numerik menggunakan Standardization tetapi penggunaan setelah splitting data agar menghindari kebocoran data.
- Dimensi reduction ini kita mengurangi jumlah fitur dengan mempertahankan informasi pada data. Dimensi reduction menggunakan PCA. Variabel yang kita dimensi reduction adalah GrAppv dan SBA_Appv karena memiliki korelasi yang sangat tinggi mendekati 1.
- Imbalance data berupa tidak seimbang pada variabel target. Imbalance data kita menggunakan oversampling dengan teknik SMOTE.SMOTE bekerja dengan memilih contoh yang dekat di 
ruang fitur, menggambar garis di antara contoh di ruang fitur dan 
menggambar sampel baru pada titik di sepanjang garis itu.

## Modeling
Sebelum melakukan modeling, kita melakukan splitting data dengan membagi dataset menjadi 2, yaitu data train dan data test. Disini saya menggunakan stratify untuk melakukan cross validation, Cross validation disini digunakan untuk menguji seberapa stabil model yang kita gunakan.

Model development adalah tahapan di mana kita menggunakan algoritma machine learning untuk menjawab problem statement dari tahap business understanding. Pada Modeling ini kita menggunakan algoritma:

1. Logistic Regression - karena algoritma ringan dan memiliki memory ram yang sedikit.
2. SVC - karena algoritma sangat bagus untuk dataset yang banyak, tetapi memiliki kelemahan di memory ram yang banyak.

Untuk modeling, kita fokus memonitoring recall. Setelah dilakukan training, SVC memiliki nilai recall tertinggi dibanding dengan Logistic Regression. SVC memiliki average recall score 93,94% sedangkan Logistic Regression memiliki average recall score 79,76%

Kemudian kita menggunakan hyperparameter tuning dengan GridSearchCV untuk mencari parameter terbaik. Setelah di training, mendapatkan nilai recall yang meningkat, tetapi tidak signifikan.SVC dengan best parameter memiliki average recall score 94,56% sedangkan Logistic Regression dengan best parameter memiliki average recall score 81,25%

## Evaluation
Dari hasil average recall score teringgi pada SVC dan SVC dengan best parameter. Kita bandingkan pada false negatif.
Dilihat dari confusion matrix, SVC dan SVC dengan best parameter memiliki false negatif sama. Setelah itu kita melihat false positif, ternyata SVC lebih baik dari SVC dengan best parameter.
Sehingga kita menggunakan SVC untuk model predictive analysis untuk klasifikasi.


**---Ini adalah bagian akhir laporan---**
