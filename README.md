# Klasifikasi Emosi Video Media Sosial — BDC Tel-U 2026

Menentukan kategori emosi dari 200 video uji, berdasarkan 10 kategori yang ada
pada data latih. Data mentah yang diberikan hanya berupa **tautan video** —
tidak ada teks, tidak ada fitur — sehingga seluruh fitur harus dibangun sendiri
dengan mengambil konten di balik tautan tersebut.

---

## Menjalankan

```bash
pip install -r requirements.txt
python run_all.py
```

Selesai dalam **±3,5 menit**. Tidak butuh internet, tidak butuh GPU.
Hasil akhir: `outputs/submission.csv`.

Hasil scraping dan transkripsi sudah disertakan di `data/cache/` (5 MB), jadi
tahap yang mahal tidak perlu diulang. Cara membangun ulang dari nol ada di
bagian [Membangun ulang cache](#membangun-ulang-cache).

---

## Hasil

Validasi silang **StratifiedGroupKFold** 5 fold, dikelompokkan per URL
ternormalisasi.

| Tahap | Akurasi | macro-F1 |
|---|---|---|
| Baseline (selalu tebak `Surprise`) | 41,2% | 0,058 |
| + fitur metadata | 29,6% | 0,107 |
| + fitur teks (hitungan) | 31,0% | 0,118 |
| + TF-IDF caption | 33,5% | 0,118 |
| + uploader (target encoding) | 35,6% | 0,134 |
| + fitur audio (numerik) | 35,6% | 0,125 |
| + transkrip Whisper ke TF-IDF | 37,4% | 0,119 |
| **Model final: TF-IDF teks saja + LogReg** | **42,0%** | **0,099** |

Model final mengalahkan baseline pada **kedua** metrik. Tabel lengkap ada di
`outputs/ablation.csv`.

### Konteks yang penting untuk membaca angka di atas

Terdapat 11 video yang dipakai pada lebih dari satu baris data latih (41 baris).
Menebaknya dengan label mayoritas dari video itu sendiri — informasi yang tidak
akan pernah dimiliki model — hanya benar **56,1%**. Satu video bahkan dilabeli
Trust 7×, Proud 4×, Surprise 3×, Fear 1×, dan Neutral 1×.

Artinya **anotator tidak sepakat pada video yang sama persis**, dan ~56% adalah
plafon praktis untuk tugas ini. Jarak dari baseline 41,2% ke plafon tersebut
hanya sekitar 15 poin.

---

## Temuan utama

**1. Label melacak topik konten, bukan kata emosi.** Leksikon emosi Bahasa
Indonesia yang dirakit manual gagal: dari lima kelas terbesar, hanya
`lex_anger` yang menempatkan kelasnya sendiri di peringkat teratas. Analisis
log-odds menjelaskan sebabnya — kata paling khas untuk Trust adalah
`dumbbell, gym` (konten kebugaran), untuk Proud `emas, meraih` (prestasi
olahraga), untuk Surprise `roda, bensin, harga` (otomotif). Tidak satu pun kata
emosi. Karena itu TF-IDF yang menangkap topik justru bekerja lebih baik daripada
pencarian kata emosi.

**2. Fitur numerik menurunkan performa, bukan sekadar tidak membantu.** Diuji
pada tiga tingkat pembobotan kelas, gabungan teks + numerik selalu kalah dari
teks saja (42,0% vs 38,4% pada bobot^0,5). Model final karena itu membuang
seluruh fitur metadata dan audio numerik. Fitur-fitur tersebut tetap
dipertahankan di dalam kode dan tabel ablation sebagai bukti terukur.

**3. Transkrip audio menaikkan cakupan jauh lebih besar daripada performa.**
Caption hanya tersedia pada 62,6% baris, sedangkan transkrip Whisper mencakup
87,5%. Kenaikan cakupan 25 poin itu hanya berbuah ~2 poin akurasi — tetapi
justru konfigurasi inilah yang pertama kali melewati baseline.

---

## Visualisasi

```bash
python src/viz.py            # tulis semua grafik ke outputs/figures/
```

Atau buka `notebooks/01_visualisasi.ipynb` untuk grafik beserta penafsirannya.
Notebook tersebut hanya menampilkan dan menafsirkan — seluruh kode plot ada di
`src/viz.py`, sehingga tidak ada dua versi yang perlu diselaraskan.

| Berkas | Isi |
|---|---|
| `01_distribusi_label.png` | ketimpangan kelas dan tinggi baseline |
| `02_cakupan_data.png` | keberhasilan akuisisi per sumber; lompatan cakupan dari transkrip |
| `03_kata_pembeda.png` | bukti bahwa label melacak topik, bukan kata emosi |
| `04_ablation.png` | akurasi & macro-F1 per tahap, terhadap baseline |
| `05_confusion_matrix.png` | pola kesalahan yang tersisa |
| `06_plafon_derau_label.png` | hasil model terhadap plafon kesepakatan anotator |

---

## Struktur

```
run_all.py              jalankan seluruh pipeline
requirements.txt
RENCANA.md              catatan lengkap: keputusan, kegagalan, dan alasannya
datatrain.csv           data latih (803 baris)
datatest.csv            data uji (200 baris)

src/
  acquire.py            unduh video + metadata (Instagram, Drive, CDN)
  transcribe.py         transkripsi audio dengan Whisper
  features_meta.py      durasi, resolusi, engagement, waktu unggah
  features_text.py      caption, hashtag, komentar, emoji, leksikon
  features_audio.py     fitur dari transkrip
  build_dataset.py      penggabungan tabel fitur
  train.py              validasi silang + tabel ablation
  predict.py            model final -> submission.csv
  package_submission.py penyiapan paket kiriman
  common.py             path dan pemuat data bersama

data/
  cache/meta/           metadata hasil scraping (941 berkas)
  cache/transcript/     transkrip Whisper (837 berkas)
  cache/manifest.csv    status akuisisi tiap tautan
  gagal_train.csv       daftar tautan yang tidak bisa diambil + sebabnya
  gagal_test.csv

outputs/
  submission.csv        hasil akhir
  ablation.csv          tabel perbandingan tahap
  post_processing.csv   catatan penimpaan prediksi
```

---

## Catatan metodologi

**Pengelompokan pada validasi silang.** Setelah URL dinormalisasi (parameter
pelacakan dibuang, bentuk `instagram.com/username/reel/X` disamakan dengan
`instagram.com/reel/X`), 1.003 baris menyusut menjadi 958 tautan unik. Karena
ada baris kembar, validasi memakai `StratifiedGroupKFold` dengan grup per
tautan — tanpa itu, baris kembar bisa muncul di data latih dan validasi
sekaligus sehingga skor menjadi terlalu optimistis.

**Vektorisasi di dalam pipeline.** TF-IDF dan target encoding dipasang di dalam
`Pipeline` sehingga di-fit ulang pada setiap fold. Bila di-fit sekali di luar,
kosakata dan statistik label dari fold validasi ikut membentuk fitur.

**Pembobotan kelas bertingkat.** Parameter `ALPHA_BOBOT` di `predict.py`
mengatur kekuatan pembobotan (0 = tanpa bobot, 1 = `balanced`). Nilai 0,5
dipilih karena satu-satunya titik yang mengalahkan baseline pada kedua metrik;
pada nilai 1 kelas bersampel tunggal (`Love`, `Loyalty`) memperoleh bobot ~80×
dan menyeret akurasi ke bawah baseline.

**Penimpaan setelah model.** Sembilan baris uji memiliki URL yang identik dengan
baris latih berlabel konsisten, sehingga labelnya diambil langsung. Empat baris
lain menunjuk video yang di data latih berlabel beragam, dan diisi dengan
modusnya. Seluruh penimpaan tercatat di `outputs/post_processing.csv`.

**Keterbatasan.** Sebagian tautan sumber sudah mati sejak dataset disusun: 94
dari 803 baris latih (11,7%) dan 25 dari 200 baris uji (12,5%) tidak dapat
diambil kontennya. Rinciannya ada di `data/gagal_train.csv` dan
`data/gagal_test.csv`. Baris uji tanpa sinyal diprediksi memakai kelas prior.

---

## Membangun ulang cache

Hanya diperlukan bila ingin mengulang tahap akuisisi dari nol. Butuh internet,
GPU, dan beberapa jam.

```bash
pip install yt-dlp gdown imageio-ffmpeg faster-whisper
python src/acquire.py                    # ±4 jam, 842 video (±6 GB)
python src/transcribe.py --model small   # ±50 menit di RTX 4050
python run_all.py
```

Perlu diketahui: tautan terus berkurang seiring waktu, sehingga membangun ulang
dari nol kemungkinan menghasilkan cakupan yang **lebih rendah** daripada cache
yang disertakan di paket ini.
