# Rencana Pengerjaan — BDC Tel-U 2026

**Deadline: 20 Agustus 2026** (13 hari dari 7 Agustus)
**Tugas:** klasifikasi emosi 200 video test ke dalam 10 kategori yang ada di data train.

---

## 1. Ringkasan Masalah

| | |
|---|---|
| Input | Hanya URL video (`id`, `video`) — tidak ada teks, tidak ada fitur apapun |
| Train | 803 baris berlabel |
| Test | 200 baris, harus diprediksi |
| Label space | 10 kategori, terkunci — dilarang bikin kategori baru |
| Tipe masalah | Klasifikasi multi-kelas, sangat tidak seimbang |

**Inti kesulitannya:** fitur harus dibuat sendiri dengan mengambil konten di balik URL. Kalau tahap akuisisi gagal, seluruh pekerjaan berikutnya ikut gagal. Ini risiko nomor satu.

### Distribusi label (train)

| Emotion | n | % |
|---|---|---|
| Surprise | 331 | 41.2% |
| Trust | 183 | 22.8% |
| Proud | 156 | 19.4% |
| Joy | 53 | 6.6% |
| Anger | 36 | 4.5% |
| Sad | 18 | 2.2% |
| Fear | 16 | 2.0% |
| Neutral | 8 | 1.0% |
| Love | 1 | 0.1% |
| Loyalty | 1 | 0.1% |

**Baseline wajib dikalahkan: tebak `Surprise` untuk semua = 41,2% akurasi.**

### Komposisi sumber

Setelah normalisasi URL, 958 item unik:

| Sumber | Unik | Jalur unduh |
|---|---|---|
| Instagram reel | 637 | yt-dlp |
| Google Drive (file) | 296 | endpoint drive.usercontent |
| CDN Instagram (link mentah) | 24 | HTTP langsung |
| Folder Drive | 1 | manual |

Butuh tiga jalur downloader berbeda. Hasilnya di §11.

### Masalah kualitas data yang sudah teridentifikasi

1. **Label bentrok pada URL duplikat.** Setelah URL dinormalisasi (buang parameter tracking `igsh`/`utm_source`, samakan `instagram.com/reel/X` dengan `instagram.com/username/reel/X`), 1.003 baris menyusut jadi **958 URL unik** — 45 duplikat, jauh lebih banyak dari 9 yang terlihat lewat pencocokan string mentah. Dari 11 grup duplikat, **8 punya label bentrok, menyangkut 35 baris train**.

   Terparah: `drive.google.com/file/d/12YBpn9zWc_...` dipakai di **20 baris** (16 train + 4 test). Dugaan awal bahwa ini link rusak **terbantah** — file-nya sudah diunduh dan ternyata video asli 51 detik. Jadi ini kesalahan copy-paste panitia, bukan link mati. Rincian dan konsekuensinya di §11.

   Karena duplikasi ini nyata, **`GroupKFold` per URL ternormalisasi jadi wajib** — tanpa itu, duplikat bocor antara train dan validation dan skor CV jadi bohong.

2. **14 baris test URL-nya identik dengan baris train** (setelah normalisasi; pencocokan string mentah hanya menemukan 11). **Sembilan labelnya konsisten → jawaban gratis, 4,5% dari test set:**

   | test id | label | | test id | label |
   |---|---|---|---|---|
   | 59 | Surprise | | 116 | Surprise |
   | 61 | Surprise | | 165 | Trust |
   | 91 | Surprise | | 170 | Trust |
   | 108 | Proud | | 194 | Trust |
   | 109 | Surprise | | | |

   Lima sisanya bentrok dan perlu keputusan sadar:

   - **id 18, 26, 27, 28** → semua menunjuk video `gd_12YBpn9zWc...`. Label train untuk video yang sama: Trust 7, Proud 4, Surprise 3, Fear 1, Neutral 1 → **pakai Trust** (modus).
   - **id 119** → Joy 1 vs Trust 1, seri. Serahkan ke prediksi model.

   Terapkan sebagai post-processing dan **dokumentasikan di laporan**, jangan disembunyikan.

3. **Love dan Loyalty hanya 1 sampel.** Tidak mungkin dipelajari model. Keputusan penanganannya harus ditulis eksplisit di laporan (lihat §6).

---

## 2. Strategi: Ikuti Bobot Penilaian

Tujuh kriteria penilaian, dan **akurasi hanya salah satunya**:

| Kriteria | Cara kita menang |
|---|---|
| Kesesuaian solusi | Framing jelas: multimodal feature engineering → klasifikasi |
| Akurasi/performa | Target realistis **45–52%** akurasi (baseline 41,2%, plafon derau label ~56%) |
| Kebersihan & struktur | Package modular, satu perintah jalan, bukan notebook raksasa |
| Dokumentasi & reproducibility | README + seed tetap + cache ikut dikirim |
| Kreativitas & inovasi | Fusi fitur audio-visual-teks buatan sendiri |
| Interpretasi & insight | Feature importance → temuan yang bisa dijelaskan |
| Visualisasi | EDA, distribusi fitur per emosi, confusion matrix |

**Konsekuensi:** jangan habiskan 10 hari mengejar +3% akurasi. Kunci akurasi di level "cukup baik", lalu investasikan sisa waktu ke interpretasi, visualisasi, dan kerapian package.

### Arsitektur dua lapis (pretrained DIIZINKAN — dikonfirmasi panitia)

Panitia mengizinkan pretrained open-source seperti IndoBERT. Yang dilarang hanya API LLM berbayar. Tapi **jangan buang fitur hand-crafted** — justru itu yang memenangkan 3 kriteria. Rancangannya dua lapis, dengan pembagian tugas yang jelas:

| Lapis | Isi | Melayani kriteria |
|---|---|---|
| **Lapis interpretasi** | Fitur hand-crafted: metadata, leksikon, akustik (MFCC/tempo/energi), visual (cut rate, saturasi, wajah) | Interpretasi & Insight, Visualisasi, Kreativitas |
| **Lapis performa** | Embedding pretrained: IndoBERT (caption + transkrip), Whisper (speech→teks), embedding visual | Akurasi/Performa Model |

Ini bukan kompromi — ini justru posisi terkuat. Embedding menaikkan akurasi tapi tidak bisa dijelaskan; fitur hand-crafted memberi cerita SHAP yang konkret. Di laporan, nyatakan pembagian peran ini secara eksplisit sebagai keputusan desain sadar. Itu sendiri poin untuk kriteria Kesesuaian Solusi.

**Urutan pengerjaan tetap hand-crafted duluan.** Alasannya bukan aturan, tapi risiko: fitur hand-crafted jalan di CPU, tidak bisa gagal karena VRAM, dan jadi jaring pengaman kalau Whisper/IndoBERT bermasalah di sisa waktu yang mepet.

Catatan kecil untuk laporan: leksikon emosi Bahasa Indonesia (InSet dsb.) dan Haar cascade bawaan OpenCV tetap disebutkan di bagian metodologi demi transparansi.

---

## 3. Arsitektur Pipeline

```
URL  →  [1] Akuisisi  →  [2] Ekstraksi Fitur  →  [3] Model  →  [4] Prediksi
              │                    │
           cache/            features/*.parquet
        (WAJIB dikirim)
```

**Prinsip yang tidak boleh dilanggar:** panitia harus bisa menjalankan pipeline **tanpa internet**. Mereka tidak akan bisa scraping ulang (rate limit, butuh login, link bisa mati). Jadi:

- Semua hasil scraping disimpan ke `data/cache/`
- Cache ikut dikirim bersama package
- `acquire.py` cek cache dulu, skip kalau sudah ada
- Ada flag `--offline` yang melewati akuisisi sepenuhnya

---

## 4. Struktur Repo

```
BDC_Tel-U_2026/
├── README.md                  # cara menjalankan — dibaca panitia pertama kali
├── requirements.txt
├── config.yaml                # seed, path, hyperparameter
├── run_all.py                 # satu perintah, end-to-end
├── data/
│   ├── raw/                   # datatrain.csv, datatest.csv
│   ├── cache/                 # hasil scraping — IKUT DIKIRIM
│   │   ├── meta/*.json        # caption, hashtag, komentar, view/like
│   │   └── video/*.mp4        # (kalau terlalu besar, kirim fitur saja)
│   └── features/              # features_train.parquet, features_test.parquet
├── src/
│   ├── acquire.py             # downloader, resume-able, throttled
│   ├── features_meta.py       # durasi, view, like, jumlah komentar
│   ├── features_text.py       # caption, hashtag, emoji, leksikon
│   ├── features_audio.py      # MFCC, energi, tempo, pitch
│   ├── features_visual.py     # warna, kecerahan, cut rate, motion
│   ├── build_dataset.py       # gabung semua fitur
│   ├── train.py               # CV + training
│   └── predict.py             # hasilkan submission.csv
├── notebooks/
│   ├── 01_eda.ipynb
│   ├── 02_feature_analysis.ipynb
│   └── 03_model_interpretation.ipynb
├── outputs/
│   ├── figures/
│   ├── models/
│   └── submission.csv
└── laporan/
```

---

## 5. Rencana Fitur

### 5A. Lapis interpretasi (hand-crafted, dikerjakan duluan)

Target ±100–150 fitur. Semuanya bisa dijelaskan — ini modal untuk kriteria Interpretasi.

### Metadata (murah, ambil duluan)
Durasi video, jumlah like, komentar, rasio like/komentar, panjang caption, jumlah hashtag, jumlah mention, jam/hari posting.

**Identitas uploader — jauh lebih lemah dari dugaan.** Menebak label dari modus uploader menghasilkan **41,7%** dengan CV yang jujur, melawan baseline 41,2%. Praktis tidak ada bedanya.

> **Koreksi angka.** Sempat tertulis 50,3% di sini. Itu **salah** — diukur tanpa cross-validation, jadi modus uploader dihitung memakai baris yang sama dengan yang dinilai. Setelah dipisah per fold, kelebihannya menguap. Pelajaran yang berlaku untuk seluruh proyek: **setiap heuristik harus divalidasi silang sebelum dipercaya**, sekalipun kelihatan sederhana.

> **Awas leakage:** target encoding **wajib** dihitung di dalam fold CV, bukan di seluruh data train. Kalau tidak, skor CV akan optimistis palsu.

> **KOREKSI dugaan awal.** Sempat saya tulis datasetnya berdomain otomotif — itu **salah**, kesimpulan dari 10 sampel pertama. Setelah 621 metadata terkumpul, isinya konten influencer/brand Indonesia lintas vertikal: kecantikan (Tasya Farasya, Muhamad Fachrian), kesehatan (Hello Sehat), otomotif (Moladin, MotoMobi, Fitra Eri, Ridwan Hanif), teknologi (GadgetIn, Jagat Review), olahraga (KONI DKI), kebugaran (2Ninefit), bahkan hewan peliharaan (drh. Citan). Fitur `is_brand_account` dibatalkan — pembelahannya bukan brand vs influencer.

### Teks — caption, hashtag, komentar teratas
- TF-IDF word (1–2 gram) + char n-gram (3–5) → tahan typo & bahasa gaul
- Hitungan emoji per kategori sentimen
- Fitur gaya: rasio huruf kapital, jumlah `!`, jumlah `?`
- Skor leksikon emosi Bahasa Indonesia — **sudah diuji, hasilnya gagal (lihat di bawah)**

> ### Temuan penting: label melacak TOPIK, bukan kata emosi
>
> Leksikon emosi buatan tangan (8 kategori) diuji pada data train: **hanya
> `lex_anger` yang menempatkan kelasnya sendiri di peringkat 1** dari 5 kelas
> besar. Sisanya peringkat 2–4 — praktis tidak membedakan.
>
> Analisis log-odds pada 503 caption menjelaskan sebabnya. Kata yang paling
> membedakan sama sekali bukan kata emosi:
>
> | Emosi | Kata paling khas | Topik sebenarnya |
> |---|---|---|
> | Trust | dumbbell, gym, gerakan, pull | kebugaran |
> | Proud | emas, meraih, repetisi, ketua | prestasi olahraga (PON XXI) |
> | Surprise | roda, bensin, mobilbaru, harga | otomotif |
> | Joy | giias 2025, om mobi, pon xxi | liputan acara |
>
> Artinya anotator memberi label berdasarkan **jenis konten**, bukan berdasarkan
> kata-kata emosional di caption. Konsekuensi desain: **TF-IDF dan embedding
> menangkap topik, dan itulah yang justru berguna** — bukan pencarian kata emosi.
>
> Fitur leksikon **tetap dipertahankan di kode** meski lemah. Alasannya bukan
> performa: hasil negatif ini adalah bahan bagus untuk kriteria Interpretasi &
> Insight, dan lebih jujur dilaporkan daripada dihapus diam-diam.

### Audio (librosa)
MFCC (13 koefisien: mean + std), RMS energy, zero-crossing rate, spectral centroid & rolloff, tempo, estimasi pitch, rasio silence.

*Hipotesis:* energi tinggi + tempo cepat → Surprise/Joy. Tempo lambat + energi rendah → Sad.

### Visual (OpenCV)
- Histogram warna HSV (mean/std hue, saturation, value)
- Kecerahan & kontras rata-rata
- **Cut rate** — jumlah pergantian shot per detik (proxy energi editing)
- Magnitudo optical flow (intensitas gerakan)
- Jumlah wajah per frame (Haar cascade) + rasio close-up
- Rasio piksel teks/overlay

*Hipotesis:* cut rate tinggi + saturasi tinggi → Surprise. Banyak wajah close-up → Trust/Proud.

**Sampling:** ambil 8–16 keyframe merata per video. Jangan proses semua frame — buang waktu.

> **Catatan resolusi.** Video diunduh pada 360×640 (lihat §11). Fitur berbasis rasio — histogram warna, kecerahan, cut rate — tidak terpengaruh resolusi. Tapi **magnitudo optical flow dalam satuan piksel ikut menyusut**, jadi normalisasi terhadap lebar frame. Deteksi wajah juga perlu `minSize` yang disesuaikan.

### 5B. Lapis performa (pretrained, setelah 5A jalan)

**Whisper (`small`) — transkrip audio.** Ini kemungkinan sumber sinyal terkuat: apa yang *diucapkan* di video jauh lebih informatif daripada tekstur audionya. `small` (±2 GB VRAM) muat nyaman di RTX 4050 6 GB dan dukungan Bahasa Indonesia-nya memadai. Estimasi ~1 jam untuk seluruh dataset. Simpan transkrip ke `data/cache/transcript/*.json` supaya tidak perlu diulang.

**IndoBERT — embedding teks.** Jalankan pada gabungan caption + hashtag + transkrip Whisper. Ambil mean-pooled hidden state, 768 dimensi.

> **Jebakan penting:** 768 dimensi untuk 803 sampel = overfit hampir pasti. **Wajib reduksi dimensi** (TruncatedSVD/PCA ke 30–50 komponen) sebelum masuk LightGBM. Alternatifnya, latih classifier linear langsung di atas embedding lalu pakai probabilitas keluarannya sebagai ~10 fitur meta (stacking). Bandingkan keduanya.

**Embedding visual (opsional, prioritas terendah).** CLIP/ResNet pada keyframe. Kerjakan hanya kalau waktu tersisa setelah semua yang lain beres — rasio manfaat terhadap waktunya paling rendah.

---

## 6. Modeling

### Validasi (kritis — gampang salah di sini)
Gunakan **StratifiedGroupKFold**, `group = URL`. Wajib grup per URL, karena ada URL duplikat — kalau tidak, duplikat bocor antara train dan validation dan skor CV jadi bohong.

Kelas singleton (Love, Loyalty) akan bikin stratifikasi gagal. Tangani eksplisit: gabungkan sementara ke kelas terdekat saat split, atau taruh di fold pertama.

### Model
Mulai sederhana, naik bertahap. Catat skor tiap tahap — ini bahan tabel ablation di laporan.

1. Baseline: majority class (41,2%)
2. Logistic Regression pada TF-IDF caption saja
3. LightGBM pada fitur metadata + teks
4. LightGBM pada semua fitur hand-crafted (meta + teks + audio + visual) ← **checkpoint aman**
5. \+ embedding IndoBERT (tereduksi SVD)
6. \+ fitur transkrip Whisper
7. Ensemble/stacking (LightGBM + SVM + Logistic Regression)

Tahap 4 adalah titik aman: kalau setelah itu waktu menipis, solusi sudah lengkap dan bisa dikirim. Tahap 5–7 murni upside.

### Ketidakseimbangan kelas
`class_weight='balanced'` sebagai default. Coba SMOTE tapi hati-hati — dengan 8 sampel Neutral, sintesis cenderung bikin noise. Bandingkan dengan/tanpa, laporkan hasilnya.

### Metrik (dikonfirmasi: tidak ada metrik tunggal resmi)
Panitia hanya menyebut "metrik yang relevan (akurasi, F1-score, dll.)" — artinya **tidak ada satu angka yang dikejar**, penilaian bersifat holistik. Konsekuensinya:

- Laporkan **accuracy + macro-F1 + weighted-F1** berdampingan di setiap eksperimen.
- Pemilihan model final: pakai **macro-F1** sebagai kriteria utama. Model yang cuma menebak `Surprise` terlihat "lumayan" di accuracy (41%) padahal tidak belajar apa-apa — macro-F1 menghukum itu.
- **Confusion matrix wajib masuk laporan.** Ini sekaligus bahan visualisasi dan interpretasi — dua kriteria terjawab satu grafik.
- Karena tidak ada leaderboard otomatis, **kejelasan cara evaluasi ikut dinilai.** Tulis skema CV secara eksplisit di laporan.

### Keputusan Love & Loyalty
Dengan 1 sampel, model tidak akan pernah memprediksinya — dan itu wajar. Total 0,1% data, dampak ke accuracy nol.

Yang bernilai justru **analisisnya**: jelaskan kenapa dua kelas ini mustahil dipelajari, tunjukkan dampaknya ke macro-F1 (menyumbang 20% bobot tapi selalu bernilai 0), lalu usulkan alternatif — misalnya penggabungan semantik Love→Joy dan Loyalty→Trust — dan laporkan skor pada kedua skema. Menjelaskan keterbatasan data dengan jernih itu poin untuk Interpretasi & Insight, bukan kelemahan.

### Post-processing
Timpa prediksi 6 baris test yang URL-nya cocok persis dengan train (lihat §1.2). Dokumentasikan sebagai keputusan sadar, jangan disembunyikan.

---

## 7. Timeline

| Hari | Tanggal | Fokus | Output |
|---|---|---|---|
| 1 | 7 Ags | Bangun `acquire.py`, mulai scraping | Laporan tingkat keberhasilan download |
| 2–3 | 8–9 Ags | Scraping jalan terus + EDA + bersihkan label bentrok | `data/cache/` terisi, notebook EDA |
| 4 | 10 Ags | Fitur metadata + teks | `features_text.parquet` |
| 5 | 11 Ags | Fitur audio + **jalankan Whisper** (~1 jam GPU, biarkan jalan di background) | `features_audio.parquet`, `cache/transcript/` |
| 6 | 12 Ags | Fitur visual | `features_visual.parquet` |
| 7 | 13 Ags | Setup CV + model tahap 1–4 | **Checkpoint aman: solusi lengkap tanpa pretrained** |
| 8 | 14 Ags | Embedding IndoBERT + reduksi dimensi | Tahap 5–6 |
| 9 | 15 Ags | Tuning, ensemble, tabel ablation | Model final |
| 10 | 16 Ags | Interpretasi + feature importance | Notebook interpretasi |
| 11 | 17 Ags | Visualisasi & figure laporan | `outputs/figures/` |
| 12 | 18 Ags | Rapikan package, tulis README, uji clean-run | Package siap kirim |
| 13 | 19 Ags | Laporan + buffer | Semua deliverable |
| — | 20 Ags | **Kirim** | |

**Gate hari 1:** kalau tingkat keberhasilan download < 60%, berhenti dan ubah strategi — bersandar ke metadata + teks saja, drop jalur audio/visual.

---

## 8. Risiko

| Risiko | Dampak | Mitigasi |
|---|---|---|
| Instagram rate limit / butuh login | Fatal | Throttle 3–5 detik, scraper resume-able, jalankan sejak hari 1 |
| Link mati sebelum sempat di-scrape | Fatal | Scraping **hari ini juga**, sebelum apapun |
| Panitia tidak bisa jalankan pipeline | Diskualifikasi teknis | Cache ikut dikirim + mode `--offline` |
| VRAM 4–6 GB kurang | Rendah | Lapis hand-crafted jalan di CPU. Whisper `small` (±2 GB) + IndoBERT base muat di RTX 4050; batch kecil di 3050 |
| Overfit karena embedding 768-dim | **Tinggi** | Wajib reduksi SVD/PCA ke 30–50 dim, atau stacking probabilitas. Jangan feed mentah ke LightGBM |
| Overfit (803 sampel, 10 kelas) | Sedang | CV ketat, model sederhana, regularisasi kuat |
| Waktu habis di modeling | Tinggi | Freeze model hari 15, sisanya untuk laporan & visualisasi |

---

## 9. Status Konfirmasi Panitia

**Sudah terjawab:**

1. ~~Model pretrained open-source~~ → **BOLEH** (IndoBERT, Whisper, dll.). Yang dilarang hanya API LLM berbayar.
2. ~~Metrik resmi~~ → **Tidak ada metrik tunggal.** "Metrik yang relevan (akurasi, F1-score, dll.)" — penilaian holistik. Lihat §6.

**Masih perlu ditanyakan:**

3. **Format submission** — nama kolom persis, nama file, apakah butuh probabilitas per kelas?
4. **Format laporan** — halaman maksimal, template, PDF atau notebook?
5. **Ukuran maksimal kiriman** — menentukan apakah cache video ikut dikirim atau hanya fitur hasil ekstraksi. Ini yang paling mendesak: kalau ada batas ketat, cache video (bisa beberapa GB) harus diganti fitur terekstrak saja.

---

## 10. Langkah Berikutnya

1. **Mulai scraping hari ini** — jalur kritis, link bisa mati kapan saja.
2. Kirim 3 pertanyaan sisa di §9 ke panitia, terutama nomor 5.
3. Sambil scraping jalan, kerjakan EDA dan bersihkan label bentrok.

---

## 11. Hasil Akuisisi (selesai 8 Agustus)

**842 dari 958 URL unik berhasil (87,9%). Cache 6,14 GB di `C:\bdc_cache`, 842 video, 0 rusak. Akuisisi DITUTUP — sisa kegagalan sudah dipastikan tidak bisa dipulihkan.**

| Sumber | Berhasil | Gagal | Keterangan |
|---|---|---|---|
| Instagram | 621 | 16 | 12 dipulihkan lewat cookie login |
| Google Drive | 221 | 75 | benar-benar terhapus |
| CDN mentah | 0 | 24 | URL bertanda tangan, kadaluarsa |
| Folder Drive | 0 | 1 | ambigu, lihat di bawah |

> **Bug yang sudah ditutup:** percobaan ulang yang gagal dulu menimpa baris
> manifest yang sebelumnya sukses, padahal video dan metadata-nya masih utuh di
> disk — 4 item terhitung hilang padahal datanya ada. `acquire.py` kini menolak
> menurunkan status `ok` selama artefaknya masih ada. Rekonsiliasi manifest vs
> isi disk sekarang cocok sempurna: 842 = 842, selisih 0. **Jalankan pengecekan
> ini tiap kali selesai retry.**

### Cakupan per baris

| | Punya video | Kosong total |
|---|---|---|
| Train (803) | 709 (**88,3%**) | 94 (11,7%) |
| Test (200) | 175 (**87,5%**) | 25 (12,5%) |

Kehilangan tersebar merata antar kelas — tidak ada kelas yang tersapu habis, jadi
bias sistematisnya kecil. Daftar lengkap baris yang hilang ada di
`data/gagal_train.csv` dan `data/gagal_test.csv` (id, label, sebab, URL).

**25 baris test (12,5%) tidak punya sinyal sama sekali** dan hanya bisa ditebak
dari prior kelas. Tulis ini eksplisit di laporan sebagai batas atas performa yang
bisa dicapai, bukan disembunyikan.

### Yang sudah dipastikan buntu (jangan diulang)

- **Cookie login Chrome** memulihkan 12 baris train, tapi 20 sisa Instagram tetap
  gagal dengan `HTTP 400` walau cookie valid. Sudah diverifikasi dengan uji
  kontrol: post yang sukses tetap bisa diakses dengan cookie yang sama, jadi
  **bukan blokir IP** — post-nya memang mati/dibatasi.
- **Endpoint embed Instagram** (`/p/{kode}/embed/captioned/`) buntu: membalas
  halaman login 609 KB bahkan untuk post yang jelas berhasil diunduh. Jangan
  buang waktu ke sini.
- **Folder Drive (test id 107)** berisi 41 video. Pencocokan ID: 33 sudah kita
  punya lewat baris lain, **0 cocok dengan yang mati**, 8 tidak ada di dataset.
  Tidak ada cara menentukan mana yang dimaksud untuk id 107 — informasinya hilang
  di sumber. Biarkan model memprediksinya.

### Durasi diambil dari file video, bukan dari scraping

Hanya 12 dari 621 metadata Instagram punya `duration` (yang diambil dengan login).
**Jangan scraping ulang untuk ini** — durasi, resolusi, dan ada/tidaknya audio
bisa dibaca langsung dari file video: 830 file dalam ~30 detik, offline, tanpa
risiko rate limit. Cara ini juga satu-satunya jalan untuk 221 video Drive yang
metadata-nya kosong total. Masuk ke `features_meta.py`.

`view_count` memang hilang dan tidak bisa dipulihkan tanpa scraping ulang, tapi
`like_count` dan `comment_count` sudah ada dan menangkap sinyal engagement yang
sama — tidak sepadan dengan risikonya.

### Temuan kualitas data

1. **`gd_12YBpn9zWc...` bukan link rusak — itu video asli** berdurasi 51 detik,
   dipakai di **20 baris** (16 train + 4 test) dengan label berbeda-beda:
   Trust 7, Proud 4, Surprise 3, Fear 1, Neutral 1. Ini kesalahan copy-paste
   panitia saat menyusun CSV. Untuk test id 18/26/27/28, tebakan paling bisa
   dipertahankan adalah **Trust** (mayoritas label train pada video yang sama).
   Total 35 baris train terdampak konflik label serupa.

2. **File Drive bernama pola Instaloader** (`2025-07-30_11-35-03_UTC.mp4`).
   Artinya konten Drive adalah **mirror Instagram**, bukan YouTube — penyebutan
   "YouTube" di brief lomba menyesatkan. Praktis seluruh dataset ini Instagram.

3. **Domain: otomotif Indonesia.** Wuling Motors ID, Moladin, Isuzu/GIIAS,
   Ridwan Hanif, Nex Carlos. Lihat §5A soal fitur uploader.

---

## 12. Catatan Teknis Akuisisi

### Kenapa video diunduh pada 360×640

Awalnya tiap video terunduh ~20 MB, proyeksi total ±20 GB. Penyebabnya bukan
konten, tapi kombinasi dua jebakan:

1. **Instagram menyajikan varian DASH terpisah** (360×640, 540×960, 720×1280,
   1080×1920) plus audio m4a. Menggabungkannya butuh ffmpeg. Tanpa ffmpeg,
   yt-dlp **diam-diam** jatuh ke format progresif kualitas penuh — tidak ada
   pesan error, file sekadar jadi 4–8× lebih besar.
2. **Reel itu potret.** Filter `[height<=480]` tidak berfungsi: varian "360p"
   berukuran 360×640, jadi height-nya 640 dan lolos filter. Yang benar adalah
   `format_sort: ["res:480"]`, karena `res` memakai sisi terpendek — benar untuk
   potret maupun lanskap.

Hasil setelah diperbaiki: **5,65 MB/video, proyeksi 5,4 GB** (dari 20 GB), audio
tetap utuh untuk Whisper.

### ffmpeg tanpa instalasi sistem

Dipakai `pip install imageio-ffmpeg` — membawa binary ffmpeg 7.1 portabel, tanpa
admin, tanpa mengubah PATH sistem.

> **Jebakan:** `ffmpeg_location` harus diisi **path lengkap ke exe**, bukan
> direktorinya. Binary imageio bernama `ffmpeg-win-x86_64-v7.1.exe`; kalau yang
> diberikan direktori, yt-dlp mencari `ffmpeg.exe`, gagal, lalu fallback diam-diam.

Untuk tahap Whisper/librosa nanti, ffmpeg perlu ada di PATH. Tambahkan di awal
skrip:

```python
import imageio_ffmpeg, os
os.environ["PATH"] = os.path.dirname(imageio_ffmpeg.get_ffmpeg_exe()) + os.pathsep + os.environ["PATH"]
```

Kalau Whisper tetap rewel karena nama binary, salin exe tersebut menjadi
`bin/ffmpeg.exe` lalu tambahkan `bin/` ke PATH.

### Cache di dalam OneDrive

Repo ini berada di dalam folder OneDrive. Cache 5,4 GB akan ikut tersinkronisasi
ke cloud. Dua jalan keluar:

- Exclude `data/cache/video` dari sync OneDrive, **atau**
- Set `BDC_CACHE_DIR=C:\bdc_cache` sebelum menjalankan `acquire.py` — path cache
  jadi di luar repo dan manifest otomatis memakai path absolut.

`.gitignore` sudah menahan folder video, audio, dan frames supaya tidak pernah
masuk git. Yang tetap di-commit: `data/cache/meta/*.json` dan `manifest.csv`,
keduanya ringan.

### Google Drive: dua jebakan berurutan

1. **`gdown >= 5` menghapus parameter `fuzzy`.** Memakainya pada gdown 6.x
   melempar `TypeError` — 221 item gagal karena ini, bukan karena datanya mati.
2. **Endpoint gdown kena throttle** setelah ~38 unduhan berturut-turut
   (`Cannot retrieve the public link of the file`).

Solusinya: pakai endpoint langsung sebagai jalur utama, gdown hanya cadangan.

```
https://drive.usercontent.google.com/download?id={FILE_ID}&export=download&confirm=t
```

Endpoint ini melayani file yang sama **tanpa throttle**. Wajib periksa
`Content-Type`: untuk file mati Google membalas **HTTP 200 berisi HTML**, bukan
4xx — tanpa pengecekan ini, halaman error tersimpan sebagai `.mp4`.

> **Jangan menilai file mati dari judul halaman.** Pengecekan
> "Halaman Tidak Ditemukan" pada HTML **tidak reliabel** — ada file yang
> halamannya berkata tidak ditemukan tapi tetap terunduh normal. Biarkan
> percobaan unduh yang memutuskan.

### Verifikasi integritas

Semua 830 video sudah dicek magic-byte (`ftyp`/`moov`) dan ukuran minimum:
0 rusak, 0 mencurigakan. Jalankan ulang pengecekan ini setiap kali menambah data.

---

## 13. Hasil Tahap Awal Model (8 Agustus) — dan Kalibrasi Ulang Target

Semua angka di bawah ini pakai StratifiedGroupKFold 5 fold, `groups=key`,
TF-IDF dan target encoding di dalam Pipeline. Hanya fitur metadata + teks;
audio dan visual belum dipakai sama sekali.

| Tahap | Akurasi | macro-F1 |
|---|---|---|
| Baseline tebak `Surprise` | **41,2%** | 0,058 |
| + metadata | 29,6% | 0,107 |
| + fitur teks (hitungan) | 31,0% | 0,118 |
| + TF-IDF | 33,5% | 0,118 |
| + uploader (target enc.) | 35,6% | 0,134 |
| tanpa `class_weight` | 38,0% | 0,123 |
| TF-IDF saja + LogReg | 40,8% | 0,065 |
| heuristik modus-uploader | 41,7% | 0,158 |

**Belum ada yang mengalahkan baseline secara meyakinkan pada akurasi.** Ini
bukan bug — sudah diperiksa lewat model sederhana, model teregularisasi kuat,
dengan dan tanpa `class_weight`, serta per subset data. Semua bertahan di
kisaran yang sama.

### Plafon dari derau label: ~56%

Ada 11 video yang dipakai di lebih dari satu baris train (41 baris). Menebaknya
dengan label mayoritas video itu sendiri — informasi yang model tidak akan
pernah punya — hanya benar **23/41 = 56,1%**.

Contoh paling telanjang: satu video dilabeli Trust 7x, Proud 4x, Surprise 3x,
Fear 1x, Neutral 1x. **Anotator tidak sepakat pada video yang sama persis.**

Konsekuensinya:

- Target 60–70% yang sempat ditulis di §2 **tidak mungkin dicapai**. Sudah
  dikoreksi jadi 45–52%.
- Jarak baseline (41,2%) ke plafon (~56%) cuma ~15 poin. Ruang geraknya sempit.
- Angka 56% ini berbasis 41 baris saja, jadi perlakukan sebagai indikasi kasar.

**Ini bahan laporan yang kuat, bukan alasan.** Mengukur plafon derau label dan
melaporkan performa relatif terhadapnya justru menjawab kriteria Interpretasi &
Insight jauh lebih baik daripada memamerkan satu angka akurasi tanpa konteks.

### Konsekuensi strategi

1. **macro-F1 jadi metrik utama, dan di situ kita menang telak.** Baseline 0,058
   → model 0,134–0,180, naik ~3x. Model yang cuma menebak `Surprise` terlihat
   "lumayan" pada akurasi padahal tidak belajar apa-apa. Laporkan keduanya, tapi
   pilih model berdasarkan macro-F1 dan jelaskan alasannya.

2. **Prioritas berikutnya: Whisper.** Alasannya cakupan, bukan kecanggihan:

   | Sinyal | Cakupan baris train |
   |---|---|
   | Caption | 62,6% |
   | **Video (→ transkrip audio)** | **88,3%** |

   Transkrip menutup ~26% baris yang punya video tapi tidak punya caption —
   satu-satunya lompatan cakupan besar yang tersisa. Fitur visual menyusul.

3. **Wajib ada penjaga fallback.** Model saat ini lebih buruk dari menebak
   `Surprise` pada baris tanpa sinyal. `predict.py` harus mundur ke prior kelas
   ketika fitur kosong, jangan memaksakan tebakan model.

---

## 14. Hasil Setelah Whisper (8 Agustus)

Transkripsi 836 video dengan Whisper `small` di RTX 4050 (float16, VAD aktif):
**27x realtime, ~50 menit, 828 ada ucapan, 3 musik saja, 5 gagal.**

Cakupan teks naik tajam — inilah alasan Whisper diprioritaskan:

| Sumber teks | Cakupan baris train |
|---|---|
| Caption saja | 62,6% |
| Transkrip saja | 87,5% |
| **Gabungan** | **88,3%** |

### Tangga model lengkap (StratifiedGroupKFold 5 fold)

| Tahap | Akurasi | macro-F1 |
|---|---|---|
| 1. baseline tebak `Surprise` | 41,2% | 0,058 |
| 2. + metadata | 29,6% | 0,107 |
| 3. + fitur teks (hitungan) | 31,0% | 0,118 |
| 4. + TF-IDF caption | 33,5% | 0,118 |
| 5. + uploader | 35,6% | 0,134 |
| 6. + audio (numerik) | 35,6% | 0,125 |
| 7. + transkrip ke TF-IDF | 37,4% | 0,119 |
| **8a. teks saja, bobot^0,0** | **42,6%** | 0,092 |
| **8b. teks saja, bobot^0,5** | **42,0%** | 0,099 |
| **8c. teks saja, bobot^1,0** | 39,7% | **0,155** |

### Tiga temuan yang mengubah arsitektur

**1. Fitur numerik MERUGIKAN, bukan sekadar tidak membantu.** Diuji adil pada
tiga tingkat pembobotan, ensemble teks+numerik selalu kalah dari teks saja:

| bobot | teks saja | ensemble 0,7 teks + 0,3 numerik |
|---|---|---|
| ^0,0 | 42,6% / 0,092 | 41,2% / 0,093 |
| ^0,5 | 42,0% / 0,099 | 38,4% / 0,110 |
| ^1,0 | 39,7% / 0,155 | 36,6% / 0,122 |

Model final karena itu **membuang seluruh fitur metadata dan audio numerik** —
TF-IDF langsung ke Logistic Regression. Sederhana, dan lebih baik.

Fitur hand-crafted **tetap dipertahankan di kode dan di tangga ablation**:
justru tabel di atas — bukti terukur bahwa fitur mahal itu tidak membantu —
adalah bahan Interpretasi & Insight yang lebih kuat daripada memakainya
diam-diam tanpa pembuktian.

**2. Fitur numerik transkrip datar, teksnya yang berharga.** Kecepatan bicara
2,15–2,82 kata/detik dan rasio bicara 0,85–0,96 di hampir semua emosi. Sejalan
dengan §5A: yang menentukan label adalah TOPIK, bukan gaya penyampaian.

**3. Sumbangan Whisper nyata tapi sederhana.** TF-IDF transkrip saja (41,8%)
mengalahkan TF-IDF caption saja (40,8%), dan gabungannya 42,6%. Kenaikan ~2pp
dari kenaikan cakupan 25pp — kecil, tapi ini satu-satunya konfigurasi yang
akhirnya melewati baseline.

### Pembobotan kelas: tukar-guling yang harus dipilih sadar

`alpha` mengatur kekuatan pembobotan (0 = tanpa bobot, 1 = `balanced` penuh).
Nilai tengah jauh lebih baik daripada kedua ujungnya, karena pada alpha=1 kelas
bersampel satu (Love, Loyalty) mendapat bobot ~80x dan menyeret model.

| alpha | Akurasi | macro-F1 | % `Surprise` di submission |
|---|---|---|---|
| 0,0 | 42,6% | 0,092 | — |
| **0,5 (dipakai)** | **42,0%** | **0,099** | **81%** |
| 1,0 | 39,7% | 0,155 | lebih menyebar |

**alpha=0,5 dipilih karena satu-satunya titik yang mengalahkan baseline pada
KEDUA metrik.** Harganya: distribusi prediksi jadi timpang (81% `Surprise`
melawan 41% di train). Kalau juri lebih menghargai macro-F1 dan distribusi yang
realistis daripada akurasi mentah, ganti `ALPHA_BOBOT = 1.0` di `predict.py` —
satu baris.

---

## 15. Visualisasi & Paket Kiriman (8 Agustus)

### Paket: 10 MB, bukan 6 GB

Model final hanya memakai teks, sehingga **video tidak perlu ikut dikirim**.
Durasi dan resolusi sudah tersimpan di `probe_video.json`, jadi tidak ada
informasi yang hilang.

| Isi | Ukuran |
|---|---|
| metadata + transkrip + manifest | 5,0 MB |
| grafik + notebook + kode | 5,0 MB |
| **Total** | **10,0 MB** |
| Video (tidak dikirim) | 6,14 GB |

Pertanyaan ke panitia soal batas ukuran kiriman jadi tidak relevan.

### Uji ruang bersih

Dijalankan persis seperti panitia: `cache_dir.txt` disingkirkan, `data/features`
dan `outputs` dihapus, tanpa video, tanpa internet, tanpa GPU.
**Hasil identik, 3,5 menit, exit code 0.**

### Determinisme LightGBM — bug yang ditemukan lewat grafik

Saat membuat grafik ablation, angkanya bergeser antar-jalankan (29,6% vs 29,1%
pada tahap yang sama) padahal `random_state` sudah dipasang. Penyebabnya
LightGBM multi-thread. Diperbaiki dengan `n_jobs=1`, `deterministic=True`,
`force_row_wise=True`; dua jalankan berturut-turut kini identik sampai 12
desimal.

Tahap teks-saja tidak pernah bergeser sejak awal — Logistic Regression memang
deterministik. Jadi model final selalu reproducible; yang bermasalah hanya tabel
ablation-nya.

**Tabel yang tidak bisa direproduksi tidak layak masuk laporan.**

### Aturan visual yang dipegang

- Satu seri = satu warna. Warna tidak pernah dipakai mengulang panjang batang.
- Magnitudo memakai satu rona terang→gelap, bukan pelangi.
- **Tidak ada sumbu-Y ganda.** Akurasi (%) dan macro-F1 (0–1) dipisah jadi dua
  panel — menumpuknya pada satu sumbu memunculkan hubungan yang tidak ada.
- Palet divalidasi terhadap buta warna: pasangan terburuk ΔE 9,1 (protan),
  di atas ambang 8. Tiap batang diberi label angka, memenuhi syarat relief untuk
  tiga warna yang kontrasnya di bawah 3:1.
- Confusion matrix dinormalisasi per baris supaya kelas kecil tidak tenggelam.

### Status kriteria penilaian

| Kriteria | Status |
|---|---|
| Kesesuaian solusi | selesai |
| Akurasi/performa | 42,0% vs baseline 41,2%, plafon ~56% |
| Kebersihan & struktur | selesai — `run_all.py` satu perintah |
| Dokumentasi & reproducibility | selesai — README + uji ruang bersih |
| Kreativitas & inovasi | multimodal + ablation + pengukuran plafon derau |
| Interpretasi & insight | selesai — tiga temuan terdokumentasi |
| Visualisasi | selesai — 6 grafik + notebook |

### Sisa pekerjaan (opsional)

Semua kriteria sudah terisi. Yang tersisa murni peningkatan:

1. **Laporan/PPT** — bahan sudah lengkap, tinggal disusun.
2. IndoBERT menggantikan TF-IDF — kemungkinan +1–3pp, tapi ruangnya sempit
   karena plafon derau label.
3. Fitur visual (cut rate, warna, wajah) — prioritas rendah: fitur numerik
   sudah terbukti merugikan pada dataset ini.

---

## 16. IndoBERT Diuji — dan Kalah. Keputusan: Tetap TF-IDF

Diuji lengkap dalam dua bentuk pada **skema validasi silang yang sama persis**
(StratifiedGroupKFold 5 fold, `groups=key`, seed sama, metrik sama).

| Model | Konfigurasi | Akurasi | macro-F1 |
|---|---|---|---|
| Baseline tebak `Surprise` | — | 41,2% | 0,058 |
| **TF-IDF + LogReg** | alpha=0,0 | **42,6%** | 0,092 |
| **TF-IDF + LogReg** | **alpha=0,5 (dipakai)** | **42,0%** | **0,099** |
| **TF-IDF + LogReg** | alpha=1,0 | 39,7% | **0,155** |
| IndoBERT beku + LogReg | alpha=0,0 | 36,0% | 0,147 |
| IndoBERT beku + LogReg | alpha=0,5 | 34,6% | 0,151 |
| IndoBERT beku + LogReg | alpha=1,0 | 29,4% | 0,144 |
| TF-IDF + IndoBERT digabung | alpha=0,5 | 36,6% | 0,149 |
| Ensemble probabilitas 0,5/0,5 | alpha=0,5 | 35,9% | 0,154 |
| **IndoBERT fine-tuned** | 4 epoch, lr 2e-5 | 37,4% | 0,129 |

**TF-IDF mendominasi Pareto.** Pada macro-F1 setara atau lebih tinggi (0,155
lawan 0,151 terbaik IndoBERT), TF-IDF unggul 5 poin akurasi. Tidak ada satu pun
titik di mana IndoBERT lebih baik pada kedua metrik sekaligus.

### Mengapa model bahasa besar justru kalah

Empat sebab, dan semuanya khas untuk dataset seperti ini:

1. **Keunggulan IndoBERT ada di tempat yang tidak dibutuhkan.** Pemahaman
   konteks, urutan kata, dan negasi berguna untuk nuansa. Tapi label di sini
   melacak **topik** (§5), dan untuk mengenali topik, hitungan kata sudah
   mendekati optimal — kata `dumbbell` menandakan konten kebugaran di posisi
   mana pun ia muncul.

2. **803 baris terlalu sedikit untuk 124 juta parameter.** Terlihat dari
   ragamnya antar-fold pada fine-tuning: **30,7% sampai 40,1%** — selisih 9,4
   poin antar fold, tanda ketidakstabilan yang jelas.

3. **Label bernoise.** Dengan plafon kesepakatan anotator ~56% (§13), kapasitas
   tambahan justru dipakai menghafal derau, bukan mempelajari pola.

4. **Chunking meredam keunggulan arsitektur.** 22,5% teks melebihi batas 512
   token, sehingga harus dipecah lalu dirata-rata — dan perataan itu sendiri
   sudah menyerupai bag-of-words.

### Nilai temuan ini untuk laporan

Hasil negatif ini **lebih berharga daripada kalau IndoBERT menang tipis**. Ia
memberi pernyataan yang bisa dipertahankan dan tidak umum:

> *Pada data kecil dengan label bernoise dan sinyal yang bersifat topik,
> bag-of-words mengalahkan model bahasa terlatih — baik dalam bentuk beku maupun
> setelah fine-tuning.*

Menjawab kriteria Kreativitas & Inovasi (metode diuji, bukan diasumsikan) dan
Interpretasi & Insight (kegagalannya dijelaskan mekanismenya, bukan sekadar
dilaporkan).

Berkas: `outputs/perbandingan_indobert.csv`,
kode: `src/features_bert.py` dan `src/finetune_bert.py`.

**Model final tidak berubah: TF-IDF (caption + transkrip) → Logistic Regression,
alpha = 0,5.**

---

## 17. Fitur Konsep Semantik — Model Final Berubah

Berawal dari satu pengamatan manual: video berlabel `Sad`
(`ig_DL1oBjdRc3-`) isinya *"mobilnya akan kita tinggal... semua barang di mobil
ini kita tinggal... bahan bakar ternyata gak kepake"*. **Tidak ada satu pun kata
emosi di sana** — tapi idenya jelas: kehilangan.

### Kenapa ini berbeda dari leksikon yang gagal di §5

| | Leksikon (§5, GAGAL) | Konsep (§17, BERHASIL) |
|---|---|---|
| Pemetaan | kata → **emosi** langsung | kata → **ide** → biar model yang belajar |
| Contoh | `bangga` → Proud | `ditinggal` → kehilangan → ? |
| Masalah | tiap emosi butuh kosakata sendiri | satu ide muncul lewat kata berbeda di topik berbeda |

Lapisan perantara itu yang membuatnya bekerja. **Kesimpulan §5 ("label melacak
topik, bukan emosi") karena itu perlu dikoreksi: label melacak topik DAN konsep
semantik — keduanya, bukan salah satu.**

### Hasil

| Tahap | Akurasi | macro-F1 |
|---|---|---|
| Baseline | 41,2% | 0,058 |
| 8. teks saja, bobot^0,5 | 42,0% | 0,099 |
| **9. + 9 fitur konsep (MODEL FINAL)** | **42,3%** | **0,135** |

**Menang di kedua metrik.** macro-F1 +36%, dan kelas yang tadinya ber-F1 nol
mulai terdeteksi (Anger 0,00 → 0,16; Fear 0,00 → 0,10).

Kokoh di tiga seed split berbeda: macro-F1 naik +0,035, +0,019, +0,042.

### Dua pelajaran yang berlawanan dengan dugaan

**1. Daftar kata lebih panjang justru LEBIH BURUK.** Setelah membaca puluhan
transkrip, daftar diperluas jadi 12 konsep dengan kosakata jauh lebih kaya.
Hasilnya turun: 41,1% vs 42,3%. Sebabnya kata umum (`indonesia`, `yuk`, `cara`,
`penting`) muncul di semua kelas dan hanya menambah derau. **Presisi mengalahkan
cakupan** — jangan panjangkan daftar tanpa mengukur ulang.

**2. Penambangan statistik TIDAK BISA menemukan konsep ini.** Dicoba n-gram 1-4
dengan log-odds: frasa `kita tinggal` cuma muncul di 1 video, jadi tak terlihat.
Lebih buruk lagi, kata benihnya ambigu — `tinggal` juga berarti "cukup/hanya"
(`tinggal pake blush`), `parah` juga berarti "keren" dalam slang. Konsep semantik
harus dirumuskan manusia yang membaca; statistik hanya bisa memvalidasinya.

### Bug yang sempat menyamarkan hasilnya

Implementasi pertama menjumlahkan `str.count()` tiap kata secara terpisah,
sehingga kata bersarang terhitung dua kali: `"masalah"` dihitung sebagai `salah`
DAN `masalah`, `"pemenang"` sebagai `menang` DAN `pemenang`. `kon_kegagalan`
menggelembung dari 437 jadi 566 dan macro-F1 turun 0,135 → 0,117.

Diperbaiki dengan regex alternation yang mencocokkan tanpa tumpang tindih.
**Untuk fitur berbasis daftar kata, selalu pakai satu regex alternation.**

### Catatan penting soal ragam akurasi

Uji tiga seed mengungkap bahwa model yang **sama persis** menghasilkan akurasi
40,2%-42,0% tergantung pembagian fold. Artinya klaim "42,0% mengalahkan baseline
41,2%" tidak pernah kokoh — selisih 0,8 poin ada di dalam derau.

**macro-F1 jauh lebih stabil** dan karena itu dipakai sebagai dasar keputusan.
Sebutkan keterbatasan ini di laporan, jangan mengklaim presisi yang tidak ada.

---

## 18. IndoBERT Percobaan Kedua — Resep Lengkap, Tetap Kalah

§16 menguji IndoBERT dengan resep polos: epoch tetap 4, tanpa validasi internal,
tanpa augmentasi, padding penuh 512. Kekalahannya (37,4% / 0,129) karena itu
belum menutup pertanyaan — bisa saja yang gagal resepnya, bukan modelnya.

Percobaan kedua memakai resep yang **terbukti pada proyek pembanding**
(klasifikasi tabel data governance: 347 sampel, 4 kelas, test accuracy 71%).
Empat hal diubah sekaligus:

| | v1 (§16) | v2 |
|---|---|---|
| Augmentasi | tidak ada | subsample 50–100% kalimat, 4 varian → 803 jadi 3.228 |
| Pemilihan epoch | tetap 4 (tebakan) | early stopping pada macro-F1 validasi dalam, patience 2 |
| Guard epoch awal | — | `min_epochs` 2 |
| Learning rate | 2e-5 | 1e-5 |
| Padding | `max_length` 512 penuh | dinamis per batch |
| Mixed precision | FP16 + GradScaler | BF16 |

Augmentasinya padanan langsung: di proyek pembanding satu tabel dijadikan lima
varian dengan mengambil 50–100% atributnya acak; di sini teksnya adalah caption
+ transkrip Whisper — deretan kalimat — jadi variannya mengambil 50–100%
kalimat. Efek sampingnya kebetulan menyerang masalah lain: 22,5% teks melebihi
512 token, dan varian pendek membuat model melihat bagian transkrip yang
berbeda-beda alih-alih selalu 512 token pertama.

Skema validasinya identik dengan seluruh tahap lain (StratifiedGroupKFold 5
fold, `groups=key`, seed sama). Validasi dalam untuk early stopping dipotong
lagi ~20% dari fold-train saja, tetap dikelompokkan per `key`, sehingga fold
luar tidak pernah ikut memilih epoch.

### Hasil: kalah lebih telak dari v1

| Model | Akurasi | macro-F1 |
|---|---|---|
| Baseline tebak `Surprise` | 41,2% | 0,058 |
| **TF-IDF + konsep (MODEL FINAL)** | **42,3%** | **0,135** |
| **TF-IDF + LogReg, bobot^1,0** | **39,7%** | **0,155** |
| IndoBERT fine-tuned v1 | 37,4% | 0,129 |
| **IndoBERT v2 (resep lengkap)** | **33,1%** | **0,143** |

**v2 didominasi mutlak.** TF-IDF bobot^1,0 mengungguli v2 pada **kedua** metrik
sekaligus — 39,7% lawan 33,1% pada akurasi, 0,155 lawan 0,143 pada macro-F1.
Dibanding v1 sendiri, resep lengkap menaikkan macro-F1 tipis (0,129 → 0,143)
tapi menjatuhkan akurasi 4,3 poin (37,4% → 33,1%). Tidak ada satu titik pun di
mana IndoBERT lebih baik dari TF-IDF. Waktu: 65 menit di RTX 4050.

### Isolasi: dari empat perubahan, hanya augmentasi yang berpengaruh

v2 mengubah empat hal sekaligus, jadi kesimpulan apa pun soal penyebabnya masih
tebakan sampai dipisah. Dijalankan ulang dengan `--n-augment 0` — early
stopping, lr 1e-5, dan dynamic padding tetap aktif, hanya augmentasi yang
dimatikan (26 menit):

| Konfigurasi | Akurasi | macro-F1 |
|---|---|---|
| v1 polos (epoch tetap 4, lr 2e-5, padding 512) | 37,4% | 0,129 |
| + early stopping + lr 1e-5 + dynamic padding | **36,7%** | **0,135** |
| + augmentasi 4 varian (= v2 penuh) | **33,1%** | **0,143** |

**Tiga perubahan pertama praktis tidak berpengaruh** (−0,7 poin akurasi, +0,006
macro-F1 — keduanya di dalam derau antar-fold). **Augmentasilah yang merusak:**
−3,6 poin akurasi untuk +0,008 macro-F1. Konsisten pula per fold — augmentasi
menurunkan akurasi di 4 dari 5 fold:

| fold | acc tanpa aug. | acc dengan aug. | macro-F1 tanpa aug. | macro-F1 dengan aug. |
|---|---|---|---|---|
| 1 | 42,9% | 38,8% | 0,158 | 0,201 |
| 2 | 37,8% | 28,8% | 0,204 | 0,209 |
| 3 | 36,5% | 34,7% | 0,144 | 0,228 |
| 4 | 33,3% | 34,0% | 0,138 | 0,105 |
| 5 | 32,5% | 28,8% | 0,153 | 0,092 |

> **Koreksi.** Sebelum isolasi ini dijalankan, di sini sempat tertulis bahwa
> derau sinyal early stopping adalah **penyebab utama** kekalahan v2. Itu
> **salah**. Sinyal itu memang terbukti bernoise (tabel di bawah), tapi secara
> agregat early stopping tidak merugikan — v1 ke konfigurasi tanpa augmentasi
> adalah seri. Yang menjatuhkan v2 adalah augmentasi. Pelajaran metodologisnya
> sama dengan §5A: **jangan mengubah empat hal sekaligus lalu menjelaskan
> hasilnya** — tanpa isolasi, penjelasan yang masuk akal bisa saja menunjuk
> tersangka yang salah.

### Kenapa augmentasi yang berhasil di sana justru merugikan di sini

**Augmentasi melipatgandakan baris, bukan informasi.** 803 jadi 3.228 sampel,
tapi tetap dari 803 video yang sama dengan derau label yang sama. Di proyek
pembanding augmentasi berhasil karena dua syarat yang di sini tidak terpenuhi:
elemennya (daftar atribut tabel) benar-benar dapat dipertukarkan, dan labelnya
bersih — jadi tiap varian adalah sampel sah yang baru. Di sini tiap varian
adalah salinan derau yang sama, diulang empat kali, sehingga bobot efektif
setiap label keliru ikut berlipat empat.

Arah tukar-gulingnya juga khas: augmentasi menggeser model ke prediksi yang
lebih menyebar — macro-F1 naik tipis, akurasi jatuh — persis pola yang sudah
berulang di §14 dan §16 setiap kali model didorong menjauhi `Surprise`. Jadi
augmentasi di sini tidak menambah kemampuan; ia hanya menggeser titik operasi,
dan menggesernya ke arah yang lebih buruk daripada yang bisa dicapai TF-IDF
hanya dengan menaikkan `alpha`.

### Temuan yang berdiri sendiri: early stopping bukan alat netral di data kecil

Terpisah dari sebab kekalahan v2, sinyal early stopping-nya sendiri terbukti
tidak bisa dipercaya. Validasi dalam berisi ~100 baris untuk 10 kelas, dengan
41% menumpuk di `Surprise` — beberapa kelas hanya kebagian 0–1 baris. Memilih
checkpoint berdasarkan macro-F1 yang dihitung atas kelas bersampel nol berarti
memilih berdasarkan lemparan koin:

| fold | epoch terpilih | macro-F1 val-dalam | macro-F1 fold-luar | selisih |
|---|---|---|---|---|
| 1 | 3 | 0,269 | 0,201 | −0,068 |
| 2 | 2 | 0,207 | 0,209 | +0,002 |
| 3 | 6 | 0,314 | 0,228 | −0,086 |
| 4 | 2 | 0,179 | 0,105 | −0,074 |
| 5 | 3 | 0,277 | **0,092** | **−0,185** |

Epoch terpilih berhamburan (3, 2, 6, 2, 3) dan validasi dalam melebih-lebihkan
di 4 dari 5 fold. Proyek pembanding memakai validasi 35 baris — lebih kecil —
tapi untuk **4 kelas berlabel bersih**, jadi ~9 baris per kelas. Di sini rasio
itu runtuh. Validasi yang cukup besar untuk memilih epoch secara andal akan
memakan porsi data latih yang tidak bisa direlakan; yang cukup kecil untuk
direlakan memilih berdasarkan derau. Itu jepitan struktural, bukan soal tuning.

**Diagnosis pendukung: macro-F1 OOF gabungan (0,143) lebih RENDAH dari rata-rata
per fold (0,167).** Kalau tiap fold menemukan aturan minoritas yang sama,
menggabungkannya akan mempertahankan presisi. Yang terjadi sebaliknya — tiap
fold bertaruh pada kelas kecil yang **berbeda-beda**, jadi begitu digabung,
presisi per kelas rontok. Model tidak mempelajari aturan yang stabil. Ragam
antar-fold mengkonfirmasi: macro-F1 0,092–0,228, rentang 2,5x.

### Uji ketiga: transkrip saja, tanpa caption

Hipotesisnya masuk akal dan datanya mendukung di muka. Caption cuma menambah
**1,7 poin cakupan** di atas transkrip (87,5% → 88,3%) tapi menambah ~100 kata
median, dan itu melipatempatkan tingkat pemotongan di batas 512 token:

| Kolom teks | Token median | Terpotong di 512 |
|---|---|---|
| `text_transcript` | 210 | **5,7%** (40 dokumen) |
| `text_combined` | 360 | **25,0%** (177 dokumen) |

Kerugian ini **khusus diderita BERT** — TF-IDF tidak punya batas panjang, jadi
baginya menggabungkan itu gratis. Karena itu kesimpulan §14 (gabungan terbaik
untuk TF-IDF) memang belum tentu berlaku di sini dan layak diuji terpisah.

Dijalankan pada konfigurasi identik dengan run tanpa augmentasi, hanya kolom
teksnya diganti:

| Sumber teks | Cakupan | Terpotong | Akurasi | macro-F1 |
|---|---|---|---|---|
| `text_combined` | 88,3% | 25,0% | **36,7%** | **0,135** |
| `text_transcript` | 87,5% | 5,7% | **36,7%** | **0,111** |

**Akurasinya identik, macro-F1-nya turun 0,024.** Membuang caption tidak
membantu sama sekali.

> **Koreksi.** Argumen pemotongan token di atas benar besarannya tapi salah
> konsekuensinya, dan sebabnya ada di `build_dataset.py:47`: **caption
> diletakkan di DEPAN** transkrip. Jadi saat teks dipotong di 512, yang hilang
> selalu **ekor transkrip**, tidak pernah caption-nya. Menurunkan pemotongan
> dari 25% ke 5,7% karena itu bukan "menyelamatkan caption" — melainkan menukar
> caption dengan ekor transkrip. Hasilnya menjawab pertukaran itu: **caption
> lebih berharga daripada ekor transkrip yang dikorbankan untuknya.**

### Diagnosa lanjutan — dan koreksi besar atas penjelasan di atas

Uji di atas masih punya satu cacat sebagai alat penjelas: cakupan caption
(62,6%) dan transkrip (87,5%) berbeda, jadi selisih skornya bisa berasal dari
"berapa baris yang terisi", bukan dari "seberapa informatif teksnya". Diagnosa
di `src/diagnosa_sumber_teks.py` mengontrol itu — seluruh analisis dijalankan
pada **irisan 497 baris yang punya caption DAN transkrip sekaligus**, dengan CV
yang sama.

**Hasilnya: pada baris yang sama, ketiga sumber tidak bisa dibedakan.**

| Sumber (irisan 497 baris) | Kata median | Akurasi | macro-F1 |
|---|---|---|---|
| caption saja | 130 | 40,8% | 0,091 |
| transkrip saja | 155 | 41,0% | 0,081 |
| caption + transkrip | 307 | 41,0% | 0,079 |

Menggabungkan tidak menambah apa-apa. Sebabnya redundansi, dan itu terukur:

| | |
|---|---|
| Kedua model sepakat | **80,1%** baris |
| Benar keduanya | 181 baris |
| Benar caption saja | 22 baris |
| Benar transkrip saja | 23 baris |
| **Salah keduanya** | **271 baris (54,5%)** |
| Plafon oracle (pilih sumber terbaik tiap baris) | **45,5%** |

Bahkan penggabungan sempurna — mustahil dicapai, karena butuh tahu jawabannya
lebih dulu — hanya menghasilkan +4,4 poin di atas sumber tunggal terbaik.
Caption dan transkrip **bukan dua sinyal yang saling melengkapi; keduanya
sinyal yang sama diucapkan dua kali.**

> **Koreksi.** Di versi sebelumnya bagian ini menjelaskan selisih 0,135 lawan
> 0,111 dengan tesis "caption jarang tapi padat, di situlah penanda topik
> eksplisit berada, dan kelas kecil bergantung padanya". Dua hal salah di situ.
>
> **Pertama, arah ketajamannya terbalik.** Diukur dengan log-odds Dirichlet,
> kata pembeda transkrip justru **lebih** tajam daripada caption — z rata-rata
> **4,25 lawan 3,43**. Klaim bahwa caption punya penanda lebih eksplisit tidak
> didukung data.
>
> **Kedua, selisih yang saya jelaskan itu kemungkinan besar derau.** Per fold,
> transkrip menang di 1 dari 5; selisih rata-ratanya 0,026, sementara ragam
> macro-F1 antar-fold pada run yang sama mencapai 2,5x (§18). Angka sekecil itu
> tidak bisa dipisahkan dari derau oleh data ini. Saya menjelaskan selisih yang
> belum tentu ada — kesalahan yang sama bentuknya dengan koreksi 50,3%→41,7% di
> §5A: **ukur dulu apakah selisihnya nyata, baru cari sebabnya.**
>
> Yang tetap berdiri: caption **memang lebih padat per kata** (1,7x — macro-F1
> di atas baseline per 100 kata: 0,0256 lawan 0,0151). Tapi kepadatan itu tidak
> berubah jadi keunggulan skor, karena isinya redundan dengan transkrip.

### Kenapa transkrip saja tidak cukup — jawaban sebenarnya

Dugaan yang wajar: transkrip berisi apa yang diucapkan di video, jadi dari
kalimatnya mestinya bisa dibaca emosinya. Kata pembeda transkrip per kelas
membantah itu:

| Kelas | Kata paling khas di transkrip |
|---|---|
| Surprise | ikan, mobil, wangi, suv, juta, roda |
| Trust | kalian, ban, ala, gerakan, dumbbell, gym |
| Proud | bibir, pokoknya, teman, extra, warna, emas |
| Joy | jakarta, kawan, menang, fit, glory, sukses |
| Anger | jawa, orang, liter, speed, itu, kotor |

**Isinya kata topik, bukan kata emosi** — persis seperti temuan §5A pada
caption, dan sekarang terbukti berlaku untuk transkrip juga. Manusia yang
membaca transkrip memang bisa menyimpulkan emosinya; tapi anotator **tidak**
melabeli berdasarkan emosi dalam ucapan, melainkan berdasarkan jenis kontennya.
Dua fungsi yang berbeda, dan model hanya bisa mempelajari yang kedua.

Karena itu menukar sumber teks tidak akan menolong dari mana pun ia diambil:
ketiganya memetakan ke topik yang sama. Angka 271 baris yang salah di kedua
model (54,5%) adalah wajah lain dari plafon derau label ~56% di §13 — bukan
kekurangan sumber teks, melainkan batas yang dipasang oleh labelnya sendiri.

`text_combined` dipertahankan, tapi alasannya sekarang lebih jujur: **bukan
karena terbukti lebih baik, melainkan karena cakupannya paling luas (88,3%) dan
tidak ada bukti ia lebih buruk.**

### Perbandingan kondisi — kenapa 71% di sana dan 33% di sini

| | Proyek pembanding | BDC |
|---|---|---|
| Kelas | 4 | 10 (2 bersampel tunggal) |
| Sampel | 347 → 1.337 | 803 → 3.228 |
| Kualitas label | bersih, satu tabel satu label | plafon kesepakatan anotator ~56% |
| Sifat sinyal | deskripsi kolom yang memang menentukan jawaban | topik + konsep, tidak langsung |
| Hasil | 71,4% / 0,687 | 33,1% / 0,143 |

Bedanya bukan resep, melainkan tugasnya — dan isolasi di atas membuktikannya
secara langsung: teknik yang identik (augmentasi subsampling) menaikkan skor di
satu tugas dan menurunkannya di tugas lain. Yang menentukan bukan tekniknya,
melainkan apakah syarat yang membuatnya sah terpenuhi.

### Kesimpulan — pernyataan §16 jadi jauh lebih kuat

Sebelumnya pernyataannya bisa dibantah dengan "resepnya kurang bagus". Sekarang
tidak bisa:

> *Pada data kecil dengan label bernoise dan sinyal yang bersifat topik,
> bag-of-words mengalahkan model bahasa terlatih — beku, fine-tuned polos,
> maupun fine-tuned dengan resep lengkap (augmentasi, early stopping, LR
> tertala) yang terbukti mencapai 71% pada tugas 4 kelas berlabel bersih. Lebih
> jauh, komponen resep itu diisolasi satu per satu: augmentasi yang menaikkan
> skor di tugas bersih justru menurunkan akurasi 3,6 poin di sini.*

Dua temuan tambahan yang berdiri sendiri dan layak masuk laporan:

1. **Augmentasi data hanya sah kalau elemennya dapat dipertukarkan DAN labelnya
   bersih.** Kalau tidak, ia melipatgandakan derau, bukan informasi. Terukur:
   −3,6 poin akurasi, konsisten di 4 dari 5 fold.

2. **Pada dataset sekecil ini, early stopping bukan alat yang netral.** Validasi
   internal yang cukup besar untuk memilih epoch secara andal akan memakan porsi
   data latih yang tidak bisa direlakan; yang cukup kecil untuk direlakan
   memilih berdasarkan derau (melebih-lebihkan di 4 dari 5 fold). Ini alasan
   mekanistik kenapa teknik standar deep learning gagal berpindah ke rezim data
   kecil, bukan sekadar "datanya kurang".

**Model final tidak berubah: TF-IDF (caption + transkrip) + fitur konsep →
Logistic Regression, alpha = 0,5.**

### Ringkasan seluruh percobaan IndoBERT

| Percobaan | Sumber teks | Akurasi | macro-F1 |
|---|---|---|---|
| §16 beku + LogReg (alpha 0,5) | gabungan | 34,6% | 0,151 |
| §16 fine-tuned v1 (resep polos) | gabungan | 37,4% | 0,129 |
| §18 v2 resep lengkap | gabungan | 33,1% | 0,143 |
| §18 tanpa augmentasi | gabungan | **36,7%** | **0,135** |
| §18 tanpa augmentasi | transkrip saja | 36,7% | 0,111 |
| **TF-IDF + konsep (MODEL FINAL)** | gabungan | **42,3%** | **0,135** |
| **TF-IDF + LogReg bobot^1,0** | gabungan | **39,7%** | **0,155** |

Lima konfigurasi IndoBERT, tidak satu pun mendekati TF-IDF. Yang terbaik
(36,7% / 0,135) kalah 5,6 poin akurasi pada macro-F1 yang sama persis.

Berkas: `outputs/oof_indobert_v2.csv`, `outputs/jejak_indobert_v2.csv`,
`outputs/oof_indobert_noaug.csv`, `outputs/jejak_indobert_noaug.csv`,
`outputs/oof_indobert_transkrip.csv`, `outputs/jejak_indobert_transkrip.csv`,
kode: `src/finetune_bert_v2.py` (`--n-augment 0` untuk tanpa augmentasi,
`--kolom-teks text_transcript` untuk transkrip saja).

---

## 19. Lapisan Emosi — Sinyalnya NYATA, dan §5A Perlu Dikoreksi Lagi

Seluruh proyek ini menyimpulkan label melacak TOPIK (§5A, §17, §18). Tapi
kesimpulan itu selalu diukur lewat model yang memang mencari topik — TF-IDF dan
IndoBERT pada teks penuh. Sinyal emosi tidak pernah diberi kesempatan berdiri
sendiri, dan leksikon yang gagal di §5A punya cacat yang belum pernah diperiksa:
**dilusi**.

Transkrip median 160 kata. Kalimat *"aku sedih deh harus tinggal di sini"*
adalah 7 kata dari 160 — TF-IDF dengan `sublinear_tf` nyaris tidak mencatatnya
sementara 153 kata topik menenggelamkannya. Leksikon §5A dihitung dengan cara
yang sama: menjumlahkan seluruh dokumen lalu membagi panjang. Jadi kegagalan itu
**belum pernah membuktikan kata emosinya tidak ada** — baru membuktikan bahwa
kalau ada, ia tenggelam.

### Rancangan: tiga lapis yang semuanya menyerang emosi

**A. Leksikon InSet** (Koto & Rahmaningtyas, IALP 2017): 3.609 kata positif dan
6.609 negatif, bobot −5..+5. Dipakai berbobot, **dengan penanganan negasi** —
perbaikan sadar atas rujukan (Zakira dkk., JIEnGS 2025) yang hanya menghitung
kata positif lawan negatif, sehingga "aku tidak sedih" terhitung sedih. Terukur:
`"aku sedih deh"` = −14,0, `"aku tidak sedih kok"` = −1,0.

**B. Ekstraksi kalimat emosional** — serangan langsung ke dilusi. Tiap segmen
diberi skor emosional; hanya yang tertinggi disimpan. Dokumen 288 kata jadi 102
kata (2,8x).

> **Jebakan yang hampir melewatkan ini.** Versi pertama memakai batas kalimat
> `[.!?]` dan mengambil 5 kalimat teratas — pemadatannya cuma 419→375 kata,
> praktis nihil. Sebabnya Whisper tidak memberi tanda baca: **15,2% "kalimat"
> memuat 62,6% seluruh kata**, ada yang panjangnya 745 kata. Diperbaiki dengan
> memecah segmen panjang jadi jendela 25 kata dan mengambil porsi relatif (30%),
> bukan jumlah tetap. **Untuk teks hasil ASR, jangan pernah percaya batas
> kalimat.**

Validasinya kuat: video `Sad` yang di §17 ditemukan **manual** karena
penambangan statistik tidak sanggup, sekarang terekstrak otomatis — potongan
*"mobilnya akan kita tinggal ... semua barang di mobil ini kita tinggal"* naik
sendiri ke permukaan.

**C. Probabilitas model emosi eksternal.** Ini kuncinya. Model-model ini dilatih
pada teks Indonesia **berlabel emosi bersih**, jadi keluarannya menjawab "teks
ini mengekspresikan emosi apa" — lepas sepenuhnya dari label topik kita yang
bernoise. Kebetulan yang menguntungkan: label kita adalah 6 dari 8 emosi dasar
Plutchik.

| Model | Kelas | Menutup label kita |
|---|---|---|
| NusaBERT Plutchik | Anger, Anticipation, Disgust, Fear, Joy, Sadness, Surprise, Trust | Surprise, Trust, Joy, Anger, Sad, Fear |
| IndoBERT thoriqfy | Sadness, Anger, Love, Fear, Happy, Neutral | + Love, Neutral |

Delapan dari sepuluh label tertutup. Probabilitas diambil **max-pooling antar
potongan, bukan mean**: emosi adalah puncak, bukan rata-rata.

### Temuan utama: sinyal emosinya NYATA

Untuk tiap label, seberapa tinggi model Plutchik membaca emosi **senama** di
video itu dibanding video lain:

| Label kita | n | prob senama | prob di kelas lain | selisih | Mann-Whitney |
|---|---|---|---|---|---|
| Surprise | 331 | 0,207 | 0,104 | +0,103 | p = 3,5e−06 |
| Trust | 183 | 0,227 | 0,154 | +0,073 | p = 0,015 |
| Joy | 53 | 0,305 | 0,194 | +0,111 | p = 0,026 |
| Anger | 36 | 0,077 | 0,008 | +0,068 | p = 3,5e−04 |
| Sad | 18 | 0,092 | 0,026 | +0,065 | p = 0,21 |
| Fear | 16 | 0,266 | 0,233 | +0,032 | p = 0,39 |

**Enam dari enam arahnya benar** (uji tanda p = 0,016). Setelah koreksi Holm
untuk 6 uji, dua bertahan: Surprise (p = 2,1e−05) dan Anger (p = 0,0017). Sad
dan Fear bersampel 18 dan 16 — terlalu kecil untuk diputuskan, bukan terbantah.

**Ini bukti terukur pertama di seluruh proyek bahwa label melacak emosi.**
Kesimpulan §5A karena itu perlu dikoreksi untuk kedua kalinya:

> Label melacak **topik secara kuat, konsep semantik secara sedang (§17), dan
> emosi secara lemah tapi terukur.** Ketiganya, bukan salah satunya. Yang gagal
> di §5A bukan gagasan emosinya — melainkan cara mengukurnya, yang membiarkan
> sinyal itu terdilusi.

### Tapi tidak cukup untuk menang

| Tahap | Akurasi | macro-F1 |
|---|---|---|
| baseline tebak `Surprise` | 41,2% | 0,058 |
| **0. acuan: TF-IDF + konsep (MODEL FINAL)** | **42,3%** | **0,135** |
| 1. probabilitas emosi SAJA (padat) | 35,5% | 0,113 |
| 1b. probabilitas emosi SAJA (penuh) | 35,6% | 0,114 |
| 2. leksikon InSet SAJA | 31,3% | 0,080 |
| 3. emosi + InSet | 32,5% | 0,134 |
| 4. TF-IDF pada teks PADAT saja | 39,6% | 0,084 |
| 5. acuan + emosi padat | 42,7% | 0,135 |
| 6. acuan + emosi penuh | 40,2% | 0,135 |
| 7. acuan + emosi padat + InSet | 39,7% | 0,139 |

Emosi sendirian mencapai macro-F1 **0,113 — hampir 2x baseline (0,058)** tanpa
melihat satu pun kata topik. Sinyalnya nyata. Tapi akurasinya 35,5%, jauh di
bawah topik (42,3%): **emosi ada, tapi jauh lebih lemah daripada topik.**

Tahap 5 (42,7%) dan tahap 7 (0,139) sedikit melampaui acuan, **tapi keduanya di
dalam derau** — §17 mengukur model yang sama persis berayun 40,2–42,0% antar
seed. Selisih +0,4 poin dan +0,004 tidak bisa diklaim. **Lapisan emosi tidak
memperbaiki model.**

### Dua pelajaran mekanistik

**1. Hipotesis dilusi benar, tapi max-pooling sudah menyelesaikannya.**
Pemadatan tidak menambah apa-apa bagi model emosi: padat 0,113 lawan penuh
0,114 — identik. Sebabnya max-pooling **sudah** merupakan obat dilusi: ia
mengambil potongan yang memuncak, bukan merata-ratakan. Pemadatan adalah
implementasi kedua dari gagasan yang sama, jadi wajar tidak menambah.

**2. Pemadatan MERUGIKAN sinyal topik.** TF-IDF pada teks padat jatuh ke 39,6% /
0,084 dari 42,3% / 0,135. Masuk akal dan menegaskan pembagian perannya: memilih
segmen berdasarkan muatan emosi berarti **membuang** justru kata-kata topik yang
selama ini jadi tulang punggung prediksi.

### Nilainya untuk laporan

Hasil ini lebih berharga daripada kenaikan akurasi kecil, karena mengubah
pernyataan proyek dari tebakan jadi terukur:

> *Pada dataset ini label melacak topik secara dominan dan emosi secara lemah
> tapi terukur (6/6 arah benar, uji tanda p = 0,016; Surprise dan Anger bertahan
> koreksi Holm). Sinyal emosi yang berdiri sendiri mencapai macro-F1 dua kali
> baseline tanpa melihat kata topik sama sekali — namun tetap 7 poin akurasi di
> bawah sinyal topik, dan menggabungkan keduanya tidak melampaui derau.*

Menjawab tiga kriteria sekaligus: Kreativitas (leksikon terbitan + transfer dari
dataset emosi eksternal + ekstraksi kalimat), Interpretasi (mekanisme dilusi
diukur, bukan dikira), dan Kesesuaian Solusi (premis diuji, tidak diasumsikan).

**Model final tidak berubah: TF-IDF (caption + transkrip) + fitur konsep →
Logistic Regression, alpha = 0,5.**

Berkas: `outputs/ablation_emosi.csv`, `data/leksikon/inset_*.tsv`,
kode: `src/features_emosi.py`, `src/eval_emosi.py`.

---

## 20. MODEL FINAL BERUBAH — Lapisan Emosi Diterima + Kalibrasi Keputusan

§19 menyimpulkan lapisan emosi "tidak memperbaiki model". Kesimpulan itu diambil
dari **satu split** dan ternyata terlalu cepat. Diuji pada 5 seed dengan
protokol §17, gambarannya berbeda.

### Bagian 1: lapisan emosi memang menaikkan macro-F1

| Model (5 seed, mean ± std) | Akurasi | macro-F1 |
|---|---|---|
| acuan: topik saja | 41,6% ± 0,6 | 0,124 ± 0,008 |
| topik + emosi | 41,6% ± 0,8 | 0,127 ± 0,008 |
| topik + emosi + InSet | 38,8% ± 0,9 | **0,135 ± 0,003** |

Topik + emosi + InSet menang macro-F1 di **5 dari 5 seed** (+0,003, +0,019,
+0,008, +0,016, +0,005), dan ragamnya paling kecil (±0,003 lawan ±0,008). Itu
persis standar yang dipakai §17 untuk menerima fitur konsep. Tapi akurasinya
jatuh 2,8 poin, jadi bobotnya disapu:

| Konfigurasi (3 seed) | Akurasi | macro-F1 |
|---|---|---|
| acuan | 41,6% | 0,126 |
| **+ emosi, bobot 0,25** | **41,7%** | **0,129** |
| + emosi, bobot 0,50 | 40,2% | 0,129 |
| + emosi + InSet, bobot 0,25 | 39,2% | 0,137 |

**Bobot 0,25 tanpa InSet menang di kedua metrik** sekaligus. Fitur InSet
dikeluarkan dari model final: sumbangannya nyata tapi harganya akurasi, dan
lapisan probabilitas emosi sudah membawa sebagian besar sinyalnya.

### Bagian 2: dua penyakit keputusan yang lebih besar dari soal fitur

Diagnosa galat pada OOF membongkar dua kerugian yang **sama sekali bukan soal
kurangnya sinyal**, melainkan soal cara keputusan diambil:

**Penyakit 1 — over-prediksi kelas mayoritas, parah.**

| Kelas | Sebenarnya | Diprediksi |
|---|---|---|
| Surprise | 331 | **482** |
| Fear | 16 | **3** |
| Sad | 18 | **3** |
| Neutral | 8 | 2 |

Argmax pada probabilitas yang condong ke prior membuat kelas kecil tidak pernah
menang — dan macro-F1 menghukum itu tepat sasaran. Obatnya koreksi prior:

    prediksi = argmax  P(y|x) / P(y)^tau

**tau ditala DI DALAM fold** memakai validasi dalam; menalanya pada fold yang
dinilai akan membocorkan jawaban.

**Penyakit 2 — baris tanpa teks lebih buruk daripada tebakan buta.** 94 baris
train (11,7%) tidak punya caption maupun transkrip. Akurasi model di sana
**27,7%**, sementara menebak `Surprise` untuk semuanya memberi ~41%. Model
memaksakan tebakan dari fitur kosong dan **kalah dari prior**. §13 sudah
menuliskan keharusan fallback ini; baru sekarang diterapkan pada penilaian.

Dekomposisi galatnya:

| Kelompok | n | Akurasi | Porsi seluruh galat |
|---|---|---|---|
| teks kosong | 94 | 27,7% | 14,0% |
| teks ada | 709 | 41,3% | 86,0% |
| label bentrok | 35 | 20,0% | — |

### Kurva tau — dial akurasi lawan macro-F1

Topik + emosi + fallback, 3 seed:

| tau | Akurasi | macro-F1 | |
|---|---|---|---|
| 0,0 | 42,0% | 0,129 | |
| **0,1** | **41,4%** | **0,137** | **dipilih — di atas baseline** |
| 0,2 | 40,6% | 0,142 | |
| 0,3 | 39,2% | 0,155 | setara rekor lama |
| 0,4 | 38,2% | 0,162 | |
| **0,5** | 37,1% | **0,180** | **macro-F1 tertinggi proyek ini** |
| 0,6 | 33,7% | 0,179 | |

**tau = 0,1 dipilih** dengan alasan yang sama seperti alpha = 0,5 di §14:
satu-satunya titik yang menaikkan macro-F1 secara meyakinkan (+0,011, di luar
pita derau ±0,008) sambil menjaga akurasi tetap **di atas baseline 41,2%**.

Kalau juri lebih menghargai macro-F1 dan distribusi realistis, naikkan
`TAU_PRIOR = 0.5` di `predict.py` — satu baris, macro-F1 jadi **0,180** (naik
43% dari 0,126) dengan akurasi 37,1%. Kurva lengkapnya ada di atas supaya
keputusan itu bisa diambil sadar, bukan ditebak.

### Hasil negatif: membersihkan label bentrok TIDAK menolong

35 baris train berlabel bentrok pada video yang sama (§11) dibuang dari data
**latih** (tetap dinilai), diuji pada tiga tau:

| tau | Tanpa dibersihkan | Dibersihkan | Selisih |
|---|---|---|---|
| 0,0 | 42,0% / 0,129 | 41,9% / 0,125 | −0,0 / **−0,005** |
| 0,2 | 40,6% / 0,142 | 40,1% / 0,137 | −0,5 / **−0,005** |
| 0,4 | 38,2% / 0,162 | 38,1% / 0,160 | −0,1 / **−0,002** |

Konsisten sedikit merugikan. Penjelasannya masuk akal: baris berlabel bentrok
tetap membawa **sinyal topik yang sah** — labelnya cuma salah satu dari beberapa
yang sama-sama masuk akal, bukan acak. Membuangnya berarti membuang 35 baris
data topik demi menghindari derau label yang ternyata lebih kecil daripada
manfaatnya.

### Model final baru

```
TF-IDF(caption + transkrip) + 9 fitur konsep + 14 probabilitas emosi
  -> Logistic Regression, bobot kelas^0,5
  -> koreksi prior tau=0,1
  -> fallback ke prior untuk baris tanpa sinyal
  -> penimpaan duplikat URL (§1.2)
```

| | Akurasi | macro-F1 |
|---|---|---|
| baseline tebak `Surprise` | 41,2% | 0,058 |
| model final LAMA (§17) | 41,6% | 0,126 |
| **model final BARU** | **41,4%** | **0,137** |

macro-F1 naik **+8,7%** dengan akurasi praktis tak berubah (−0,2 poin, di dalam
derau ±0,6). Distribusi submission juga jadi lebih sehat:

| | Lama | Baru |
|---|---|---|
| `Surprise` | 149 (74,5%) | **132 (66,0%)** |
| Kelas terwakili | 6 | **7** (`Sad` muncul) |

### Pelajaran

**Kalibrasi keputusan lebih murah dan lebih besar dampaknya daripada menambah
fitur.** Seluruh percobaan fitur di §16–§19 — IndoBERT beku, fine-tuned, resep
lengkap, transkrip-saja, lapisan emosi — menghasilkan pergeseran macro-F1 di
kisaran ±0,01. Satu baris koreksi prior menggeser 0,126 → 0,180. Pada masalah
yang tidak seimbang dan dinilai dengan macro-F1, **cara memilih kelas adalah
hyperparameter, dan ia sering lebih penting daripada representasi teksnya.**

Berkas: `outputs/kalibrasi.csv`, `outputs/oof_kalibrasi.csv`,
kode: `src/kalibrasi.py`, `TAU_PRIOR` di `src/predict.py`.

---

## 21. Stemming — Berguna, Tapi Hanya Kalau Bigram Dibuang

Bahasa Indonesia sangat berimbuhan: `menang`, `pemenang`, `kemenangan`,
`memenangkan` berakar sama tapi jadi empat token berbeda. Dengan 803 baris dan
`min_df=2`, varian yang sendirian jarang akan terbuang padahal gabungannya cukup
sering. Jurnal rujukan (Zakira dkk., JIEnGS 2025) memakai stemming sebagai
langkah baku.

Risikonya juga nyata dan sudah terdokumentasi di §17 — dan terverifikasi pada
Sastrawi:

    "mobilnya ditinggalkan"  ->  "mobil tinggal"
    "tinggal pake blush"     ->  "tinggal pake blush"

Dua makna `tinggal` yang berbeda kini jadi token identik. Jadi jawabannya harus
datang dari data.

### Percobaan pertama: tampak tidak berguna

| Konfigurasi (3 seed) | Dimensi | Akurasi | macro-F1 |
|---|---|---|---|
| asli, uni+bigram | 22.808 | 41,7% | 0,129 |
| stemming, uni+bigram | 22.431 | 41,6% | 0,130 |
| char n-gram 3-5 | 63.980 | 41,2% | 0,122 |
| stemming + char n-gram | 53.336 | 41,6% | 0,126 |

Stemming praktis nihil (+0,001 macro-F1), char n-gram justru merugikan dengan
3x dimensi. Kesimpulan yang mudah diambil: "stemming tidak berguna di sini."

**Kesimpulan itu salah**, dan satu angka membongkarnya.

### Kenapa efeknya hilang: bigram menelannya

| TF-IDF | Kosakata asli | Setelah stemming | Menyusut |
|---|---|---|---|
| **unigram saja** | 8.913 | 7.363 | **17,4%** |
| unigram + bigram | 27.833 | 27.378 | **1,6%** |

Stemming bekerja persis seperti yang diharapkan — **pada unigram**. Tapi model
memakai `ngram_range=(1,2)`, dan bigram menguasai **68%** ruang fitur. Bigram
nyaris tidak bertabrakan setelah di-stem: `mobil listrik` tetap `mobil listrik`,
dan kombinasinya terlalu beragam untuk menyatu. Jadi penyusutan 17,4% pada
unigram terencerkan jadi 1,6% pada keseluruhan, dan manfaatnya ikut hilang.

**Stemming dan unigram harus berpasangan.** Diuji terpisah:

| Konfigurasi (3 seed) | Akurasi | macro-F1 |
|---|---|---|
| unigram, asli | 41,0% | 0,132 |
| **unigram, STEMMED** | **41,6%** | **0,138** |
| uni+bigram, asli | 41,7% | 0,129 |
| uni+bigram, stemmed | 41,6% | 0,130 |

Pada unigram, stemming menang di **kedua** metrik. Diuji ulang 5 seed dengan
kalibrasi §20 terpasang:

| Model (5 seed) | Akurasi | macro-F1 |
|---|---|---|
| uni+bigram asli | 41,4% ± 0,4 | 0,136 ± 0,008 |
| **unigram + stemming** | **41,6% ± 0,8** | **0,149 ± 0,009** |

**macro-F1 menang 5 dari 5 seed** (+0,018, +0,008, +0,001, +0,016, +0,018) —
standar penerimaan yang sama dengan §17 dan §19. Diterima.

### Kenapa ambiguitas `tinggal` ternyata tidak merusak

Kekhawatiran §17 masuk akal tapi tidak terwujud, dan sebabnya bisa dijelaskan:
kata ambigu seperti `tinggal` dan `parah` **jumlahnya sedikit**, sementara
penggabungan varian berimbuhan menyentuh 17,4% kosakata. Kerugian dari segelintir
tabrakan makna kalah jauh dari keuntungan menyatukan ribuan varian yang
sebelumnya masing-masing terlalu jarang untuk lolos `min_df=2`. Pada data
sebesar ini, **masalahnya kelangkaan, bukan ambiguitas.**

### Tangga ablation final

| Tahap | Akurasi | macro-F1 | weighted-F1 |
|---|---|---|---|
| 1. baseline (kelas mayoritas) | 41,2% | 0,058 | 0,241 |
| 2. + metadata | 29,3% | 0,105 | 0,288 |
| 3. + fitur teks (hitungan) | 31,4% | 0,118 | 0,302 |
| 4. + TF-IDF caption | 34,1% | 0,119 | 0,310 |
| 5. + uploader | 35,1% | 0,132 | 0,322 |
| 6. + audio (numerik) | 36,1% | 0,126 | 0,329 |
| 7. + transkrip ke TF-IDF | 38,2% | 0,127 | 0,346 |
| 8. teks saja, bobot^1,0 | 39,7% | 0,155 | 0,361 |
| 9. + fitur konsep (§17) | 42,3% | 0,135 | 0,354 |
| 10. + lapisan emosi (§19) | 42,7% | 0,135 | 0,364 |
| **11. + stemming, unigram (FINAL)** | **43,1%** | **0,148** | **0,376** |

Tahap 11 memberi **akurasi tertinggi DAN weighted-F1 tertinggi** di seluruh
tabel, dengan macro-F1 hanya 0,007 di bawah tahap 8 yang akurasinya 3,4 poin
lebih rendah.

### Model final

```
stemming Sastrawi -> TF-IDF unigram (caption + transkrip)
  + 9 fitur konsep semantik + 14 probabilitas emosi
  -> Logistic Regression, bobot kelas^0,5
  -> koreksi prior tau=0,1  ->  fallback prior  ->  penimpaan duplikat URL
```

| | Akurasi | macro-F1 |
|---|---|---|
| baseline tebak `Surprise` | 41,2% | 0,058 |
| model final §17 (sebelum hari ini) | 42,3% | 0,135 |
| **model final sekarang** | **43,1%** | **0,148** |

Distribusi submission juga terus membaik sepanjang tiga perbaikan hari ini:

| | §17 | + emosi & tau (§20) | + stemming (§21) |
|---|---|---|---|
| `Surprise` | 149 (74,5%) | 132 (66,0%) | **121 (60,5%)** |
| Kelas terwakili | 6 | 7 | **7** |

Bandingkan dengan train: `Surprise` 41,2%. Distribusinya masih timpang, tapi
jaraknya menyusut dari +33 poin jadi +19 poin.

### Pelajaran

**Preprocessing tidak bisa dinilai sendirian — ia berinteraksi dengan representasi.**
Stemming dinyatakan "tidak berguna" pada uji pertama, padahal yang salah bukan
stemming-nya melainkan pasangannya. Kalau uji itu berhenti di tabel pertama,
+0,013 macro-F1 hilang tanpa pernah terlihat. **Ketika sebuah teknik yang
seharusnya bekerja ternyata tidak, periksa apakah ada komponen lain yang
menelan efeknya sebelum menyimpulkan teknik itu tidak berguna.**

Berkas: `outputs/uji_stemming.csv`, `data/features/teks_stem.parquet`,
kode: `src/uji_stemming.py`, `stem_kolom()` di `src/build_dataset.py`,
`KOLOM_TEKS_FINAL`/`NGRAM_FINAL` di `src/train.py`.

---

## 22. Apa yang Masih Menahan — Bedah Sisa Galat

Model final: **42,6% / macro-F1 0,160** (seed 42, dengan kalibrasi §20). Sisa
galatnya dibedah untuk tahu mana yang masih bisa diperbaiki dan mana yang tidak.

### Temuan utama: kebingungan model MENIRU ketidaksepakatan anotator

Delapan video dilabeli lebih dari satu kelas oleh anotator. Pasangan yang mereka
pertukarkan, dibandingkan dengan pasangan yang model pertukarkan:

| Pasangan | Ditukar ANOTATOR | Ditukar MODEL |
|---|---|---|
| Proud ↔ Surprise | 4 video | 112 baris |
| Proud ↔ Trust | 4 video | 67 baris |
| Surprise ↔ Trust | 3 video | 127 baris |
| (pasangan lain) | ≤1 video | jauh lebih jarang |

**Tiga pasangan yang sama persis, dan itulah satu-satunya pasangan yang
anotatornya tidak sepakat lebih dari sekali.** Model tidak gagal secara acak —
ia gagal tepat di tempat manusia gagal.

Ini bukti terkuat bahwa sisa galatnya adalah derau label, bukan kekurangan
model. Dan bebannya besar: Surprise + Trust + Proud = 670 dari 803 baris
(**83%**). Tiga kelas terbesar saling tertukar, dan anotatornya sendiri
menukarnya.

### Plafon struktural macro-F1: tiga kelas mustahil

| Kelas | n | F1 |
|---|---|---|
| Surprise | 331 | 0,592 |
| Trust | 183 | 0,311 |
| Proud | 156 | 0,256 |
| Anger | 36 | 0,185 |
| Fear | 16 | 0,095 |
| Sad | 18 | 0,083 |
| Joy | 53 | 0,077 |
| **Love** | **1** | **0,000** |
| **Loyalty** | **1** | **0,000** |
| **Neutral** | **8** | **0,000** |

Love, Loyalty, dan Neutral berjumlah **10 sampel (1,2% data)** tapi menempati
**30% penyebut macro-F1** — dan ketiganya permanen nol.

| Skema penilaian | macro-F1 | Selisih |
|---|---|---|
| 10 kelas (dipakai) | 0,160 | — |
| tanpa 2 singleton (Love, Loyalty) | 0,200 | **+0,040 (+25%)** |
| tanpa 3 kelas mustahil (+ Neutral) | 0,228 | **+0,069 (+43%)** |

**43% dari "kekurangan" macro-F1 bersifat struktural**, bukan kegagalan model.
Ini menjawab usulan §6 yang belum pernah dieksekusi: skor dilaporkan pada kedua
skema, submission tetap 10 kelas (ruang label terkunci oleh aturan lomba).

### Ringkasan: apa yang masih menahan, dan bisa/tidaknya diperbaiki

| Penahan | Besaran | Bisa diperbaiki? |
|---|---|---|
| Derau label Surprise/Trust/Proud | 83% data, plafon ~56% (§13) | **Tidak** — anotator sendiri tidak sepakat |
| 3 kelas bersampel 1–8 | −0,069 macro-F1 (43%) | **Tidak** untuk submission; dilaporkan sebagai analisis |
| 12,5% test tanpa sinyal | ~25 baris hanya bisa ditebak prior | **Tidak** — sumbernya sudah mati (§11) |
| Baris kosong di data latih | — | **Sudah optimal** — dibuang justru merugikan (uji di bawah) |
| Label bentrok di data latih | — | **Sudah optimal** — dibuang justru merugikan (§20) |

### Dua uji "membersihkan data" yang keduanya GAGAL

Naluri yang wajar: buang data kotor. Diuji dua kali, keduanya merugikan.

**Buang 94 baris tanpa teks dari data latih** (5 seed):

| | Akurasi | macro-F1 |
|---|---|---|
| dipertahankan | **41,6% ± 0,8** | **0,149 ± 0,009** |
| dibuang | 41,1% ± 0,7 | 0,148 ± 0,011 |

Kalah di 4 dari 5 seed pada kedua metrik. Sebabnya: baris kosong tetap
mengajarkan **distribusi kelas** (Surprise 37, Trust 24, Proud 20, ...) yang
ikut mengkalibrasi model, meski teksnya nihil.

**Buang 35 baris label bentrok** (§20): juga merugikan, −0,005 macro-F1 konsisten
di tiga nilai tau.

**Pelajaran: pada data sekecil ini, membuang baris "kotor" lebih mahal daripada
derau yang dibawanya.** Baris cacat tetap membawa informasi prior dan topik yang
sah. Perlakuan yang benar bukan menghapus, melainkan **menangani saat prediksi**
— fallback ke prior untuk baris tanpa sinyal (§20), bukan mengeluarkannya dari
pelatihan.

### Jawaban soal video rusak / folder

- **75 Google Drive terhapus, 24 CDN kadaluarsa, 16 Instagram mati** → barisnya
  **dipertahankan**, teks kosong, prediksi mundur ke prior. Sudah diverifikasi
  optimal lewat uji di atas.
- **1 folder Drive (test id 107)** berisi 41 video, 0 cocok dengan yang mati,
  8 tidak ada di dataset (§11). Informasi mana yang dimaksud hilang di sumber,
  jadi **dibiarkan diprediksi model** — tidak ada dasar untuk memilih satu.

Berkas: `outputs/oof_final.csv`, kode: analisis di `src/kalibrasi.py` dan
tangga di `src/train.py`.

---

## 23. Aturan Pembeda Manusia + Ketimpangan Kelas — Dua Jalan Buntu yang Berguna

Dua saran diuji pada ronde terakhir. Keduanya ditolak, tapi keduanya
menghasilkan aturan yang bisa dipakai orang lain.

### A. Konsep gelombang kedua dari aturan pembeda manusia

§22 menunjukkan sisa galat terkonsentrasi di segitiga Trust/Proud/Surprise
(83% data). Seorang manusia menonton video-video yang paling sering tertukar
dan merumuskan aturannya:

| Kelas | Aturan |
|---|---|
| Trust | menonjolkan informasi/pengetahuan yang bisa dipercaya — menyampaikan fakta ATAU menunjukkan caranya |
| Proud | menggambarkan pencapaian atau spesifikasi |
| Surprise | produk baru yang lebih bagus dari yang lama; kemegahan |

Diterjemahkan jadi 6 sub-konsep. **Arah pembedanya sebagian besar benar:**

| Konsep | Trust | Proud | Surprise | Arah |
|---|---|---|---|---|
| **kon_mengajari** | **0,918** | 0,679 | 0,420 | benar, monoton, **2,2x** |
| kon_prestasi | 0,098 | **0,186** | 0,036 | benar |
| kon_spesifikasi | 0,175 | **0,314** | 0,272 | benar |
| kon_menjelaskan | **0,230** | 0,154 | 0,142 | benar, lemah |
| kon_produk_baru | 0,142 | 0,122 | 0,139 | salah arah |
| kon_kemegahan | 0,180 | 0,308 | 0,302 | tidak memisah |

`kon_mengajari` adalah pembeda tunggal terkuat yang pernah dibuat proyek ini.
**Tapi pada CV ia ditolak:**

| Konfigurasi (5 seed) | Akurasi | macro-F1 |
|---|---|---|
| 9 konsep §17 (model final) | **41,6% ± 0,8** | 0,149 ± 0,009 |
| + 6 konsep baru | 40,4% ± 0,4 | 0,152 ± 0,009 |
| + 4 konsep (arah benar saja) | 40,5% ± 0,6 | 0,151 ± 0,012 |
| + kon_mengajari saja | 41,0% ± 0,7 | 0,151 ± 0,012 |

Akurasi kalah di **5 dari 5 seed** (−1,1 poin, jatuh di bawah baseline 41,2%),
macro-F1 cuma +0,004 — di dalam pita derau ±0,009.

**Sebabnya terukur: 17 dari 20 katanya sudah ada di kosakata TF-IDF.**

| Konsep | Kata yang SUDAH di TF-IDF |
|---|---|
| kon_mengajari | tips (52 dok), langkah (19), tutorial (15), newbie (3) |
| kon_spesifikasi | fitur (45), varian (22), canggih (13), spesifikasi (6) |
| kon_kemegahan | harga (**149**), mewah (21), miliar (11) |

Fitur konsepnya hanya **mengulang informasi yang model sudah punya**, dalam
bentuk jumlah kasar yang justru membuang pembobotan per-kata TF-IDF.

> **Aturan yang bisa dipakai lagi.** Bandingkan dengan §17 yang BERHASIL:
> `kon_kehilangan` menangkap frasa `"kita tinggal"` yang muncul di **1 video** —
> mustahil dilihat TF-IDF unigram, dan §17 mencatat penambangan statistik pun
> tidak menemukannya. **Fitur konsep hanya menolong kalau menangkap yang TF-IDF
> tidak bisa: FRASA LANGKA, bukan kata umum.** Kata seperti `tips`, `varian`,
> `mewah` sudah tertangkap; merumuskan konsep di atasnya adalah pekerjaan ganda.

Kode disimpan sebagai `KONSEP_UJI` di `features_text.py` — dihitung dan masuk
dataset untuk analisis, tapi `kolom_konsep()` hanya mengembalikan 9 konsep §17.

### B. Ketimpangan kelas: SMOTE mustahil, dan semua tuas itu tuas yang sama

§6 menandai "coba SMOTE tapi hati-hati" dan tidak pernah mengeksekusinya.
Sekarang dieksekusi, dan jawabannya lebih tegas daripada "hati-hati":

| Metode (3 seed) | Akurasi | macro-F1 |
|---|---|---|
| **class_weight bertingkat + tau (dipakai)** | **41,4%** | 0,147 |
| RandomOverSampler | 38,7% | 0,166 |
| **SMOTE k=1 dan k=3** | — | **GAGAL TOTAL** |
| class_weight balanced penuh (alpha=1) | 36,2% | 0,179 |

**SMOTE tidak bisa dijalankan sama sekali.** Ia menginterpolasi antar tetangga,
jadi butuh minimal 2 sampel per kelas; `Love` dan `Loyalty` punya tepat 1.
Bukan "hati-hati" — **mustahil secara struktural**, dan itu jawaban final untuk
§6.

Yang lebih penting: **ketiga tuas ketimpangan bergerak di kurva tukar-guling
yang sama.** `alpha` (§14), `tau` (§20), dan oversampling semuanya membeli
macro-F1 dengan akurasi pada laju yang praktis identik — bandingkan
RandomOverSampler (38,7% / 0,166) dengan titik `tau≈0,4` (38,2% / 0,162).
Mereka **tuas yang berulang, bukan perbaikan yang bertumpuk.** Jadi tidak ada
yang "belum dicoba" di sisi ketimpangan; yang ada hanya pilihan titik di kurva,
dan kurvanya sudah dipetakan lengkap di §20.

### Kesimpulan gabungan

Tiga percobaan besar hari ini (§18 IndoBERT, §19 emosi, §21 stemming)
menghasilkan +0,8 poin akurasi dan +0,025 macro-F1. Ronde ini menghasilkan nol.
Pola penurunan hasilnya jelas, dan sebabnya sudah terdiagnosis di §22: sisa
galat terkunci di derau anotator (plafon ~56%), tiga kelas bersampel 1–8, dan
12,5% test tanpa sinyal. **Sisi model dinyatakan selesai.**

Berkas: `KONSEP_UJI` di `src/features_text.py`.

---

## 24. Terobosan — Ensemble LogReg + ComplementNB Menggeser Kurvanya

§23 menyimpulkan sisi model selesai. Kesimpulan itu **salah**, dan yang
membongkarnya adalah pengamatan atas pola kegagalannya sendiri.

### Pengamatan yang membuka jalan

Ditumpuk berdampingan, seluruh percobaan yang pernah dilakukan ternyata jatuh
di **satu kurva tukar-guling yang sama**:

| Perubahan | Akurasi | macro-F1 |
|---|---|---|
| alpha 0,5 → 1,0 (§14) | 42,0% → 39,7% | 0,099 → 0,155 |
| tau 0,0 → 0,5 (§20) | 42,0% → 37,1% | 0,129 → 0,180 |
| RandomOverSampler (§23) | 41,4% → 38,7% | 0,147 → 0,166 |
| C 3 → 10 | 41,4% → 40,3% | 0,147 → 0,155 |
| min_df 2 → 3 | 41,4% → 41,0% | 0,147 → 0,151 |
| ComplementNB sendirian | 41,4% → 35,2% | 0,147 → 0,173 |

**Enam tuas yang tampak berbeda, laju pertukaran yang praktis sama.** Itu bukan
kebetulan — itu tanda bahwa yang membatasi bukan *letak titik operasi*,
melainkan *kualitas estimasi probabilitasnya*. Menggeser titik di sepanjang
kurva tidak akan pernah menaikkan kedua metrik; yang dibutuhkan adalah
menggeser **kurvanya**.

Yang menggeser kurva hanya tiga hal: fitur yang lebih baik (sudah dikuras
habis, §16–§23), data lebih banyak (terkunci, §22), atau **estimasi
probabilitas yang lebih baik**. Yang terakhir belum pernah dicoba.

### Solusinya: dua keluarga model yang salah di tempat berbeda

| | Logistic Regression | ComplementNB |
|---|---|---|
| Jenis | diskriminatif | generatif |
| Rancangan | umum | **khusus teks tidak seimbang** |
| Cara kerja | batas keputusan | distribusi kata KOMPLEMEN tiap kelas |
| Sendirian | 41,4% / 0,147 | 35,2% / 0,173 |

ComplementNB memodelkan kata yang khas untuk **bukan-kelas-itu**, sehingga
kelas kecil tidak tenggelam oleh kelas besar — persis penyakit dataset ini.
Sendirian ia kalah telak pada akurasi, tapi **salahnya di tempat yang berbeda**
dari LogReg. Merata-ratakan probabilitasnya menurunkan ragam estimasi.

### Hasil: satu-satunya perubahan yang memperbaiki KEDUA metrik

| w(NB) — 5 seed | Akurasi | macro-F1 | Menang vs sekarang |
|---|---|---|---|
| 0,0 (LogReg saja) | 41,6% ± 0,8 | 0,149 ± 0,009 | — |
| 0,3 | 41,9% ± 1,0 | 0,157 ± 0,015 | acc 3/5, F1 4/5 |
| 0,4 | 42,1% ± 1,1 | 0,163 ± 0,017 | acc 4/5, F1 4/5 |
| **0,5 (dipilih)** | **42,3% ± 1,0** | **0,167 ± 0,012** | **acc 4/5, F1 5/5** |
| 0,6 | 42,3% ± 1,2 | 0,170 ± 0,012 | acc 4/5, F1 5/5 |
| 0,7 | 42,0% ± 0,8 | 0,170 ± 0,010 | acc 3/5, F1 5/5 |

Selisih per seed pada w=0,5: macro-F1 **+0,013, +0,019, +0,031, +0,027, +0,018**
— positif di semuanya. **+0,7 poin akurasi DAN +0,018 macro-F1.**

Dipilih **w = 0,5**: bobot sama rata, setara dengan 0,6 di dalam derau, dan
tidak berkesan ditala berlebihan pada 5 seed.

### Yang juga diuji di ronde ini dan TIDAK berhasil

| Percobaan | Hasil |
|---|---|
| IDF transduktif (kosakata dari train+test) | **+0,000 / +0,000** — persis nol |
| LinearSVC + kalibrasi | gagal — butuh ≥3 sampel per kelas untuk CV internal |
| SGD modified_huber | 35,0% / 0,129 — kalah di kedua metrik |
| Sapuan C dan min_df | hanya bergerak di kurva yang sama |

### Tangga ablation final

| Tahap | Akurasi | macro-F1 | weighted-F1 |
|---|---|---|---|
| 1. baseline (kelas mayoritas) | 41,2% | 0,058 | 0,241 |
| 9. + fitur konsep (§17) | 42,3% | 0,135 | 0,354 |
| 10. + lapisan emosi (§19) | 42,7% | 0,135 | 0,364 |
| 11. + stemming, unigram (§21) | 43,1% | 0,148 | 0,376 |
| **12. + ensemble LogReg+CNB (FINAL)** | **43,2%** | **0,153** | **0,374** |

### Model final

```
stemming Sastrawi -> TF-IDF unigram (caption + transkrip)
  + 9 fitur konsep semantik + 14 probabilitas emosi
  -> ensemble 0,5 x LogisticRegression + 0,5 x ComplementNB
  -> koreksi prior tau=0,1  ->  fallback prior  ->  penimpaan duplikat URL
```

| | Akurasi | macro-F1 |
|---|---|---|
| baseline tebak `Surprise` | 41,2% | 0,058 |
| model final §17 (awal hari ini) | 42,3% | 0,135 |
| **model final sekarang** | **43,2%** | **0,153** |

Terhadap baseline: **+2,0 poin akurasi dan macro-F1 2,6x.**

### Pelajaran

**Ketika banyak perubahan berbeda menghasilkan pertukaran yang sama, itu
informasi — bukan kebuntuan.** Enam tuas yang jatuh di satu kurva adalah bukti
bahwa kendalanya ada di tempat lain, dan justru pengamatan itu yang menunjukkan
ke mana harus mencari. Kalau kesimpulan §23 ("sisi model selesai") diterima
begitu saja, ensemble ini tidak akan pernah ditemukan.

Catatan kejujuran: §23 menyatakan sisi model selesai, dan §24 membatalkannya.
Yang salah bukan datanya, melainkan **kesimpulan yang ditarik dari pola
kegagalan tanpa memeriksa apa yang pola itu sebenarnya tunjukkan.**

Berkas: `EnsembleLRNB` di `src/train.py`, tahap 12 di tangga ablation.

---

## 25. Brainstorm Ulang dari Nol — Dua Ide Baru, Dua Hasil Negatif, Satu Pola

Setelah §24, seluruh pendekatan dilupakan dan masalah dipikirkan ulang dari
awal. Sembilan arah dirumuskan; dua yang paling menjanjikan dieksekusi.

### Ide 9: pencocokan distribusi prediksi — GAGAL

Model memprediksi `Surprise` untuk 60% baris test, padahal proporsinya di train
41,2%. Kalau test dari populasi yang sama, distribusi prediksi kita salah
sistematis. Dicari pengali per-kelas (Sinkhorn satu sisi) agar jumlah
prediksinya cocok dengan proporsi harapan — generalisasi dari `tau` yang cuma
punya satu parameter global.

| Metode (5 seed) | Akurasi | macro-F1 |
|---|---|---|
| **tau = 0,1 (dipakai)** | **42,3% ± 1,0** | **0,167 ± 0,012** |
| pencocokan distribusi | 41,5% ± 0,7 | 0,133 ± 0,007 |

macro-F1 kalah di **0 dari 5 seed**. Dan diagnostiknya menjelaskan sebabnya:
algoritmanya bahkan **tidak berhasil mencocokkan distribusinya** (472 `Surprise`
lawan target 331). Argmax itu tidak kontinu — banyak baris punya probabilitas
`Surprise` yang begitu dominan sehingga tidak ada pengali wajar yang membaliknya.

**Pelajaran: memaksa distribusi prediksi memenuhi kuota prior itu merusak.**
Keyakinan model membawa informasi; memaksa baris `Surprise` berkeyakinan rendah
menjadi `Sad` demi kuota menghancurkan presisi di kedua sisi. `tau` menang
justru karena memakai prior secara **lunak**, bukan sebagai kuota keras.

### Ide 4: fitur visual CLIP — sinyalnya NYATA, sumbangannya NOL

842 video (12 GB) tidak pernah dipakai. §14 membuang fitur visual, tapi yang
dibuang itu fitur numerik buatan tangan (cut rate, saturasi, histogram). CLIP
berbeda — ia semantik.

**Keputusan desain:** bukan embedding mentah 512 dimensi (pasti overfit pada 803
baris, persis peringatan §5B), melainkan **18 skor kemiripan zero-shot** terhadap
prompt yang dirancang dari aturan pembeda manusia (§23), di-maks-pool dan
rata-pool antar 8 keyframe → 36 fitur yang semuanya bisa dijelaskan.

Alasan optimisme awal: aturan yang sama gagal di teks karena 17 dari 20 katanya
sudah ada di TF-IDF. Di ranah visual redundansi itu **seharusnya** tidak ada.

**Sinyalnya memang nyata.** 9 dari 18 prompt signifikan (ANOVA p < 0,05), dan
dua arahnya persis benar: `pameran` dan `mobil_mewah` tertinggi di `Surprise` —
aturan Surprise yang gagal di teks, berhasil di visual.

| Konfigurasi (5 seed) | Akurasi | macro-F1 | Menang |
|---|---|---|---|
| **visual SENDIRIAN, tanpa teks** | 34,1% ± 0,3 | **0,132 ± 0,005** | — |
| model final §24 | 42,3% ± 1,0 | 0,167 ± 0,012 | acuan |
| + visual, bobot 0,05 | 42,3% ± 1,0 | 0,167 ± 0,013 | **+0,000** |
| + visual, bobot 0,10 | 42,1% ± 1,0 | 0,164 ± 0,013 | acc 1/5 |
| + visual, bobot 0,25 | 41,9% ± 0,9 | 0,160 ± 0,016 | acc **0/5** |

Visual sendirian mencapai macro-F1 **0,132 — 2,3x baseline (0,058)** tanpa
melihat satu kata pun. Tapi menambahkannya ke model memberi **nol persis**, dan
bobot lebih besar justru merugikan.

> **Dua jebakan teknis yang perlu dicatat.**
>
> 1. **Skala CLIP menipu.** Skor kemiripan berkerumun di pita 0,18–0,26 dengan
>    rerata ~0,21. Yang membawa informasi adalah SIMPANGANNYA, bukan levelnya.
>    Tanpa penskalaan ulang, mengalikan bobot 0,25 hanya menghasilkan konstanta
>    ~0,05 dan classifier tidak melihat apa-apa. Dipakai `MinMaxScaler` — bukan
>    `StandardScaler`, karena ComplementNB dalam ensemble (§24) menolak nilai
>    negatif yang pasti dihasilkan pembakuan-z.
> 2. **`pip install opencv-python-headless` menaikkan numpy ke 2.x** dan
>    memutus ABI pandas (`numpy.dtype size changed`), walaupun numpy sudah dipin
>    di `requirements.txt`. Seluruh pipeline sempat mati. Perbaikan:
>    `pip install opencv-python-headless "numpy<2"`. Sudah dicatat di
>    `requirements.txt`.

### Pola yang menyatukan tiga hasil negatif

Ini kali ketiga sumber sinyal baru ternyata **redundan dengan teks yang sudah
ada**:

| Sumber baru | Sinyal sendirian | Sumbangan ke model |
|---|---|---|
| Caption vs transkrip (§18) | keduanya ~41% | sepakat 80,1% baris; oracle cuma +4,4 poin |
| Konsep dari aturan manusia (§23) | arah benar, `kon_mengajari` 2,2x | −1,1 poin akurasi (17/20 kata sudah di TF-IDF) |
| Visual CLIP (§25) | macro-F1 2,3x baseline | **+0,000** |

**Videonya, transkripnya, dan captionnya menceritakan hal yang sama.** Video
gym punya kata `dumbbell` di transkrip DAN tampak seperti gym; mobil di pameran
punya kata `harga` DAN tampak seperti panggung pameran. Tiap modalitas membawa
sinyal nyata, tapi **sinyal yang sama**.

Itu menjelaskan kenapa seluruh rekayasa fitur sejak §16 menghasilkan pergeseran
±0,01, sementara satu perubahan yang tidak menambah informasi sama sekali —
ensemble dua keluarga model (§24) — memberi +0,7 poin dan +0,018. **Pada dataset
ini, batasnya bukan berapa banyak informasi yang kita punya, melainkan berapa
banyak informasi yang BERBEDA.**

### Ide yang dirumuskan tapi belum diuji

| # | Ide | Kenapa belum |
|---|---|---|
| 1 | Struktur anotator dari urutan `id` (chi2 p = 8,6e−13) | p-value ekstrem, tapi test dinomori terpisah 1–200 — kalau strukturnya tidak berpindah, CV bagus dan test hancur |
| 2 | Koreksi loss dengan matriks transisi derau dari 8 video berlabel ganda | butuh implementasi khusus; waktu habis |
| 3 | Geometri roda Plutchik (label smoothing menurut ketetanggaan) | idem |
| 5 | OCR teks tempel di layar | satu-satunya sumber yang benar-benar belum ada di mana pun — tapi §25 menunjukkan modalitas baru cenderung redundan |
| 6 | wav2vec2 speech emotion recognition Bahasa Indonesia | idem |
| 7 | Metric learning / k-NN untuk kelas bersampel 1 | idem |
| 8 | Zero-shot NLI | idem |

Nomor 1 paling menggoda dan paling berisiko; nomor 5 paling berpotensi memberi
sinyal yang benar-benar berbeda, tapi pola §25 memperkirakan ia pun redundan.

**Model final tidak berubah dari §24: 43,2% / 0,153 / weighted-F1 0,374.**

Berkas: `outputs/ablation_visual.csv`, `data/features/visual.parquet`,
kode: `src/features_visual.py`, `src/eval_visual.py`, `src/cocok_distribusi.py`.
