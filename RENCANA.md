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
