# Rencana Penelitian — Klasifikasi Emosi pada Video

## 1. Gambaran Umum

Penelitian ini bertujuan untuk melakukan klasifikasi kategori emosi pada video.

Untuk tahap awal, fokus utama berada pada **informasi tekstual** yang terdapat pada video, seperti caption, subtitle, script, atau hasil transkripsi dari audio.

Pendekatan awal:

```text
Video
  ↓
Caption / Subtitle / Script
  ↓
Text Preprocessing
  ↓
Analisis Keyword & Linguistic Pattern
  ↓
Ekstraksi Textual Features
  ↓
Klasifikasi Emosi
```

Pendekatan audio dan visual belum menjadi fokus utama pada tahap awal dan akan didiskusikan lebih lanjut dengan dosen pembimbing.

---

## 2. Rencana Dataset

Dataset terdiri dari video yang telah memiliki kategori/label emosi.

Contoh:

```text
Video 1 → Senang
Video 2 → Sedih
Video 3 → Marah
Video 4 → Takut
...
```

Untuk setiap video, informasi yang akan dikumpulkan antara lain:

* Video ID
* Label/kategori emosi
* Caption/subtitle jika tersedia
* Script/transkrip
* Metadata lain yang dianggap relevan

Jika video tidak memiliki script atau subtitle, audio dapat digunakan untuk menghasilkan transkripsi menggunakan **Speech-to-Text**.

---

## 3. Ekstraksi Teks

Informasi tekstual dari video akan dikumpulkan berdasarkan prioritas:

1. Caption/subtitle yang tersedia.
2. Script/transkrip yang tersedia.
3. Speech-to-Text dari audio apabila script/subtitle tidak tersedia.

Contoh:

```text
Video
↓
Audio
↓
Speech-to-Text
↓
Transcript
```

Tujuan tahap ini adalah mendapatkan representasi teks yang dapat dianalisis untuk setiap kategori emosi.

---

## 4. Text Preprocessing

Teks yang telah dikumpulkan akan melalui preprocessing sebelum dilakukan analisis.

Tahapan yang kemungkinan digunakan:

* Lowercasing
* Cleaning noise
* Tokenization
* Normalisasi kata
* Stopword removal
* Stemming/Lemmatization jika diperlukan

Contoh:

```text
"AKU BENAR-BENAR KECEWA!!!"

        ↓ preprocessing

"aku benar benar kecewa"
```

Tahapan preprocessing akan disesuaikan dengan karakteristik dataset.

---

## 5. Eksplorasi Pattern pada Setiap Kategori

Setelah teks dikelompokkan berdasarkan kategori emosi, akan dilakukan eksplorasi untuk mengetahui karakteristik bahasa pada masing-masing kategori.

Contoh:

```text
Kategori: MARAH

Kata yang sering muncul:
- kesal
- benci
- marah
- kenapa
- tidak
- bodoh
```

Hal yang dapat dianalisis:

### Keyword

Mencari kata-kata yang sering muncul pada masing-masing kategori.

### Word Frequency

Menghitung frekuensi kemunculan kata pada setiap kategori.

### TF-IDF

Mengidentifikasi kata yang relatif lebih khas pada suatu kategori dibandingkan kategori lainnya.

### N-Gram

Menganalisis kombinasi kata yang sering muncul.

Contoh:

```text
"tidak suka"
"kenapa harus"
"aku benci"
```

### Linguistic Pattern

Melihat pola penggunaan bahasa yang mungkin berkaitan dengan kategori emosi.

Contohnya:

* kata tertentu
* kombinasi kata
* intensitas penggunaan kata
* tanda seru
* panjang teks
* pola kalimat

Pattern tersebut tidak langsung dianggap sebagai aturan klasifikasi, tetapi digunakan untuk memahami karakteristik dataset.

---

## 6. Visualisasi Eksplorasi

Hasil eksplorasi dapat divisualisasikan untuk membantu memahami perbedaan antar kategori.

Contoh visualisasi:

* Word Cloud
* Top-N Word Frequency
* TF-IDF Feature Ranking
* N-Gram Frequency
* Distribusi panjang teks
* Perbandingan kata antar kategori

Contoh konsep:

```text
             TEXT DATA
                 ↓
        ┌────────┴────────┐
        ↓                 ↓
     SENANG             SEDIH
        ↓                 ↓
   Keyword A          Keyword X
   Keyword B          Keyword Y
   N-Gram A           N-Gram B
        ↓                 ↓
        └────────┬────────┘
                 ↓
        Pattern Analysis
```

---

## 7. Ekstraksi Fitur

Setelah karakteristik teks dipahami, fitur tekstual dapat digunakan sebagai input model klasifikasi.

Tahap awal dapat menggunakan:

```text
Text
 ↓
TF-IDF
 ↓
Feature Vector
 ↓
Machine Learning Classifier
```

Beberapa algoritma yang dapat digunakan sebagai baseline:

* Logistic Regression
* SVM
* Random Forest
* Naive Bayes

Metode terbaik belum ditentukan dan akan dipilih berdasarkan hasil eksperimen.

---

## 8. Pengembangan NLP

Setelah baseline berhasil, pendekatan NLP yang lebih kompleks dapat dibandingkan.

Salah satu kandidat:

```text
Text
 ↓
IndoBERT / Transformer
 ↓
Text Embedding / Fine-tuning
 ↓
Emotion Classification
```

Tujuannya adalah membandingkan apakah representasi berbasis transformer memberikan performa lebih baik dibandingkan pendekatan TF-IDF + machine learning.

---

## 9. Audio sebagai Pengembangan

Audio belum menjadi fokus utama tahap awal.

Namun terdapat dua kemungkinan pemanfaatan audio:

### A. Audio sebagai sumber teks

```text
Audio
 ↓
Speech-to-Text
 ↓
Transcript
 ↓
Text Classification
```

Pendekatan ini masih termasuk dalam pendekatan tekstual.

### B. Audio sebagai fitur emosi

Audio dapat dianalisis secara langsung berdasarkan karakteristik suara, misalnya:

* Pitch
* Energy
* Tempo
* Intonation
* Prosody

Pendekatan ini lebih dekat dengan **Speech Emotion Recognition**.

Contoh:

```text
Audio
 ↓
Audio Feature Extraction
 ↓
Prosodic Features
 ↓
Emotion Classification
```

Penggunaan pendekatan ini akan dipertimbangkan setelah berdiskusi dengan dosen pembimbing.

---

## 10. Visual sebagai Pengembangan

Informasi visual dari video untuk sementara belum menjadi fokus utama.

Kemungkinan pengembangan:

```text
Video
 ↓
Frame Extraction
 ↓
Visual Feature Extraction
 ↓
Emotion Classification
```

Contohnya dapat mencakup ekspresi wajah, gesture, atau informasi visual lainnya.

Namun multimodal/visual analysis belum menjadi bagian dari rencana utama saat ini.

---

## 11. Rencana Eksperimen Awal

Tahapan eksperimen yang direncanakan:

```text
1. Kumpulkan dataset video
        ↓
2. Identifikasi label/kategori emosi
        ↓
3. Ekstrak caption/script/transcript
        ↓
4. Preprocessing teks
        ↓
5. Eksplorasi keyword
        ↓
6. Analisis word frequency
        ↓
7. Analisis TF-IDF
        ↓
8. Analisis N-Gram
        ↓
9. Identifikasi linguistic pattern
        ↓
10. Feature extraction
        ↓
11. Baseline classification
        ↓
12. Evaluasi model
        ↓
13. Eksperimen IndoBERT/Transformer
        ↓
14. Bandingkan hasil
```

---

## 12. Evaluasi Model

Model akan dievaluasi menggunakan metrik klasifikasi yang sesuai, misalnya:

* Accuracy
* Precision
* Recall
* F1-Score
* Confusion Matrix

Jika dataset tidak seimbang antar kategori, evaluasi tidak hanya menggunakan accuracy. F1-score dan metrik per kelas akan menjadi perhatian utama.

---

## 13. Pertanyaan yang Perlu Didiskusikan dengan Dosen

Beberapa keputusan masih perlu dikonfirmasi:

* Apakah fokus penelitian cukup menggunakan textual modality?
* Apakah audio perlu digunakan sebagai fitur tambahan?
* Apakah intonasi/prosody perlu dianalisis?
* Apakah visual perlu dimasukkan?
* Apakah penelitian diarahkan menjadi multimodal emotion classification?
* Apakah caption/script dapat dianggap sebagai representasi utama emosi video?
* Model NLP yang digunakan: TF-IDF + ML atau Transformer/IndoBERT?
* Apakah keyword/pattern analysis digunakan hanya sebagai exploratory analysis atau menjadi bagian dari model?
* Bagaimana definisi dan jumlah kategori emosi yang digunakan?

---

## 14. Status Rencana Saat Ini

**Fokus utama:**

> Text-based emotion classification pada video.

**Tahap yang sedang direncanakan:**

> Ekstraksi script/caption → preprocessing → keyword analysis → pattern analysis → feature extraction → classification.

**Audio:**

> Dipertimbangkan, terutama Speech-to-Text dan kemungkinan analisis intonasi/prosody.

**Visual:**

> Belum menjadi prioritas dan akan dipertimbangkan setelah diskusi dengan dosen.

**Model:**

> Belum dikunci. Baseline TF-IDF + Machine Learning akan dipertimbangkan sebelum menggunakan model NLP berbasis Transformer.

---

## 15. Prinsip Pengembangan

Penelitian tidak langsung dimulai dari model yang kompleks.

Urutan yang digunakan:

```text
Understand the Data
        ↓
Understand the Pattern
        ↓
Build a Baseline
        ↓
Evaluate
        ↓
Improve the Model
        ↓
Consider Additional Modalities
```

Dengan demikian, alasan pemilihan metode dapat dijelaskan berdasarkan karakteristik dataset dan hasil eksperimen, bukan hanya berdasarkan penggunaan model yang lebih kompleks.
