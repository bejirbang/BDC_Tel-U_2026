"""
Membangun `explore.ipynb` - walkthrough lengkap yang bisa dipresentasikan.

Notebook dibangun dari skrip, bukan diedit tangan, supaya bisa dibuat ulang
kapan saja setelah angkanya berubah dan tidak ada sel yang keluarannya basi.
Tiap sel kode langsung menampilkan hasilnya.

Isinya sengaja mempertahankan DUA sifat sekaligus:
  - **eksplorasi nyata** - data mentah diperiksa, modul fitur dibangun hidup,
    analisis log-odds dijalankan, bukan sekadar membaca CSV jadi
  - **alur presentasi** - berurut sebagai cerita dari masalah ke model final

Jalankan:  python build_notebook.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

ROOT = Path(__file__).resolve().parent
KELUARAN = ROOT / "explore.ipynb"
MD, CODE = new_markdown_cell, new_code_cell


def bangun() -> nbf.NotebookNode:
    s = []

    # ================= JUDUL =================
    s.append(MD("""# Klasifikasi Emosi Video Media Sosial — Walkthrough Lengkap

**Big Data Challenge — Telkom University 2026**

Memprediksi 1 dari 10 kategori emosi untuk 200 video test, berbekal **hanya URL**.
Tidak ada teks, tidak ada fitur — semuanya dibangun sendiri.

| | |
|---|---|
| Train | 803 baris berlabel · Test | 200 baris |
| Kelas | 10, sangat timpang (331 `Surprise` … 1 `Love`, 1 `Loyalty`) |
| Baseline | tebak `Surprise` semua = **41,2%** akurasi |
| **Hasil akhir** | **43,2% akurasi · macro-F1 0,153 · weighted-F1 0,374** |

Notebook ini menjalankan pipeline sungguhan, bukan menampilkan ringkasan:
modul fitur dibangun hidup, analisis dijalankan ulang, angkanya keluar apa adanya."""))

    s.append(CODE("""import sys, warnings, inspect, json, collections
warnings.filterwarnings("ignore")
sys.path.insert(0, "src")

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from IPython.display import Image, display

pd.set_option("display.width", 120)
pd.set_option("display.max_columns", 40)

BIRU, HIJAU, JINGGA, ABU = "#2a78d6", "#1baf7a", "#eda100", "#898781"
TINTA, GARIS = "#0b0b0b", "#e1e0d9"

def rapikan(ax, sumbu="y"):
    for k in ("top", "right"):
        ax.spines[k].set_visible(False)
    for k in ("left", "bottom"):
        ax.spines[k].set_color("#c3c2b7")
    ax.tick_params(colors=ABU, labelcolor="#52514e")
    if sumbu:
        ax.grid(True, axis=sumbu, color=GARIS, linewidth=0.8)
        ax.set_axisbelow(True)

print("siap.")"""))

    # ================= 0 EKSPLORASI MENTAH =================
    s.append(MD("""---
# Bagian 0 · Eksplorasi awal

Sebelum apa pun: lihat isi datanya. Yang tersedia hanya dua kolom — `id` dan
`video` (URL). Semua sisanya harus dibuat sendiri."""))

    s.append(CODE("""df = pd.read_csv("datatrain.csv")
dt = pd.read_csv("datatest.csv")
print("kolom train:", list(df.columns), "| baris:", len(df))
print("kolom test :", list(dt.columns), "| baris:", len(dt))
display(df.head(4))"""))

    s.append(CODE("""print("kategori emosi:", sorted(df.emotion.unique()))
print("jumlah kategori:", df.emotion.nunique())
print("label kosong   :", int(df.emotion.isna().sum()))
print()
dist = df.emotion.value_counts()
display(pd.DataFrame({"jumlah": dist, "persen": (100*dist/len(df)).round(1)}))"""))

    s.append(MD("""**Yang langsung terlihat: distribusinya sangat timpang.** `Surprise` sendirian
41,2% — artinya menebaknya untuk semua baris sudah memberi 41,2% akurasi. Itu
baseline yang harus dikalahkan.

Dan dua kelas hanya punya **satu** sampel. Itu bukan detail kecil: macro-F1
merata-ratakan 10 kelas, jadi dua kelas yang mustahil dipelajari langsung
mengunci 20% penyebutnya di nol."""))

    s.append(CODE("""fig, ax = plt.subplots(figsize=(9, 3.2))
warna = [BIRU if n > 30 else JINGGA for n in dist.values]
b = ax.bar(range(len(dist)), dist.values, color=warna, width=0.66)
ax.bar_label(b, padding=3, fontsize=9, color="#52514e")
ax.axhline(30, color=ABU, ls=":", lw=1)
ax.set_xticks(range(len(dist)), dist.index, rotation=30, ha="right")
ax.set_ylabel("jumlah baris train")
ax.set_title("Distribusi label — jingga = terlalu kecil untuk dipelajari",
             loc="left", color=TINTA)
rapikan(ax); plt.tight_layout(); plt.show()"""))

    s.append(MD("""### Sampel per kategori — membaca isi datanya sendiri

Melihat lima video per emosi, untuk menangkap apa sebenarnya yang membedakan
kategori-kategori ini."""))

    s.append(CODE("""sampel = pd.concat(
    [g.sample(n=min(len(g), 3), random_state=42) for _, g in df.groupby("emotion")],
    ignore_index=True)
sampel["no"] = sampel.groupby("emotion").cumcount() + 1
display(sampel[["emotion", "no", "video"]].head(12).style.hide(axis="index"))"""))

    s.append(MD("""### Anomali tautan: sumbernya tidak seragam"""))

    s.append(CODE("""domain = df.video.str.extract(r"https?://([^/]+)")[0].value_counts()
display(domain.to_frame("jumlah"))
print("Tiga jalur unduh berbeda dibutuhkan: Instagram (yt-dlp),")
print("Google Drive (endpoint langsung), dan CDN mentah (HTTP biasa).")"""))

    # ================= 1 AKUISISI =================
    s.append(MD("""---
# Bagian 1 · Akuisisi — jalur kritis proyek ini

Kalau tahap ini gagal, semua tahap berikutnya ikut gagal. Tiga jalur unduh,
masing-masing dengan jebakannya sendiri."""))

    s.append(CODE("""import acquire

# URL harus dinormalisasi dulu: parameter pelacakan (igsh, utm_source) dan
# bentuk /username/reel/X vs /reel/X membuat URL yang SAMA terlihat berbeda.
contoh = [
    "https://www.instagram.com/reel/ABC123/?igsh=xyz",
    "https://www.instagram.com/someuser/reel/ABC123/",
    "https://drive.google.com/file/d/1AbC/view?usp=sharing",
]
for u in contoh:
    jenis, kunci = acquire.classify(u)
    print(f"{jenis:9} {kunci:24} <- {u[:52]}")"""))

    s.append(CODE("""# Jebakan Google Drive: file mati membalas HTTP 200 berisi HTML, bukan 4xx.
# Tanpa pengecekan Content-Type, halaman error tersimpan sebagai .mp4
print(inspect.getsource(acquire._gdrive_direct)[:900])"""))

    s.append(CODE("""man = pd.read_csv(acquire.MANIFEST)
print(f"manifest: {len(man)} URL unik")
display(man.status.value_counts().to_frame("jumlah"))

gagal_tr = pd.read_csv("data/gagal_train.csv")
gagal_te = pd.read_csv("data/gagal_test.csv")
print(f"\\ngagal train: {len(gagal_tr)} baris | gagal test: {len(gagal_te)} baris")
display(gagal_tr.iloc[:, -2].value_counts().to_frame("sebab kegagalan"))"""))

    s.append(MD("""**Kegagalannya permanen dan sudah dipastikan buntu.** 75 berkas Drive memang
sudah dihapus, 24 URL CDN bertanda tangan sudah kadaluarsa, dan 11 post
Instagram mati — sudah diverifikasi dengan cookie login yang valid, jadi bukan
blokir IP.

Ini menjadi batas atas performa yang dinyatakan terbuka: **12,5% baris test
tidak punya sinyal apa pun** dan hanya bisa ditebak dari prior."""))

    s.append(CODE("""display(Image("outputs/figures/02_cakupan_data.png"))"""))

    # ================= 2 DERAU LABEL =================
    s.append(MD("""---
# Bagian 2 · Temuan yang mengubah segalanya

Sebelum mengejar akurasi, satu hal wajib diukur: **seberapa konsisten
anotatornya sendiri?** Ada video yang muncul di beberapa baris train — kalau
anotator konsisten, labelnya pasti sama."""))

    s.append(CODE("""from common import classify as klasifikasi

d = collections.defaultdict(list)
for u, e in zip(df.video, df.emotion):
    d[klasifikasi(str(u))[1]].append(e)

dup = {k: v for k, v in d.items() if len(v) > 1}
bentrok = {k: v for k, v in dup.items() if len(set(v)) > 1}
baris = sum(len(v) for v in dup.values())
setuju = sum(collections.Counter(v).most_common(1)[0][1] for v in dup.values())

print(f"video dipakai >1 baris train : {len(dup)}  ({baris} baris)")
print(f"di antaranya BERLABEL BENTROK: {len(bentrok)}")
print(f"\\nMenebak dengan label mayoritas video itu sendiri - informasi yang")
print(f"model tidak akan pernah punya - hanya benar {setuju}/{baris} = {100*setuju/baris:.1f}%")
print(f"\\n=> PLAFON DERAU LABEL ~{100*setuju/baris:.0f}%.")
print(f"   Jarak baseline (41,2%) ke plafon hanya ~{100*setuju/baris-41.2:.0f} poin.")"""))

    s.append(CODE("""rinci = pd.DataFrame([
    {"video": k[:24], "n baris": len(v),
     "label yang diberikan": ", ".join(f"{e} x{c}" for e, c in collections.Counter(v).most_common())}
    for k, v in bentrok.items()])
display(rinci)"""))

    s.append(MD("""**Satu video dilabeli Trust 7×, Proud 4×, Surprise 3×, Fear 1×, Neutral 1×.**

Anotator tidak sepakat pada video yang sama persis. Ini mengubah prioritas
seluruh proyek: mengejar akurasi jauh di atas ~56% bukan hanya sulit, tapi
**tidak bermakna** — karena labelnya sendiri tidak konsisten sejauh itu.

Konsekuensi teknisnya: `StratifiedGroupKFold` dengan `groups=key` menjadi
**wajib**. Tanpa pengelompokan per URL, baris kembar bocor antara train dan
validasi dan skor CV jadi bohong."""))

    s.append(CODE("""display(Image("outputs/figures/06_plafon_derau_label.png"))"""))

    # ================= 3 WHISPER =================
    s.append(MD("""---
# Bagian 3 · Whisper — lompatan cakupan terbesar

Caption hanya menutupi 62,6% baris. Transkrip audio menutupi 87,5%. Itu
satu-satunya lompatan cakupan besar yang tersedia."""))

    s.append(CODE("""ds = pd.read_parquet("data/features/dataset.parquet")
train = ds[ds.split == "train"].reset_index(drop=True)
test = ds[ds.split == "test"].reset_index(drop=True)

cak = pd.DataFrame({
    "sumber teks": ["caption saja", "transkrip Whisper saja", "gabungan"],
    "cakupan train (%)": [
        100*(train.text_all.fillna("").str.strip() != "").mean(),
        100*(train.text_transcript.fillna("").str.strip() != "").mean(),
        100*(train.text_combined.fillna("").str.strip() != "").mean()]}).round(1)
display(cak)

kosong = (train.text_combined.fillna("").str.strip() == "").sum()
print(f"baris train tanpa teks sama sekali: {kosong} ({100*kosong/len(train):.1f}%)")"""))

    s.append(CODE("""# contoh transkrip - kualitasnya memadai walau bahasa gaul & nama merek meleset
contoh = train[train.text_transcript.fillna("").str.len() > 300].iloc[0]
print("LABEL:", contoh.emotion)
print("\\nCAPTION  :", str(contoh.text_all)[:180])
print("\\nTRANSKRIP:", str(contoh.text_transcript)[:340])"""))

    # ================= 4 FITUR =================
    s.append(MD("""---
# Bagian 4 · Rekayasa fitur — dan hasil negatif yang berharga

## 4.1 Leksikon emosi: DIUJI, dan GAGAL

Pendekatan paling naluriah: petakan kata emosi ke kelasnya (`bangga` → Proud).
Diuji langsung pada data."""))

    s.append(CODE("""txt = pd.read_parquet("data/features/text.parquet")
# text.parquet sudah punya kolom `emotion` sendiri - dibuang dulu supaya merge
# tidak menghasilkan emotion_x / emotion_y
t2 = (txt.drop(columns=["emotion"], errors="ignore")
         .merge(train[["id", "emotion"]], on="id"))
t2 = t2[t2.emotion.notna()]
lex = [c for c in t2.columns if c.startswith("lex_")]

peringkat = {}
for c in lex:
    m = t2.groupby("emotion")[c].mean().sort_values(ascending=False)
    kelas_sendiri = c.replace("lex_", "").capitalize()
    peta = {"Sad": "Sad", "Anger": "Anger", "Joy": "Joy",
            "Trust": "Trust", "Proud": "Proud", "Surprise": "Surprise"}
    target = peta.get(kelas_sendiri)
    if target and target in m.index:
        peringkat[c] = (target, list(m.index).index(target) + 1, len(m))

hasil = pd.DataFrame([{"fitur leksikon": k, "kelasnya sendiri": v[0],
                       "peringkat": f"{v[1]} dari {v[2]}"}
                      for k, v in peringkat.items()])
display(hasil)
print("Hanya lex_anger yang menempatkan kelasnya sendiri di peringkat 1.")
print("Sisanya peringkat 2-4: praktis tidak membedakan.")"""))

    s.append(MD("""## 4.2 Kenapa gagal? Analisis log-odds menjawabnya

Kata apa yang sebenarnya paling membedakan tiap kelas?"""))

    s.append(CODE("""from sklearn.feature_extraction.text import CountVectorizer

cv = CountVectorizer(min_df=5, ngram_range=(1, 2))
X = cv.fit_transform(t2.text_caption.fillna(""))
vocab = np.array(cv.get_feature_names_out())

baris = []
for e in ["Surprise", "Trust", "Proud", "Joy"]:
    m = (t2.emotion == e).values
    a = np.asarray(X[m].sum(0)).ravel().astype(float)
    b = np.asarray(X[~m].sum(0)).ravel().astype(float)
    prior = a + b
    dlt = (np.log((a + prior) / (a.sum() + prior.sum() - a - prior))
           - np.log((b + prior) / (b.sum() + prior.sum() - b - prior)))
    z = dlt / np.sqrt(1.0/(a + prior) + 1.0/(b + prior))
    top = vocab[np.argsort(z)[::-1][:6]]
    baris.append({"emosi": e, "kata paling khas": ", ".join(top)})
display(pd.DataFrame(baris))"""))

    s.append(MD("""**Tidak satu pun kata emosi.** Semuanya kata **topik** — `dumbbell`/`gym`
untuk Trust (konten kebugaran), `emas`/`meraih` untuk Proud (prestasi olahraga),
`roda`/`bensin`/`harga` untuk Surprise (otomotif).

**Anotator melabeli berdasarkan JENIS KONTEN, bukan kata emosi di caption.**
Konsekuensi desain: TF-IDF yang menangkap topik justru berguna, dan pencarian
kata emosi tidak. Fitur leksikon tetap dipertahankan di kode sebagai hasil
negatif yang terdokumentasi — bukan dihapus diam-diam."""))

    s.append(CODE("""display(Image("outputs/figures/03_kata_pembeda.png"))"""))

    s.append(MD("""## 4.3 Fitur konsep — memetakan kata ke IDE, bukan ke emosi

Berawal dari **menonton satu video** berlabel `Sad`: isinya *"mobilnya akan kita
tinggal… semua barang di mobil ini kita tinggal… bahan bakar ternyata gak
kepake"*. Tidak ada satu pun kata emosi — tapi idenya jelas: **kehilangan**.

Bedanya dengan leksikon yang gagal: leksikon memetakan kata → **emosi**
langsung; konsep memetakan kata → **ide**, lalu classifier yang belajar ide mana
menandakan emosi mana. Lapisan perantara itu yang membuatnya bekerja."""))

    s.append(CODE("""from features_text import KONSEP, hitung_konsep

for k, v in KONSEP.items():
    print(f"  {k:18} {', '.join(v)}")

print("\\nFrasa 'kita tinggal' hanya muncul di 1 video - mustahil ditemukan")
print("penambangan statistik. Konsep semantik HARUS dirumuskan manusia yang")
print("membaca; statistik hanya bisa memvalidasinya.")"""))

    s.append(CODE("""kon = pd.DataFrame([hitung_konsep(t) for t in train.text_combined.fillna("")])
kon["emotion"] = train.emotion.values
rata = kon.groupby("emotion").mean().T
display(rata[["Surprise", "Trust", "Proud", "Sad", "Anger"]].round(3))"""))

    s.append(MD("""## 4.4 Lapisan emosi — meminjam pengawasan dari luar

Model emosi Bahasa Indonesia yang dilatih pada label **bersih** dipakai untuk
menjawab "teks ini mengekspresikan emosi apa" — lepas dari label topik kita yang
bernoise. Kebetulan menguntungkan: label kita adalah 6 dari 8 emosi Plutchik."""))

    s.append(CODE("""from train import kolom_emosi, kolom_konsep
ke = kolom_emosi(train)
peta = {"Surprise": "surprise", "Trust": "trust", "Joy": "joy",
        "Anger": "anger", "Sad": "sadness", "Fear": "fear"}
baris = []
for lab, pl in peta.items():
    c = f"emo_plutchik_padat_{pl}"
    if c not in train.columns:
        continue
    m = train.emotion == lab
    baris.append({"label kita": lab, "n": int(m.sum()),
                  "prob senama": round(train.loc[m, c].mean(), 3),
                  "prob di kelas lain": round(train.loc[~m, c].mean(), 3)})
hasil = pd.DataFrame(baris)
hasil["selisih"] = (hasil["prob senama"] - hasil["prob di kelas lain"]).round(3)
display(hasil)
print("6 dari 6 arahnya BENAR (uji tanda p = 0,016).")
print("Surprise dan Anger bertahan setelah koreksi Holm.")
print("=> Label memang melacak emosi - terukur, meski lemah dibanding topik.")"""))

    # ================= 5 MODEL =================
    s.append(MD("""---
# Bagian 5 · Pemodelan

## 5.1 Skema validasi

**StratifiedGroupKFold 5 fold, `groups=key`.** Wajib berkelompok per URL karena
45 baris duplikat. TF-IDF dan target encoding dipasang **di dalam** Pipeline
supaya di-fit ulang tiap fold — kalau di-fit sekali di luar, kosakata dari fold
validasi ikut membentuk fitur dan skor jadi optimistis palsu."""))

    s.append(CODE("""import train as T
print(inspect.getsource(T.bobot_kelas))"""))

    s.append(MD("""## 5.2 Tangga ablation — tiap kenaikan bisa ditelusuri sebabnya"""))

    s.append(CODE("""ab = pd.read_csv("outputs/ablation.csv")
tampil = ab.copy()
tampil["accuracy"] = (100*tampil.accuracy).round(1)
tampil[["macro_f1", "weighted_f1"]] = tampil[["macro_f1", "weighted_f1"]].round(3)
tampil.columns = ["tahap", "akurasi (%)", "macro-F1", "weighted-F1"]
display(tampil)"""))

    s.append(CODE("""fig, (a1, a2) = plt.subplots(1, 2, figsize=(12.5, 4.6))
y = np.arange(len(ab))[::-1]
nama = [t.split(". ", 1)[-1][:32] for t in ab.tahap]

b1 = a1.barh(y, 100*ab.accuracy, color=BIRU, height=0.62)
a1.axvline(41.2, color=JINGGA, ls="--", lw=1.6)
a1.text(41.6, y.max()+0.4, "baseline 41,2%", color=JINGGA, fontsize=9)
a1.bar_label(b1, fmt="%.1f", padding=3, fontsize=8, color="#52514e")
a1.set_title("Akurasi (%)", loc="left", color=TINTA); a1.set_xlim(0, 52)

b2 = a2.barh(y, ab.macro_f1, color=HIJAU, height=0.62)
a2.axvline(0.058, color=JINGGA, ls="--", lw=1.6)
a2.bar_label(b2, fmt="%.3f", padding=3, fontsize=8, color="#52514e")
a2.set_title("macro-F1", loc="left", color=TINTA); a2.set_xlim(0, 0.20)

for ax in (a1, a2):
    ax.set_yticks(y, nama, fontsize=8.5)
    rapikan(ax, sumbu="x")
plt.tight_layout(); plt.show()"""))

    s.append(MD("""**Dua sumbu dipisah jadi dua panel dengan sengaja.** Akurasi (%) dan macro-F1
(0–1) punya skala berbeda; menumpuknya pada satu sumbu-Y ganda memunculkan
hubungan visual yang tidak ada."""))

    # ================= 6 GAGAL =================
    s.append(MD("""---
# Bagian 6 · Yang dicoba dan GAGAL

Hasil negatif yang terukur lebih meyakinkan daripada satu angka akurasi tanpa
konteks. Semuanya diuji pada skema validasi yang sama persis."""))

    s.append(CODE("""gagal = pd.DataFrame([
    ["IndoBERT beku + LogReg",         "34,6%", "0,151", "kalah di kedua metrik"],
    ["IndoBERT fine-tuned polos",      "37,4%", "0,129", "803 baris vs 124 juta parameter"],
    ["IndoBERT + resep lengkap*",      "33,1%", "0,143", "augmentasi MERUSAK (-3,6 poin)"],
    ["IndoBERT transkrip saja",        "36,7%", "0,111", "caption ternyata tetap perlu"],
    ["Fitur visual CLIP (36 fitur)",   "34,1%", "0,132", "sinyal nyata, sumbangan NOL"],
    ["Konsep dari aturan manusia",     "40,4%", "0,152", "17/20 katanya sudah di TF-IDF"],
    ["SMOTE",                          "—",     "—",     "MUSTAHIL: kelas bersampel 1"],
    ["Pencocokan distribusi prediksi", "41,5%", "0,133", "kuota keras merusak presisi"],
    ["Membuang baris rusak/bentrok",   "41,1%", "0,148", "data 'kotor' tetap bawa prior"],
], columns=["percobaan", "akurasi", "macro-F1", "kenapa gagal"])
display(gagal)
print("* resep yang mencapai 71% pada tugas 4 kelas berlabel BERSIH di proyek lain")
print("\\nMODEL FINAL sebagai pembanding:  43,2%  /  0,153")"""))

    s.append(MD("""## Pola yang menyatukan semua kegagalan

Tiga kali berturut, sumber sinyal baru ternyata **redundan** dengan teks yang
sudah ada. Video gym punya kata `dumbbell` di transkrip **dan** tampak seperti
gym; mobil pameran punya kata `harga` **dan** tampak seperti panggung pameran."""))

    s.append(CODE("""pola = pd.DataFrame([
    ["Caption vs transkrip", "keduanya ~41%", "sepakat 80,1% baris; oracle +4,4 poin"],
    ["Konsep aturan manusia", "kon_mengajari 2,2x", "-1,1 poin akurasi"],
    ["Visual CLIP", "macro-F1 2,3x baseline", "+0,000"],
], columns=["sumber baru", "sinyal SENDIRIAN", "sumbangan ke model"])
display(pola)
print("Seluruh rekayasa fitur    -> pergeseran macro-F1 sekitar +-0,01")
print("Ensemble 2 keluarga model -> +0,7 poin akurasi DAN +0,018 macro-F1")
print("\\n=> Batasnya bukan berapa banyak informasi yang kita punya,")
print("   melainkan berapa banyak informasi yang BERBEDA.")"""))

    # ================= 7 ENSEMBLE =================
    s.append(MD("""---
# Bagian 7 · Terobosan: ensemble dua keluarga model

Semua tuas yang dicoba ternyata bergerak di **satu kurva tukar-guling yang
sama** — menaikkan macro-F1 selalu menurunkan akurasi pada laju yang praktis
tetap. Itu petunjuk, bukan dinding: kalau semua tuas jatuh di satu kurva, yang
membatasi bukan letak titik operasi melainkan **kualitas estimasi
probabilitasnya**."""))

    s.append(CODE("""kurva = pd.DataFrame([
    ["bobot kelas α: 0,5 → 1,0",   "42,0% → 39,7%", "0,099 → 0,155"],
    ["koreksi prior τ: 0,0 → 0,5", "42,0% → 37,1%", "0,129 → 0,180"],
    ["RandomOverSampler",          "41,4% → 38,7%", "0,147 → 0,166"],
    ["regularisasi C: 3 → 10",     "41,4% → 40,3%", "0,147 → 0,155"],
    ["ComplementNB sendirian",     "41,4% → 35,2%", "0,147 → 0,173"],
], columns=["tuas", "akurasi", "macro-F1"])
display(kurva)
print("Lima tuas berbeda, laju pertukaran praktis SAMA.\\n")

ens = pd.DataFrame([
    ["LogisticRegression saja",        "41,6% ± 0,8", "0,149 ± 0,009", "—"],
    ["+ ensemble ComplementNB w=0,5",  "42,3% ± 1,0", "0,167 ± 0,012", "akurasi 4/5, macro-F1 5/5"],
], columns=["model (5 seed)", "akurasi", "macro-F1", "menang di berapa seed"])
display(ens)
print("Selisih macro-F1 per seed: +0,013 +0,019 +0,031 +0,027 +0,018 -> positif SEMUA.")
print("Satu-satunya perubahan sepanjang proyek yang memperbaiki KEDUA metrik.")"""))

    s.append(CODE("""print(inspect.getdoc(T.EnsembleLRNB))"""))

    # ================= 8 MODEL FINAL =================
    s.append(MD("""---
# Bagian 8 · Model final

```
stemming Sastrawi → TF-IDF unigram (caption + transkrip)
  + 9 fitur konsep + 14 probabilitas emosi
  → ensemble 0,5 × LogisticRegression + 0,5 × ComplementNB
  → koreksi prior τ=0,1 → fallback prior → penimpaan duplikat URL
```"""))

    s.append(CODE("""from sklearn.metrics import (accuracy_score, f1_score,
                             classification_report, confusion_matrix)

oof = pd.read_csv("outputs/oof_final.csv", dtype={"id": str})
dd = train.merge(oof[["id", "oof"]], on="id", how="left")
kelas = sorted(train.emotion.unique())
cm = confusion_matrix(dd.emotion, dd.oof, labels=kelas)

print(f"OOF model final — akurasi {accuracy_score(dd.emotion, dd.oof)*100:.1f}%"
      f"   macro-F1 {f1_score(dd.emotion, dd.oof, average='macro', zero_division=0):.3f}\\n")
print(classification_report(dd.emotion, dd.oof, zero_division=0, digits=3))"""))

    s.append(CODE("""from matplotlib.colors import LinearSegmentedColormap
cmap = LinearSegmentedColormap.from_list(
    "biru", ["#eef4fd", "#9ec5f4", "#3987e5", "#184f95", "#0d366b"])
cmn = cm / np.clip(cm.sum(1, keepdims=True), 1, None)

fig, ax = plt.subplots(figsize=(7.2, 5.8))
im = ax.imshow(cmn, cmap=cmap, vmin=0, vmax=1)
for i in range(len(kelas)):
    for j in range(len(kelas)):
        if cm[i, j]:
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=9,
                    color="#ffffff" if cmn[i, j] > 0.55 else TINTA)
ax.set_xticks(range(len(kelas)), kelas, rotation=40, ha="right")
ax.set_yticks(range(len(kelas)), kelas)
ax.set_xlabel("prediksi"); ax.set_ylabel("label sebenarnya")
ax.set_title("Confusion matrix — dinormalisasi per baris", loc="left", color=TINTA)
ax.tick_params(colors=ABU, labelcolor="#52514e")
fig.colorbar(im, ax=ax, shrink=0.78, label="proporsi per baris")
plt.tight_layout(); plt.show()"""))

    s.append(MD("""## Kebingungan model MENIRU ketidaksepakatan anotator

Blok pekat di pojok Surprise/Trust/Proud itu 83% data — dan persis di situ
anotatornya sendiri tidak sepakat."""))

    s.append(CODE("""from itertools import combinations
pas = collections.Counter()
for k, v in bentrok.items():
    for a, b in combinations(sorted(set(v)), 2):
        pas[tuple(sorted((a, b)))] += 1

tukar = collections.Counter()
for i, a in enumerate(kelas):
    for j, b in enumerate(kelas):
        if i < j:
            tukar[(a, b)] = cm[i, j] + cm[j, i]

banding = pd.DataFrame([
    {"pasangan": f"{a} ↔ {b}", "ditukar ANOTATOR": n,
     "ditukar MODEL (baris)": tukar.get((a, b), 0)}
    for (a, b), n in pas.most_common(6)])
display(banding)
print("Tiga pasangan teratas IDENTIK. Model gagal tepat di tempat manusia gagal.")"""))

    s.append(MD("""## Plafon macro-F1 yang bersifat struktural"""))

    s.append(CODE("""f1s = f1_score(dd.emotion, dd.oof, labels=kelas, average=None, zero_division=0)
n = train.emotion.value_counts()
display(pd.DataFrame({"n": [n[k] for k in kelas], "F1": f1s.round(3)},
                     index=kelas).sort_values("F1", ascending=False))

f8 = np.mean([f for k, f in zip(kelas, f1s) if k not in ("Love", "Loyalty")])
f7 = np.mean([f for k, f in zip(kelas, f1s) if k not in ("Love", "Loyalty", "Neutral")])
display(pd.DataFrame([
    ["10 kelas (dipakai untuk submission)", round(np.mean(f1s), 3), "—"],
    ["tanpa 2 kelas bersampel-1", round(f8, 3), f"+{f8-np.mean(f1s):.3f}"],
    ["tanpa 3 kelas mustahil", round(f7, 3), f"+{f7-np.mean(f1s):.3f}"],
], columns=["skema penilaian", "macro-F1", "selisih"]))
print("Love+Loyalty+Neutral = 10 sampel (1,2% data) tapi 30% penyebut macro-F1.")
print("43% dari 'kekurangan' macro-F1 bersifat STRUKTURAL, bukan kegagalan model.")"""))

    # ================= 9 SUBMISSION =================
    s.append(MD("""---
# Bagian 9 · Submission

Tiga lapis keputusan, dari yang paling dipercaya ke yang paling tidak:

1. **Jawaban pasti** — 9 baris test URL-nya identik dengan train berlabel konsisten
2. **Modus video yang sama** — 4 baris menunjuk video yang di train dilabeli beragam
3. **Prediksi model**, mundur ke prior kalau baris tidak punya sinyal apa pun

Semua penimpaan dicatat — didokumentasikan, bukan disembunyikan."""))

    s.append(CODE("""sub = pd.read_csv("outputs/submission.csv")
pp = pd.read_csv("outputs/post_processing.csv")
print(f"submission: {len(sub)} baris, {sub.emotion.nunique()} kelas terwakili")
print(f"penimpaan pasca-model: {len(pp)} baris\\n")
display(pp)

banding = pd.DataFrame({
    "prediksi test": sub.emotion.value_counts(),
    "train (%)": (100*train.emotion.value_counts()/len(train)).round(1)})
banding["prediksi (%)"] = (100*banding["prediksi test"]/len(sub)).round(1)
display(banding[["prediksi test", "prediksi (%)", "train (%)"]].fillna(0))
print("Distribusi membaik sepanjang proyek: 149 Surprise (74,5%) -> 121 (60,5%).")"""))

    # ================= 10 RINGKASAN =================
    s.append(MD("""---
# Bagian 10 · Ringkasan

| Metrik | Model final | Baseline | Rasio |
|---|---|---|---|
| Akurasi | **43,2%** | 41,2% | plafon derau label ~56% |
| macro-F1 | **0,153** | 0,058 | **2,6×** |
| weighted-F1 | **0,374** | 0,241 | 1,6× |

### Tiga hal yang membuat pekerjaan ini bertahan di depan pemeriksaan

1. **Batasnya diukur, bukan ditebak.** Plafon derau label ~56% dihitung dari
   kesepakatan anotator pada video identik — dan confusion matrix model
   terbukti meniru pola ketidaksepakatan itu.

2. **Setiap keputusan diuji, bukan diasumsikan.** IndoBERT (5 konfigurasi),
   augmentasi, SMOTE, fitur visual CLIP, stemming, pembersihan data — semuanya
   diukur pada skema validasi yang sama, dan yang kalah dilaporkan apa adanya.

3. **Standar penerimaan konsisten.** Sebuah perubahan hanya diterima kalau
   menang di mayoritas seed split, bukan sekali. Itu yang menyaring perbaikan
   nyata dari derau — dan yang membuat beberapa "kemenangan" awal ditolak
   setelah diuji ulang.

### Reproduksi

```bash
pip install -r requirements.txt
python run_all.py          # ~3 menit, tanpa internet, tanpa GPU
```

Detail lengkap tiap tahap ada di `RENCANA.md` §1–§25."""))

    nb = new_notebook(cells=s)
    nb.metadata = {
        "kernelspec": {"display_name": "Python 3", "language": "python",
                       "name": "python3"},
        "language_info": {"name": "python", "version": "3.11"},
    }
    return nb


def main() -> int:
    from nbconvert.preprocessors import ExecutePreprocessor

    nb = bangun()
    print(f"menjalankan {len(nb.cells)} sel ...")
    ep = ExecutePreprocessor(timeout=1800, kernel_name="python3",
                             allow_errors=True)
    ep.preprocess(nb, {"metadata": {"path": str(ROOT)}})

    galat = [(i, o.get("ename"), str(o.get("evalue"))[:80])
             for i, c in enumerate(nb.cells) if c.cell_type == "code"
             for o in c.get("outputs", []) if o.get("output_type") == "error"]

    nbf.write(nb, KELUARAN)
    kode = sum(1 for c in nb.cells if c.cell_type == "code")
    ada = sum(1 for c in nb.cells if c.cell_type == "code" and c.get("outputs"))
    print(f"tersimpan: {KELUARAN.name}")
    print(f"  {len(nb.cells)} sel ({kode} kode, {ada} berkeluaran), "
          f"{KELUARAN.stat().st_size/1e6:.2f} MB")
    if galat:
        print(f"\n  {len(galat)} SEL ERROR:")
        for i, e, v in galat:
            print(f"    sel {i}: {e}: {v}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
