"""
Diagnosa: kenapa transkrip saja kalah dari caption + transkrip?

Uji di §18 punya satu cacat sebagai alat penjelas: cakupan caption (62,6%) dan
transkrip (87,5%) berbeda, jadi selisih skornya bisa berasal dari "berapa baris
yang terisi", bukan dari "seberapa informatif teksnya". Di sini cakupan
dikontrol - seluruh analisis dijalankan pada IRISAN baris yang punya caption
DAN transkrip sekaligus, sehingga tiap sumber dinilai pada baris yang sama
persis.

Tiga pengukuran:

1. **Daya prediksi per sumber**, CV identik dengan train.py. Menjawab: pada
   baris yang sama, mana yang lebih informatif?
2. **Kepadatan informasi**, yaitu daya diskriminatif per 1.000 kata. Menjawab:
   apakah caption menang karena isinya lebih baik, atau sekadar karena berbeda?
3. **Kata pembeda per sumber** lewat log-odds dengan prior Dirichlet. Menjawab:
   apa sebenarnya isi sinyal masing-masing sumber - kata emosi atau kata topik?

Dipakai TF-IDF + LogReg, bukan IndoBERT: pertanyaannya soal isi teks, bukan soal
arsitektur, dan model linear membuat sumbangan tiap kata bisa dibaca langsung.
Pola yang ditemukan di sini berlaku untuk kedua model - §14 dan §18 sama-sama
menemukan gabungan mengalahkan transkrip saja.

Jalankan:  python src/diagnosa_sumber_teks.py
"""

from __future__ import annotations

import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import accuracy_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES_DIR, OUTPUTS_DIR, SEED  # noqa: E402

SUMBER = {
    "caption saja": "text_all",
    "transkrip saja": "text_transcript",
    "caption + transkrip": "text_combined",
}
ALPHA = 0.5
MIN_HITUNG = 5      # kata terlalu jarang tidak bisa dinilai log-odds-nya


def log_odds_dirichlet(teks: list[str], y: np.ndarray,
                       kelas: str, min_hitung: int = MIN_HITUNG) -> pd.Series:
    """Log-odds ratio dengan prior Dirichlet informatif (Monroe dkk. 2008).

    Dipilih daripada selisih frekuensi mentah karena mengoreksi ragam: kata
    langka bisa punya rasio frekuensi ekstrem hanya karena kebetulan. Metode ini
    membagi dengan simpangan bakunya, sehingga skornya adalah z-score - bisa
    dibandingkan antar kata dengan frekuensi yang jauh berbeda.
    """
    cv = CountVectorizer(min_df=min_hitung, strip_accents="unicode")
    X = cv.fit_transform(teks)
    kosakata = np.array(cv.get_feature_names_out())

    mask = (y == kelas)
    a = np.asarray(X[mask].sum(0)).ravel().astype(float)    # hitungan di kelas
    b = np.asarray(X[~mask].sum(0)).ravel().astype(float)   # hitungan di luar
    prior = (a + b)                                         # prior = korpus penuh
    a0, b0 = prior.sum(), prior.sum()

    # odds terhadap prior, lalu dibakukan
    d = (np.log((a + prior) / (a.sum() + a0 - a - prior))
         - np.log((b + prior) / (b.sum() + b0 - b - prior)))
    var = 1.0 / (a + prior) + 1.0 / (b + prior)
    return pd.Series(d / np.sqrt(var), index=kosakata).sort_values(ascending=False)


def main() -> int:
    from sklearn.model_selection import StratifiedGroupKFold

    import train as T

    ds = pd.read_parquet(FEATURES_DIR / "dataset.parquet")
    tr = ds[ds.split == "train"].reset_index(drop=True)

    # KONTROL CAKUPAN: hanya baris yang punya kedua sumber.
    ada = ((tr["text_all"].fillna("").str.strip() != "")
           & (tr["text_transcript"].fillna("").str.strip() != ""))
    sub = tr[ada].reset_index(drop=True)
    y = sub["emotion"].to_numpy()
    grup = sub["key"].to_numpy()

    print(f"train penuh {len(tr)} baris -> irisan punya-keduanya {len(sub)} baris "
          f"({100*len(sub)/len(tr):.1f}%)")
    print(f"distribusi kelas pada irisan: "
          f"{dict(pd.Series(y).value_counts().head(5))}\n")

    # --- 1. Daya prediksi per sumber, cakupan terkontrol ---
    print("=" * 72)
    print("1. DAYA PREDIKSI PER SUMBER (baris identik, CV identik dengan train.py)")
    print("=" * 72)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    fold = list(cv.split(sub, y, groups=grup))
    hasil = []

    for nama, kol in SUMBER.items():
        oof = np.empty(len(sub), dtype=object)
        for a, b in fold:
            pipe = T.buat_pipeline_teks(ALPHA)
            pipe.set_params(clf__class_weight=T.bobot_kelas(y[a], ALPHA))
            pipe.fit(sub[kol].iloc[a], y[a])
            oof[b] = pipe.predict(sub[kol].iloc[b])
        acc = accuracy_score(y, oof)
        f1 = f1_score(y, oof, average="macro", zero_division=0)
        kata = sub[kol].str.split().str.len().median()
        hasil.append({"sumber": nama, "kata_median": int(kata),
                      "accuracy": acc, "macro_f1": f1})
        print(f"  {nama:22} {int(kata):4d} kata median   "
              f"acc {acc*100:5.1f}%   macroF1 {f1:.3f}")

    # --- 2. Kepadatan informasi: daya prediksi per 1.000 kata ---
    print()
    print("=" * 72)
    print("2. KEPADATAN INFORMASI (daya prediksi dibagi panjang teks)")
    print("=" * 72)
    dasar = 1.0 / len(np.unique(y))     # macro-F1 tebakan acak sebagai titik nol
    for h in hasil[:2]:
        lebih = h["macro_f1"] - 0.058   # macro-F1 baseline tebak Surprise
        print(f"  {h['sumber']:22} macroF1 di atas baseline {lebih:+.3f}  "
              f"per 100 kata: {100*lebih/h['kata_median']:+.4f}")
    r = (hasil[0]["macro_f1"] - 0.058) / hasil[0]["kata_median"]
    s = (hasil[1]["macro_f1"] - 0.058) / hasil[1]["kata_median"]
    print(f"\n  -> caption {r/s:.1f}x lebih padat per kata daripada transkrip"
          if s > 0 else "\n  -> transkrip tidak melampaui baseline")

    # --- 3. Kata pembeda per sumber ---
    print()
    print("=" * 72)
    print("3. KATA PEMBEDA PER SUMBER (log-odds Dirichlet, z-score)")
    print("=" * 72)
    besar = pd.Series(y).value_counts()
    baris = []
    for kelas in besar[besar >= 15].index:
        for nama, kol in list(SUMBER.items())[:2]:
            sk = log_odds_dirichlet(sub[kol].fillna("").tolist(), y, kelas)
            atas = sk.head(6)
            baris.append({"kelas": kelas, "sumber": nama,
                          "z_maks": atas.iloc[0],
                          "kata": ", ".join(atas.index)})
    tabel = pd.DataFrame(baris)
    for kelas in tabel["kelas"].unique():
        blok = tabel[tabel["kelas"] == kelas]
        print(f"\n  {kelas} (n={besar[kelas]})")
        for _, r_ in blok.iterrows():
            print(f"    {r_['sumber']:16} z={r_['z_maks']:5.2f}  {r_['kata']}")

    print()
    print("  z maks rata-rata per sumber:")
    for nama in list(SUMBER)[:2]:
        print(f"    {nama:22} {tabel[tabel['sumber']==nama]['z_maks'].mean():.2f}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hasil).to_csv(OUTPUTS_DIR / "diagnosa_sumber_teks.csv", index=False)
    tabel.to_csv(OUTPUTS_DIR / "diagnosa_kata_pembeda.csv", index=False)
    print(f"\ndisimpan: outputs/diagnosa_sumber_teks.csv, "
          f"outputs/diagnosa_kata_pembeda.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
