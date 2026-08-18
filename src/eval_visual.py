"""
Evaluasi fitur visual CLIP - apakah modalitas yang menganggur itu membawa sesuatu?

Diuji dalam tiga bentuk supaya sumbangannya terisolasi:

  1. **Visual SENDIRIAN** - menjawab pertanyaan paling mendasar: apakah tampilan
     video membawa sinyal label sama sekali? Kalau ini tidak mengalahkan
     baseline, sisanya tidak perlu dilanjutkan.
  2. **Model final + visual** - apakah menambah di atas yang sudah ada?
  3. **Sapuan bobot** - kalau menambah, di bobot berapa.

Skema CV identik dengan train.py, dan penilaiannya memakai standar penerimaan
yang sama dengan §17/§19/§24: harus menang konsisten lintas seed, bukan sekali.

Catatan cakupan: fitur visual hanya ada untuk 88,3% baris (sisanya videonya
mati sejak §11). Baris tanpa video diisi 0 - bukan NaN - karena LogisticRegression
tidak menerima NaN, dan 0 di sini bermakna "tidak ada bukti visual", bukan
"kemiripan nol".

Jalankan:  python src/eval_visual.py
"""

from __future__ import annotations

import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES_DIR, OUTPUTS_DIR  # noqa: E402

warnings.filterwarnings("ignore")

ALPHA = 0.5
TAU = 0.1
SEEDS = [42, 7, 2024, 13, 99]


def muat() -> pd.DataFrame:
    ds = pd.read_parquet(FEATURES_DIR / "dataset.parquet")
    vis = pd.read_parquet(FEATURES_DIR / "visual.parquet")
    df = ds.merge(vis.drop(columns=["key", "emotion"]), on=["split", "id"],
                  how="left", validate="one_to_one")
    kol = [c for c in vis.columns if c.startswith("vis_")]
    # 0 = "tidak ada bukti visual". NaN ditolak LogReg.
    df[kol] = df[kol].fillna(0.0)
    return df[df.split == "train"].reset_index(drop=True)


def main() -> int:
    import train as T
    from train import (KOLOM_TEKS_FINAL, NGRAM_FINAL, buat_pipeline_teks,
                       kolom_emosi, kolom_konsep)
    from kalibrasi import prediksi_terkalibrasi

    df = muat()
    kk, ke = kolom_konsep(df), kolom_emosi(df)
    kv = [c for c in df.columns if c.startswith("vis_")]
    y = df["emotion"].to_numpy()
    g = df["key"].to_numpy()
    kosong = (df["text_combined"].fillna("").str.strip() == "").to_numpy()
    ada_vis = (df[kv].abs().sum(axis=1) > 0).mean()
    print(f"train {len(df)} baris | fitur visual {len(kv)} | "
          f"cakupan visual {100*ada_vis:.1f}%")

    # Fitur visual WAJIB dibakukan sebelum dipakai. Skor kemiripan CLIP semuanya
    # berkerumun di pita sempit (0,18-0,26) dengan rerata ~0,21; yang membawa
    # informasi adalah SIMPANGANNYA, bukan levelnya. Tanpa pembakuan, mengalikan
    # bobot 0,25 hanya menghasilkan konstanta ~0,05 dan classifier tidak melihat
    # apa-apa. Ini beda mendasar dari fitur emosi (§19) yang sudah 0-1 dan
    # tersebar lebar, sehingga cukup diskalakan.
    from sklearn.compose import ColumnTransformer
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import (FunctionTransformer, MinMaxScaler,
                                       StandardScaler)
    from train import BOBOT_EMOSI, BOBOT_KONSEP, EnsembleLRNB

    def bikin(kol_konsep, kol_emosi, kol_visual, bobot_vis):
        bagian = [("tfidf", TfidfVectorizer(min_df=2, ngram_range=NGRAM_FINAL,
                                            sublinear_tf=True,
                                            strip_accents="unicode"),
                   KOLOM_TEKS_FINAL)]
        if kol_konsep:
            bagian.append(("konsep", FunctionTransformer(
                lambda X: np.asarray(X) * BOBOT_KONSEP), kol_konsep))
        if kol_emosi:
            bagian.append(("emosi", FunctionTransformer(
                lambda X: np.asarray(X) * BOBOT_EMOSI), kol_emosi))
        if kol_visual:
            # MinMax, bukan Standard: ComplementNB dalam ensemble menolak nilai
            # negatif, sementara pembakuan-z pasti menghasilkannya. MinMax
            # sama-sama melebarkan pita sempit CLIP tapi tetap non-negatif.
            bagian.append(("visual", Pipeline([
                ("sc", MinMaxScaler()),
                ("w", FunctionTransformer(
                    lambda X, w=bobot_vis: np.asarray(X) * w)),
            ]), kol_visual))
        return Pipeline([("fitur", ColumnTransformer(bagian)),
                         ("clf", EnsembleLRNB(C=3.0))])

    def uji(kol_konsep, kol_emosi, kol_visual, bobot_vis=0.25, teks=True):
        A, F = [], []
        for s in SEEDS:
            kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s)
            oof = np.empty(len(df), dtype=object)
            for a, b in kf.split(df, y, groups=g):
                if teks:
                    p = clone(bikin(kol_konsep, kol_emosi, kol_visual, bobot_vis))
                    p.set_params(clf__class_weight=T.bobot_kelas(y[a], ALPHA))
                    p.fit(df.iloc[a], y[a])
                    P, cls = p.predict_proba(df.iloc[b]), p.classes_
                else:
                    # visual sendirian: tanpa teks sama sekali
                    p = Pipeline([("sc", StandardScaler()),
                                  ("clf", LogisticRegression(
                                      max_iter=4000, C=1.0,
                                      class_weight=T.bobot_kelas(y[a], ALPHA)))])
                    p.fit(df.iloc[a][kol_visual], y[a])
                    P, cls = p.predict_proba(df.iloc[b][kol_visual]), p.classes_
                prior = np.array([(y[a] == c).mean() for c in cls])
                pr = prediksi_terkalibrasi(P, cls, prior, TAU)
                oof[b] = np.where(kosong[b], cls[prior.argmax()], pr)
            A.append(accuracy_score(y, oof))
            F.append(f1_score(y, oof, average="macro", zero_division=0))
        return np.array(A), np.array(F)

    print(f"\n{'konfigurasi':34} {'akurasi':>16} {'macro-F1':>16}")
    print("-" * 68)

    a_v, f_v = uji(None, None, kv, teks=False)
    print(f"{'visual SENDIRIAN (tanpa teks)':34} {100*a_v.mean():7.1f}% "
          f"+-{100*a_v.std():3.1f} {f_v.mean():10.3f} +-{f_v.std():.3f}")

    a0, f0 = uji(kk, ke, None)
    print(f"{'model final (§24)':34} {100*a0.mean():7.1f}% +-{100*a0.std():3.1f} "
          f"{f0.mean():10.3f} +-{f0.std():.3f}   <- acuan")

    hasil = {}
    for w in (0.05, 0.10, 0.25):
        a, f = uji(kk, ke, kv, bobot_vis=w)
        hasil[w] = (a, f)
        print(f"{'+ visual, bobot ' + str(w):34} {100*a.mean():7.1f}% "
              f"+-{100*a.std():3.1f} {f.mean():10.3f} +-{f.std():.3f}   "
              f"acc {(a > a0).sum()}/5, F1 {(f > f0).sum()}/5")

    print("-" * 68)
    print("baseline tebak Surprise: 41,2% / 0,058")
    terbaik = max(hasil, key=lambda w: hasil[w][1].mean())
    a, f = hasil[terbaik]
    print(f"\nbobot terbaik {terbaik}: akurasi {100*(a.mean()-a0.mean()):+.1f} poin, "
          f"macro-F1 {f.mean()-f0.mean():+.3f}")
    print("  selisih akurasi per seed:", [f"{100*x:+.1f}" for x in (a - a0)])
    print("  selisih macroF1 per seed:", [f"{x:+.3f}" for x in (f - f0)])

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([
        {"konfigurasi": "visual sendirian", "accuracy": a_v.mean(), "macro_f1": f_v.mean()},
        {"konfigurasi": "model final", "accuracy": a0.mean(), "macro_f1": f0.mean()},
        *[{"konfigurasi": f"+ visual w={w}", "accuracy": v[0].mean(),
           "macro_f1": v[1].mean()} for w, v in hasil.items()],
    ]).to_csv(OUTPUTS_DIR / "ablation_visual.csv", index=False)
    print(f"\ndisimpan: {OUTPUTS_DIR/'ablation_visual.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
