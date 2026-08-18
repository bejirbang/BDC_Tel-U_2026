"""
Pencocokan distribusi prediksi - menyerang gejala yang sudah terukur di §22.

Model memprediksi `Surprise` untuk 60% baris test, padahal proporsinya di train
41,2%. Kalau test berasal dari populasi yang sama - asumsi yang wajar dan bisa
dinyatakan terbuka di laporan - maka distribusi prediksi kita **salah secara
sistematis**, dan kesalahan itu bisa diperbaiki tanpa menyentuh modelnya.

Caranya: cari pengali per-kelas `lambda_c` sehingga

    prediksi = argmax_c  P(c|x) * lambda_c

menghasilkan jumlah prediksi yang cocok dengan proporsi harapan. Ini
**generalisasi dari `tau`** (§20): `tau` memakai satu parameter global
`P(c)^-tau` untuk semua kelas sekaligus, sedangkan di sini tiap kelas punya
pengali sendiri yang ditala sampai jumlahnya pas. Karena itu ia bisa
memperbaiki kelas yang kurang-prediksi dan kelebih-prediksi secara terpisah -
sesuatu yang `tau` tidak bisa.

Penalaannya memakai iterative proportional fitting (Sinkhorn satu sisi):
naikkan pengali kelas yang kurang terwakili, turunkan yang kelebihan, ulangi
sampai konvergen. Tidak ada label yang dipakai - hanya proporsi harapan, yang
diambil dari fold-train. Jadi tidak ada kebocoran.

Jalankan:  python src/cocok_distribusi.py
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
from common import FEATURES_DIR, OUTPUTS_DIR, SEED  # noqa: E402

warnings.filterwarnings("ignore")

ALPHA = 0.5
TAU = 0.1
SEEDS = [42, 7, 2024, 13, 99]
N_ITER = 200


def pengali_sinkhorn(proba: np.ndarray, target: np.ndarray,
                     n_iter: int = N_ITER, lr: float = 0.5) -> np.ndarray:
    """Cari pengali per-kelas agar distribusi argmax mendekati `target`.

    Bekerja di ruang log. Tiap putaran: bandingkan proporsi prediksi sekarang
    dengan target, lalu geser log-pengali sebesar log(target/sekarang) yang
    diredam `lr`. Peredaman perlu karena argmax itu tidak kontinu - tanpa itu
    pengalinya berayun dan tidak pernah menetap.
    """
    n, k = proba.shape
    logp = np.log(np.clip(proba, 1e-12, None))
    lam = np.zeros(k)
    target = np.clip(target, 1e-6, None)
    target = target / target.sum()

    terbaik, jarak_terbaik = lam.copy(), np.inf
    for _ in range(n_iter):
        pred = (logp + lam).argmax(1)
        kini = np.bincount(pred, minlength=k) / n
        jarak = np.abs(kini - target).sum()
        if jarak < jarak_terbaik:
            jarak_terbaik, terbaik = jarak, lam.copy()
        if jarak < 1e-3:
            break
        # kelas yang kurang terwakili dinaikkan, yang kelebihan diturunkan
        lam = lam + lr * (np.log(target) - np.log(np.clip(kini, 1e-6, None)))
        lam -= lam.mean()          # jaga agar tidak melayang
    return terbaik


def main() -> int:
    import train as T
    from train import (KOLOM_TEKS_FINAL, NGRAM_FINAL, buat_pipeline_teks,
                       kolom_emosi, kolom_konsep)

    df = pd.read_parquet(FEATURES_DIR / "dataset.parquet")
    df = df[df.split == "train"].reset_index(drop=True)
    kk, ke = kolom_konsep(df), kolom_emosi(df)
    y = df["emotion"].to_numpy()
    g = df["key"].to_numpy()
    kosong = (df["text_combined"].fillna("").str.strip() == "").to_numpy()

    def jalankan(mode: str, seed: int):
        kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)
        oof = np.empty(len(df), dtype=object)
        for a, b in kf.split(df, y, groups=g):
            p = clone(buat_pipeline_teks(ALPHA, kol_konsep=kk, kol_emosi=ke,
                                         kolom_teks=KOLOM_TEKS_FINAL,
                                         ngram=NGRAM_FINAL, ensemble=True))
            p.set_params(clf__class_weight=T.bobot_kelas(y[a], ALPHA))
            p.fit(df.iloc[a], y[a])
            P = p.predict_proba(df.iloc[b])
            cls = p.classes_
            prior = np.array([(y[a] == c).mean() for c in cls])

            if mode == "tau":
                skor = np.log(np.clip(P, 1e-12, None)) - TAU * np.log(np.clip(prior, 1e-12, None))
                pred = cls[skor.argmax(1)]
            else:
                # target diambil dari FOLD-TRAIN, bukan dari fold yang dinilai
                lam = pengali_sinkhorn(P, prior)
                pred = cls[(np.log(np.clip(P, 1e-12, None)) + lam).argmax(1)]

            pred = np.where(kosong[b], cls[prior.argmax()], pred)
            oof[b] = pred
        return oof

    print("PENCOCOKAN DISTRIBUSI vs KALIBRASI TAU  (5 seed, model final §24)")
    print(f"{'metode':28} {'akurasi':>16} {'macro-F1':>16}")
    print("-" * 64)
    hasil = {}
    for mode, nama in [("tau", "tau=0,1 (sekarang)"),
                       ("sinkhorn", "cocok distribusi")]:
        A, F = [], []
        for s in SEEDS:
            oof = jalankan(mode, s)
            A.append(accuracy_score(y, oof))
            F.append(f1_score(y, oof, average="macro", zero_division=0))
        A, F = np.array(A), np.array(F)
        hasil[mode] = (A, F)
        print(f"{nama:28} {100*A.mean():7.1f}% +-{100*A.std():3.1f} "
              f"{F.mean():10.3f} +-{F.std():.3f}")

    a0, f0 = hasil["tau"]
    a1, f1 = hasil["sinkhorn"]
    print("-" * 64)
    print("selisih per seed (cocok-distribusi - tau):")
    print("  akurasi :", [f"{100*x:+.1f}" for x in (a1 - a0)],
          f" menang {(a1 > a0).sum()}/{len(SEEDS)}")
    print("  macro-F1:", [f"{x:+.3f}" for x in (f1 - f0)],
          f" menang {(f1 > f0).sum()}/{len(SEEDS)}")

    oof = jalankan("sinkhorn", SEED)
    print("\ndistribusi prediksi (seed 42):")
    print(pd.DataFrame({
        "sebenarnya": pd.Series(y).value_counts(),
        "diprediksi": pd.Series(oof).value_counts(),
    }).fillna(0).astype(int).to_string())
    print("\n" + classification_report(y, oof, zero_division=0))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": df["id"], "y": y, "oof": oof}).to_csv(
        OUTPUTS_DIR / "oof_cocok_distribusi.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
