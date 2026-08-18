"""
Kalibrasi keputusan - memperbaiki dua penyakit yang terukur, bukan menambah fitur.

Diagnosa galat pada OOF model final menunjukkan dua kerugian yang tidak
disebabkan oleh kurangnya sinyal, melainkan oleh cara keputusan diambil:

**1. Over-prediksi kelas mayoritas.** `Surprise` ditebak 482 kali padahal
aslinya 331; `Fear` ditebak 3 kali padahal 16, `Sad` 3 kali padahal 18. Argmax
pada probabilitas yang condong ke prior membuat kelas kecil tidak pernah menang
- dan macro-F1 menghukum itu tepat sasaran. Obatnya koreksi prior:

    prediksi = argmax  P(y|x) / P(y)^tau

`tau=0` sama dengan argmax biasa; `tau=1` membagi habis pengaruh prior. Nilai
di antaranya memberi titik tukar-guling. **tau ditala DI DALAM fold** memakai
validasi dalam - kalau ditala pada fold yang dinilai, skornya bohong.

**2. Baris tanpa teks lebih buruk daripada tebakan buta.** 94 baris train tidak
punya caption maupun transkrip. Akurasi model di sana 27,7%, sementara menebak
`Surprise` untuk semuanya memberi ~41%. Model memaksakan tebakan dari fitur
kosong dan kalah dari prior. §13 sudah menuliskan ini sebagai keharusan
("wajib ada penjaga fallback") tapi belum pernah diterapkan pada penilaian.

Keduanya murni pasca-pemrosesan: tidak ada fitur baru, tidak ada model baru,
tidak ada risiko overfit tambahan selain dari tau yang ditala di dalam fold.

Jalankan:  python src/kalibrasi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES_DIR, OUTPUTS_DIR, SEED  # noqa: E402

ALPHA = 0.5
BOBOT_EMOSI = 0.25
SEEDS = [42, 7, 2024]
TAU_GRID = np.arange(0.0, 1.01, 0.1)


def prediksi_terkalibrasi(proba: np.ndarray, kelas: np.ndarray,
                          prior: np.ndarray, tau: float) -> np.ndarray:
    """argmax P(y|x) / P(y)^tau, dihitung di ruang log demi kestabilan."""
    skor = np.log(np.clip(proba, 1e-12, None)) - tau * np.log(np.clip(prior, 1e-12, None))
    return kelas[skor.argmax(1)]


def tala_tau(proba: np.ndarray, y: np.ndarray, kelas: np.ndarray,
             prior: np.ndarray) -> float:
    """Pilih tau yang memaksimalkan macro-F1 pada data yang diberikan."""
    terbaik, tau_terbaik = -1.0, 0.0
    for tau in TAU_GRID:
        f = f1_score(y, prediksi_terkalibrasi(proba, kelas, prior, tau),
                     average="macro", zero_division=0)
        if f > terbaik:
            terbaik, tau_terbaik = f, tau
    return tau_terbaik


def jalankan(df: pd.DataFrame, bikin, seed: int, pakai_fallback: bool,
             pakai_tau: bool) -> tuple[np.ndarray, np.ndarray, list[float]]:
    import train as T

    y = df["emotion"].to_numpy()
    grup = df["key"].to_numpy()
    kosong = (df["text_combined"].fillna("").str.strip() == "").to_numpy()

    oof = np.empty(len(df), dtype=object)
    taus = []
    kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=seed)

    for a, b in kf.split(df, y, groups=grup):
        p = clone(bikin())
        p.set_params(clf__class_weight=T.bobot_kelas(y[a], ALPHA))

        if pakai_tau:
            # tau ditala pada validasi DALAM, dipotong lagi dari fold-train.
            # Tanpa ini tau akan disetel memakai baris yang dinilai -> bocor.
            kf2 = StratifiedGroupKFold(n_splits=4, shuffle=True, random_state=seed)
            ia, ib = next(kf2.split(df.iloc[a], y[a], groups=grup[a]))
            dalam, vdalam = a[ia], a[ib]
            p2 = clone(bikin())
            p2.set_params(clf__class_weight=T.bobot_kelas(y[dalam], ALPHA))
            p2.fit(df.iloc[dalam], y[dalam])
            prior_d = np.array([(y[dalam] == c).mean() for c in p2.classes_])
            tau = tala_tau(p2.predict_proba(df.iloc[vdalam]), y[vdalam],
                           p2.classes_, prior_d)
        else:
            tau = 0.0
        taus.append(tau)

        p.fit(df.iloc[a], y[a])
        prior = np.array([(y[a] == c).mean() for c in p.classes_])
        pred = prediksi_terkalibrasi(p.predict_proba(df.iloc[b]), p.classes_,
                                     prior, tau)

        if pakai_fallback:
            # Baris tanpa teks: model kalah dari prior, jadi pakai prior fold ini.
            modus = p.classes_[prior.argmax()]
            pred = np.where(kosong[b], modus, pred)

        oof[b] = pred

    return oof, y, taus


def main() -> int:
    from eval_emosi import kolom, muat, pipa

    df = muat()
    kon = kolom(df, "kon_")
    emo = [c for c in df.columns if c.startswith("emo_") and "_padat_" in c]
    df["text_emosi"] = df["text_emosi"].fillna("")

    kandidat = {
        "acuan (topik saja)": lambda: pipa("text_combined", [], 0, kon),
        "topik + emosi": lambda: pipa("text_combined", emo, BOBOT_EMOSI, kon),
    }
    varian = [
        ("apa adanya (argmax)", False, False),
        ("+ fallback teks kosong", True, False),
        ("+ kalibrasi prior (tau)", False, True),
        ("+ keduanya", True, True),
    ]

    print(f"KALIBRASI KEPUTUSAN - {len(SEEDS)} seed, mean +- std")
    print(f"{'model':22} {'varian':26} {'akurasi':>14} {'macro-F1':>15}")
    print("-" * 80)

    hasil = []
    for nama_m, bikin in kandidat.items():
        for nama_v, fb, tau in varian:
            A, F, T_ = [], [], []
            for s in SEEDS:
                oof, y, taus = jalankan(df, bikin, s, fb, tau)
                A.append(accuracy_score(y, oof))
                F.append(f1_score(y, oof, average="macro", zero_division=0))
                T_ += taus
            A, F = np.array(A), np.array(F)
            hasil.append({"model": nama_m, "varian": nama_v,
                          "accuracy": A.mean(), "acc_std": A.std(),
                          "macro_f1": F.mean(), "f1_std": F.std(),
                          "tau_mean": float(np.mean(T_)) if tau else 0.0})
            ket = f"  tau~{np.mean(T_):.2f}" if tau else ""
            print(f"{nama_m:22} {nama_v:26} {100*A.mean():7.1f}% +-{100*A.std():3.1f} "
                  f"{F.mean():9.3f} +-{F.std():.3f}{ket}", flush=True)

    print("-" * 80)
    print("acuan: baseline tebak Surprise  41,2% / 0,058 | model final §17  42,3% / 0,135")

    tabel = pd.DataFrame(hasil)
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    tabel.to_csv(OUTPUTS_DIR / "kalibrasi.csv", index=False)

    terbaik = tabel.loc[tabel.macro_f1.idxmax()]
    print(f"\nterbaik macro-F1: {terbaik['model']} / {terbaik['varian']} "
          f"-> {100*terbaik['accuracy']:.1f}% / {terbaik['macro_f1']:.3f}")

    # laporan per kelas untuk konfigurasi terbaik, seed utama
    bikin = kandidat[terbaik["model"]]
    fb, tau = dict((v[0], (v[1], v[2])) for v in varian)[terbaik["varian"]]
    oof, y, _ = jalankan(df, bikin, SEED, fb, tau)
    print(classification_report(y, oof, zero_division=0))
    pd.DataFrame({"id": df["id"], "y": y, "oof": oof}).to_csv(
        OUTPUTS_DIR / "oof_kalibrasi.csv", index=False)
    print(f"disimpan: {OUTPUTS_DIR/'kalibrasi.csv'}, {OUTPUTS_DIR/'oof_kalibrasi.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
