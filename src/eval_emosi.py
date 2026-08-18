"""
Evaluasi lapisan emosi - apakah menyerang emosi langsung mengalahkan topik?

Pertanyaannya tajam dan layak dijawab dengan tajam pula: seluruh proyek ini
menyimpulkan label melacak TOPIK, tapi kesimpulan itu selalu diukur lewat model
yang memang mencari topik (TF-IDF, IndoBERT pada teks penuh). Di sini sinyal
emosi diberi kesempatan berdiri sendiri.

Tangga ujinya disusun supaya tiap pertanyaan terjawab terpisah:

| Tahap | Menjawab |
|---|---|
| 0 | acuan model final (TF-IDF + konsep) |
| 1 | **probabilitas emosi SENDIRIAN** - sinyal emosi nyata atau tidak? |
| 2 | leksikon InSet sendirian - apakah leksikon terbitan lebih baik dari buatan tangan (§5A)? |
| 3 | emosi pada teks PADAT vs PENUH - hipotesis dilusi benar atau tidak? |
| 4 | TF-IDF pada teks padat - apakah memadatkan membantu model topik juga? |
| 5-7 | gabungan emosi + model final - apakah menambah di atas yang sudah ada? |

Tahap 1 dan 3 adalah inti ilmiahnya. Tahap 1 menguji premis; tahap 3 menguji
penjelasan kenapa premis itu dulu tampak salah.

Skema CV identik dengan train.py: StratifiedGroupKFold 5 fold, groups=key,
seed sama - supaya angkanya bisa langsung disandingkan dengan tabel ablation.

Jalankan:  python src/eval_emosi.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES_DIR, OUTPUTS_DIR, SEED  # noqa: E402

ALPHA = 0.5
C_REG = 3.0
BOBOT_KONSEP = 0.25
# Probabilitas emosi ada di skala 0-1 sementara TF-IDF sudah L2-normalized;
# tanpa penskalaan, 28 kolom emosi menenggelamkan ribuan kolom TF-IDF.
BOBOT_EMOSI = 0.25


def muat() -> pd.DataFrame:
    ds = pd.read_parquet(FEATURES_DIR / "dataset.parquet")
    emo = pd.read_parquet(FEATURES_DIR / "emosi.parquet")
    teks_e = pd.read_parquet(FEATURES_DIR / "teks_emosi.parquet")
    kunci = ["split", "id", "key"]
    df = ds.merge(emo.drop(columns=["emotion"]), on=kunci, how="left")
    df = df.merge(teks_e, on=kunci, how="left")
    return df[df.split == "train"].reset_index(drop=True)


def kolom(df: pd.DataFrame, awalan: str) -> list[str]:
    return [c for c in df.columns if c.startswith(awalan)]


def pipa(kolom_teks: str | None, kol_num: list[str], bobot_num: float,
         kol_konsep: list[str] | None = None, bakukan: bool = False) -> Pipeline:
    """Rakit pipeline dari komponen yang diminta.

    `bakukan` dipakai untuk fitur leksikon yang skalanya liar (skor InSet bisa
    -40..+40); probabilitas emosi sudah 0-1 jadi cukup diskalakan saja.
    """
    bagian = []
    if kolom_teks:
        bagian.append(("tfidf", TfidfVectorizer(min_df=2, ngram_range=(1, 2),
                                                sublinear_tf=True,
                                                strip_accents="unicode"), kolom_teks))
    if kol_konsep:
        bagian.append(("konsep", FunctionTransformer(
            lambda X: np.asarray(X) * BOBOT_KONSEP,
            feature_names_out="one-to-one"), kol_konsep))
    if kol_num:
        if bakukan:
            trafo = Pipeline([("sc", StandardScaler()),
                              ("bobot", FunctionTransformer(
                                  lambda X: np.asarray(X) * bobot_num))])
        else:
            trafo = FunctionTransformer(lambda X: np.asarray(X) * bobot_num,
                                        feature_names_out="one-to-one")
        bagian.append(("num", trafo, kol_num))

    return Pipeline([
        ("fitur", ColumnTransformer(bagian, remainder="drop")),
        ("clf", LogisticRegression(max_iter=3000, C=C_REG, random_state=SEED)),
    ])


def cv(df: pd.DataFrame, bikin, nama: str) -> dict:
    from sklearn.base import clone

    import train as T

    y = df["emotion"].to_numpy()
    grup = df["key"].to_numpy()
    oof = np.empty(len(df), dtype=object)
    kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    for a, b in kf.split(df, y, groups=grup):
        p = clone(bikin())
        p.set_params(clf__class_weight=T.bobot_kelas(y[a], ALPHA))
        p.fit(df.iloc[a], y[a])
        oof[b] = p.predict(df.iloc[b])
    return {"tahap": nama, "accuracy": accuracy_score(y, oof),
            "macro_f1": f1_score(y, oof, average="macro", zero_division=0),
            "_oof": oof, "_y": y}


def diagnosa_keselarasan(df: pd.DataFrame) -> None:
    """Apakah emosi yang dibaca model eksternal selaras dengan label kita?

    Ini pemeriksaan paling langsung dan paling mudah dibaca: untuk tiap label,
    berapa rata-rata probabilitas kelas Plutchik yang SENAMA. Kalau video
    berlabel Surprise memang lebih 'terbaca Surprise' daripada video lain,
    angka diagonalnya akan menonjol.
    """
    peta = {"Surprise": "surprise", "Trust": "trust", "Joy": "joy",
            "Anger": "anger", "Sad": "sadness", "Fear": "fear"}
    print("\n" + "=" * 74)
    print("DIAGNOSA: apakah pembacaan emosi model eksternal selaras dengan label?")
    print("=" * 74)
    print("  (prob rata-rata kelas Plutchik SENAMA, pada teks padat)\n")
    print(f"  {'label kita':10} {'n':>4}  {'prob senama':>11}  "
          f"{'prob di kelas lain':>18}  {'selisih':>8}")
    for lab, pl in peta.items():
        kol = f"emo_plutchik_padat_{pl}"
        if kol not in df.columns:
            continue
        m = df["emotion"] == lab
        if m.sum() == 0:
            continue
        dalam, luar = df.loc[m, kol].mean(), df.loc[~m, kol].mean()
        tanda = "  <-- selaras" if dalam > luar else ""
        print(f"  {lab:10} {int(m.sum()):4d}  {dalam:11.3f}  {luar:18.3f}  "
              f"{dalam-luar:+8.3f}{tanda}")


def main() -> int:
    df = muat()
    kon = kolom(df, "kon_")
    ins = kolom(df, "ins_")
    emo_penuh = [c for c in df.columns if c.startswith("emo_") and "_penuh_" in c]
    emo_padat = [c for c in df.columns if c.startswith("emo_") and "_padat_" in c]
    df["text_emosi"] = df["text_emosi"].fillna("")

    print(f"train {len(df)} baris | konsep {len(kon)} | InSet {len(ins)} | "
          f"emosi penuh {len(emo_penuh)} | emosi padat {len(emo_padat)}")

    diagnosa_keselarasan(df)

    tahapan = [
        ("0. acuan: TF-IDF gabungan + konsep",
         lambda: pipa("text_combined", [], 0, kon)),
        ("1. probabilitas emosi SAJA (padat)",
         lambda: pipa(None, emo_padat, 1.0)),
        ("1b. probabilitas emosi SAJA (penuh)",
         lambda: pipa(None, emo_penuh, 1.0)),
        ("2. leksikon InSet SAJA",
         lambda: pipa(None, ins, 1.0, bakukan=True)),
        ("3. emosi padat + InSet",
         lambda: pipa(None, emo_padat + ins, 1.0, bakukan=True)),
        ("4. TF-IDF pada teks PADAT saja",
         lambda: pipa("text_emosi", [], 0)),
        ("5. acuan + emosi padat",
         lambda: pipa("text_combined", emo_padat, BOBOT_EMOSI, kon)),
        ("6. acuan + emosi penuh",
         lambda: pipa("text_combined", emo_penuh, BOBOT_EMOSI, kon)),
        ("7. acuan + emosi padat + InSet",
         lambda: pipa("text_combined", emo_padat + ins, BOBOT_EMOSI, kon,
                      bakukan=True)),
    ]

    print("\n" + "=" * 74)
    print("TANGGA UJI (StratifiedGroupKFold 5 fold, groups=key, seed 42)")
    print("=" * 74)
    hasil = []
    for nama, bikin in tahapan:
        r = cv(df, bikin, nama)
        hasil.append(r)
        print(f"  {nama:36} acc {r['accuracy']*100:5.1f}%   "
              f"macroF1 {r['macro_f1']:.3f}", flush=True)

    print("\n  acuan pembanding:")
    print("    baseline tebak Surprise            acc  41,2%   macroF1 0,058")
    print("    model final (§17)                  acc  42,3%   macroF1 0,135")
    print("    TF-IDF bobot^1,0                   acc  39,7%   macroF1 0,155")

    terbaik = max(hasil, key=lambda r: r["macro_f1"])
    print(f"\nterbaik menurut macro-F1: {terbaik['tahap']}")
    print(classification_report(terbaik["_y"], terbaik["_oof"], zero_division=0))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                  for r in hasil]).to_csv(
        OUTPUTS_DIR / "ablation_emosi.csv", index=False)
    print(f"disimpan: {OUTPUTS_DIR/'ablation_emosi.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
