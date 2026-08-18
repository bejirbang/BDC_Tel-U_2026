"""
Menguji stemming dan char n-gram - dua cara berbeda menangani imbuhan.

Bahasa Indonesia sangat berimbuhan: `menang`, `pemenang`, `kemenangan`,
`memenangkan` berakar sama tapi jadi empat token berbeda. Dengan 803 baris dan
`min_df=2`, varian yang sendirian jarang akan terbuang - padahal gabungannya
cukup sering. Jurnal rujukan (Zakira dkk., JIEnGS 2025) memakai stemming sebagai
langkah baku.

Tapi ada risiko yang sudah terdokumentasi di §17: `tinggal` berarti
"ditinggalkan" DAN "cukup/hanya"; `parah` berarti buruk DAN keren. Stemming
menabrakkan makna-makna itu jadi satu token. Terverifikasi pada Sastrawi:

    "mobilnya ditinggalkan"  -> "mobil tinggal"
    "tinggal pake blush"     -> "tinggal pake blush"

Keduanya kini token `tinggal` yang sama.

**Char n-gram adalah jalan ketiga** yang mencapai tujuan sama tanpa risiko itu:
`memenangkan` dan `kemenangan` berbagi potongan `menang`, jadi kemiripannya
tertangkap TANPA memaksa keduanya jadi token identik. §5A merencanakannya tapi
tidak pernah masuk model final - jadi sekalian diuji di sini.

Empat konfigurasi diadu pada CV yang sama persis dengan train.py.

Jalankan:  python src/uji_stemming.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.pipeline import FeatureUnion, Pipeline
from sklearn.preprocessing import FunctionTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES_DIR, OUTPUTS_DIR, SEED  # noqa: E402

ALPHA = 0.5
SEEDS = [42, 7, 2024]
CACHE_STEM = FEATURES_DIR / "teks_stem.parquet"


def stem_semua(teks: list[str]) -> list[str]:
    """Stem seluruh dokumen, di-cache karena Sastrawi lambat (Python murni)."""
    from Sastrawi.Stemmer.StemmerFactory import StemmerFactory

    stemmer = StemmerFactory().create_stemmer()
    # Cache per KATA, bukan per dokumen: kosakata jauh lebih kecil daripada
    # jumlah token, jadi ini memangkas waktu secara drastis.
    memo: dict[str, str] = {}
    keluar = []
    t0 = time.time()
    for i, t in enumerate(teks):
        kata = (t or "").lower().split()
        hasil = []
        for k in kata:
            if k not in memo:
                memo[k] = stemmer.stem(k)
            hasil.append(memo[k])
        keluar.append(" ".join(hasil))
        if i % 200 == 0:
            print(f"  {i}/{len(teks)}  ({time.time()-t0:.0f} detik, "
                  f"{len(memo)} kata unik di-cache)", flush=True)
    return keluar


def muat() -> pd.DataFrame:
    ds = pd.read_parquet(FEATURES_DIR / "dataset.parquet")
    if CACHE_STEM.exists():
        st = pd.read_parquet(CACHE_STEM)
        ds = ds.merge(st, on=["split", "id"], how="left")
        print(f"memakai cache: {CACHE_STEM.name}")
    else:
        print("melakukan stemming (sekali saja, hasilnya di-cache) ...")
        ds["text_stem"] = stem_semua(ds["text_combined"].fillna("").tolist())
        ds[["split", "id", "text_stem"]].to_parquet(CACHE_STEM, index=False)
        print(f"disimpan: {CACHE_STEM}")
    return ds[ds.split == "train"].reset_index(drop=True)


def vek_kata() -> TfidfVectorizer:
    return TfidfVectorizer(min_df=2, ngram_range=(1, 2), sublinear_tf=True,
                           strip_accents="unicode")


def vek_char() -> TfidfVectorizer:
    # `char_wb` menghormati batas kata, jadi potongannya tidak melompati spasi -
    # lebih bersih daripada `char` untuk bahasa berimbuhan.
    return TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), min_df=3,
                           sublinear_tf=True, strip_accents="unicode")


def bikin(kolom_teks: str, pakai_char: bool, kon: list[str],
          emo: list[str]) -> Pipeline:
    from train import BOBOT_EMOSI, BOBOT_KONSEP

    if pakai_char:
        teks_trafo = FeatureUnion([("kata", vek_kata()), ("char", vek_char())])
    else:
        teks_trafo = vek_kata()

    bagian = [("teks", teks_trafo, kolom_teks)]
    if kon:
        bagian.append(("konsep", FunctionTransformer(
            lambda X: np.asarray(X) * BOBOT_KONSEP), kon))
    if emo:
        bagian.append(("emosi", FunctionTransformer(
            lambda X: np.asarray(X) * BOBOT_EMOSI), emo))
    return Pipeline([
        ("fitur", ColumnTransformer(bagian)),
        ("clf", LogisticRegression(max_iter=3000, C=3.0, random_state=SEED)),
    ])


def uji(df: pd.DataFrame, pipe, y, g) -> tuple[float, float, int]:
    import train as T

    A, F, dim = [], [], 0
    for s in SEEDS:
        kf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=s)
        oof = np.empty(len(df), dtype=object)
        for a, b in kf.split(df, y, groups=g):
            p = clone(pipe)
            p.set_params(clf__class_weight=T.bobot_kelas(y[a], ALPHA))
            p.fit(df.iloc[a], y[a])
            oof[b] = p.predict(df.iloc[b])
            dim = max(dim, p.named_steps["clf"].coef_.shape[1])
        A.append(accuracy_score(y, oof))
        F.append(f1_score(y, oof, average="macro", zero_division=0))
    return float(np.mean(A)), float(np.mean(F)), dim


def main() -> int:
    from train import kolom_emosi, kolom_konsep

    df = muat()
    kon, emo = kolom_konsep(df), kolom_emosi(df)
    y = df["emotion"].to_numpy()
    g = df["key"].to_numpy()

    kv = vek_kata().fit(df["text_combined"].fillna(""))
    ks = vek_kata().fit(df["text_stem"].fillna(""))
    print(f"\nkosakata TF-IDF: asli {len(kv.vocabulary_)} -> "
          f"setelah stemming {len(ks.vocabulary_)} "
          f"({100*(1-len(ks.vocabulary_)/len(kv.vocabulary_)):.1f}% menyusut)")

    konfig = [
        ("asli (model final)", "text_combined", False),
        ("stemming", "text_stem", False),
        ("char n-gram 3-5", "text_combined", True),
        ("stemming + char n-gram", "text_stem", True),
    ]

    print(f"\n{'konfigurasi':26} {'dim':>7} {'akurasi':>9} {'macro-F1':>10}")
    print("-" * 56)
    hasil = []
    for nama, kol, char in konfig:
        a, f, d = uji(df, bikin(kol, char, kon, emo), y, g)
        hasil.append({"konfigurasi": nama, "dim": d, "accuracy": a, "macro_f1": f})
        print(f"{nama:26} {d:7d} {100*a:8.1f}% {f:10.3f}", flush=True)
    print("-" * 56)
    print(f"{len(SEEDS)} seed, mean. Acuan baseline: 41,2% / 0,058")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(hasil).to_csv(OUTPUTS_DIR / "uji_stemming.csv", index=False)
    print(f"disimpan: {OUTPUTS_DIR/'uji_stemming.csv'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
