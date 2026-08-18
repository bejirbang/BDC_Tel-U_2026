"""
Pelatihan dan validasi silang, disusun sebagai tangga bertahap.

Tiap tahap menambah satu kelompok fitur di atas tahap sebelumnya, sehingga
kenaikan skor bisa ditelusuri ke penyebabnya. Tabel hasilnya langsung jadi
tabel ablation untuk laporan.

Dua keputusan yang menentukan kebenaran angkanya:

1. **StratifiedGroupKFold dengan `groups=key`.** Ada 45 baris yang URL-nya
   duplikat setelah normalisasi. Tanpa pengelompokan, baris kembar bisa jatuh
   di train dan validasi sekaligus - model tinggal menghafal dan skor CV
   jadi bohong.

2. **TF-IDF dan target encoding dipasang DI DALAM Pipeline.** Keduanya di-fit
   ulang pada tiap fold. Kalau di-fit sekali di luar pada seluruh data train,
   kosakata dan statistik label dari fold validasi ikut membentuk fitur -
   skor jadi optimistis palsu.

Jalankan:  python src/train.py
"""

from __future__ import annotations

import json
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.base import BaseEstimator, ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import TruncatedSVD
from sklearn.dummy import DummyClassifier
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, f1_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.naive_bayes import ComplementNB
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, TargetEncoder

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES_DIR, OUTPUTS_DIR, SEED  # noqa: E402

warnings.filterwarnings("ignore", category=UserWarning)

N_SPLITS = 5
KUNCI = ["split", "id", "key", "emotion"]
TEKS = ["text_caption", "text_all", "text_transcript", "text_combined", "text_stem"]
UPLOADER = ["uploader", "uploader_id"]

# Caption + transkrip digabung: cakupannya saling menutupi (63% vs 88%).
KOLOM_TEKS = "text_combined"

# Model final memakai teks yang sudah di-stem, dengan UNIGRAM saja (§21).
# Keduanya harus berpasangan: pada bigram, stemming tidak berpengaruh karena
# bigram jarang bertabrakan setelah di-stem (kosakata cuma menyusut 1,6% lawan
# 17,4% pada unigram). Dipisah begini supaya tangga ablation bisa menunjukkan
# sumbangan masing-masing.
KOLOM_TEKS_FINAL = "text_stem"
NGRAM_FINAL = (1, 1)


def kelompok_fitur(df: pd.DataFrame) -> dict[str, list[str]]:
    meta_cols = pd.read_parquet(FEATURES_DIR / "meta.parquet").columns
    text_cols = pd.read_parquet(FEATURES_DIR / "text.parquet").columns
    audio_path = FEATURES_DIR / "audio.parquet"
    audio_cols = pd.read_parquet(audio_path).columns if audio_path.exists() else []
    num = df.select_dtypes("number").columns
    return {
        "meta": [c for c in num if c in meta_cols and c not in KUNCI + UPLOADER],
        "text": [c for c in num if c in text_cols and c not in KUNCI],
        "audio": [c for c in num if c in audio_cols and c not in KUNCI],
    }


def bobot_kelas(y, alpha: float) -> dict | None:
    """Pembobotan kelas bertingkat.

    alpha=0 tanpa bobot, alpha=1 sama dengan 'balanced'. Nilai di antaranya
    memberi titik tukar-guling yang jauh lebih baik: pada alpha=1 akurasi jatuh
    di bawah baseline karena kelas bersampel 1 (Love, Loyalty) mendapat bobot
    ~80x dan menyeret model. alpha=0.5 menang di dua sisi sekaligus.

    Bobot dihitung dari `y` FOLD ITU SAJA - kelas yang tidak muncul di fold
    tidak boleh ikut, karena LightGBM menolak kunci kelas yang tak dikenalnya.
    """
    if alpha == 0:
        return None
    c = pd.Series(y).value_counts()
    return {k: (len(y) / (len(c) * n)) ** alpha for k, n in c.items()}


# Bobot fitur konsep relatif terhadap TF-IDF. Disapu 0,1-0,4; 0,25 terbaik.
# Terlalu besar membuat konsep menenggelamkan TF-IDF dan akurasi jatuh.
BOBOT_KONSEP = 0.25

# Bobot fitur emosi (§19). Disapu 0,10-0,50 pada 3 seed; 0,25 terbaik - di
# bawahnya sumbangannya hilang, di atasnya akurasi mulai jatuh.
BOBOT_EMOSI = 0.25


def kolom_konsep(df: pd.DataFrame) -> list[str]:
    """Hanya 9 konsep gelombang pertama (§17).

    Konsep gelombang kedua (§23) tetap dihitung dan tersimpan di dataset untuk
    analisis laporan, tapi TIDAK dipakai model: akurasinya kalah di 5 dari 5
    seed karena katanya sudah tercakup TF-IDF.
    """
    from features_text import KONSEP_FINAL
    return [c for c in df.columns if c in KONSEP_FINAL]


def kolom_emosi(df: pd.DataFrame) -> list[str]:
    """Probabilitas emosi dari model eksternal, dihitung pada teks padat.

    Hanya varian `_padat_` yang dipakai: varian `_penuh_` setara secara skor
    (§19) tapi teks padat lebih murah dan lebih mudah dijelaskan di laporan.
    """
    return [c for c in df.columns if c.startswith("emo_") and "_padat_" in c]


# Bobot ComplementNB dalam ensemble (§24). Disapu 0,3-0,7 pada 5 seed; 0,5 dan
# 0,6 setara dan keduanya menang macro-F1 di 5/5 seed serta akurasi di 4/5.
# Dipilih 0,5 - bobot sama rata, tanpa kesan ditala berlebihan.
BOBOT_NB = 0.5
ALPHA_NB = 0.3


class EnsembleLRNB(BaseEstimator, ClassifierMixin):
    """Rata-rata probabilitas Logistic Regression dan ComplementNB.

    Kenapa PASANGAN INI, bukan sekadar "dua model":

    Seluruh percobaan sebelumnya - fitur emosi (§19), stemming (§21), konsep
    gelombang kedua (§23), sapuan C, oversampling, kalibrasi tau - bergerak di
    **kurva tukar-guling yang sama**: menaikkan macro-F1 selalu menurunkan
    akurasi pada laju yang praktis tetap. Itu tanda kendalanya bukan letak titik
    operasi, melainkan kualitas estimasi probabilitasnya.

    LogReg diskriminatif; ComplementNB generatif dan dirancang khusus untuk teks
    tidak seimbang (ia memodelkan distribusi kata KOMPLEMEN tiap kelas, sehingga
    kelas kecil tidak tenggelam oleh kelas besar). Keduanya salah di tempat yang
    berbeda, jadi merata-ratakan probabilitasnya menurunkan ragam estimasi dan
    **menggeser seluruh kurva ke luar** - bukan memindahkan titik di atasnya.

    Terukur: menang macro-F1 di 5/5 seed dan akurasi di 4/5, satu-satunya
    perubahan sepanjang proyek yang memperbaiki KEDUA metrik secara konsisten.
    """

    def __init__(self, C: float = 3.0, alpha_nb: float = ALPHA_NB,
                 bobot_nb: float = BOBOT_NB, class_weight=None):
        self.C = C
        self.alpha_nb = alpha_nb
        self.bobot_nb = bobot_nb
        self.class_weight = class_weight

    def fit(self, X, y):
        self.lr_ = LogisticRegression(max_iter=4000, C=self.C, random_state=SEED,
                                      class_weight=self.class_weight).fit(X, y)
        # ComplementNB menolak nilai negatif; fitur kita (TF-IDF, hitungan
        # konsep, probabilitas emosi) semuanya non-negatif jadi aman.
        self.nb_ = ComplementNB(alpha=self.alpha_nb).fit(X, y)
        self.classes_ = self.lr_.classes_
        # urutan kelas kedua model belum tentu sama - dipetakan sekali di sini
        self._idx_nb = [list(self.nb_.classes_).index(c) for c in self.classes_]
        return self

    def predict_proba(self, X):
        p = self.lr_.predict_proba(X)
        q = self.nb_.predict_proba(X)[:, self._idx_nb]
        return (1.0 - self.bobot_nb) * p + self.bobot_nb * q

    def predict(self, X):
        return self.classes_[self.predict_proba(X).argmax(1)]


def buat_pipeline_teks(alpha_bobot: float, C: float = 3.0,
                       kolom_teks: str = KOLOM_TEKS,
                       kol_konsep: list[str] | None = None,
                       kol_emosi: list[str] | None = None,
                       ngram: tuple[int, int] = (1, 2),
                       ensemble: bool = False) -> Pipeline:
    """Model final: TF-IDF pada caption+transkrip + fitur konsep, ke LogReg.

    Terbukti mengalahkan LightGBM, ensemble teks+numerik, maupun IndoBERT
    (beku dan fine-tuned) - lihat RENCANA.md §13 dan §16.

    Fitur konsep menaikkan macro-F1 secara konsisten di tiga seed split yang
    berbeda (+0,035, +0,019, +0,042) dan menghidupkan kelas kecil yang tadinya
    ber-F1 nol. Kalau `kol_konsep` None, pipeline jadi teks-saja seperti versi
    sebelumnya - dipakai sebagai pembanding di tangga ablation.
    """
    if not kol_konsep and not kol_emosi:
        return Pipeline([
            ("tfidf", TfidfVectorizer(min_df=2, ngram_range=ngram, sublinear_tf=True,
                                      strip_accents="unicode")),
            ("clf", EnsembleLRNB(C=C) if ensemble
             else LogisticRegression(max_iter=3000, C=C, random_state=SEED)),
        ])

    bagian = [
        ("tfidf", TfidfVectorizer(min_df=2, ngram_range=ngram, sublinear_tf=True,
                                  strip_accents="unicode"), kolom_teks),
    ]
    if kol_konsep:
        # Diskalakan supaya sebanding dengan TF-IDF yang sudah L2-normalized.
        bagian.append(("konsep", FunctionTransformer(
            lambda X: np.asarray(X) * BOBOT_KONSEP,
            feature_names_out="one-to-one"), kol_konsep))
    if kol_emosi:
        bagian.append(("emosi", FunctionTransformer(
            lambda X: np.asarray(X) * BOBOT_EMOSI,
            feature_names_out="one-to-one"), kol_emosi))

    return Pipeline([
        ("fitur", ColumnTransformer(bagian)),
        ("clf", EnsembleLRNB(C=C) if ensemble
         else LogisticRegression(max_iter=3000, C=C, random_state=SEED)),
    ])


def buat_pipeline(kolom_num: list[str], pakai_tfidf: bool, pakai_uploader: bool,
                  dummy: bool = False, kolom_teks: str = KOLOM_TEKS) -> Pipeline:
    if dummy:
        return Pipeline([("clf", DummyClassifier(strategy="most_frequent"))])

    bagian = []
    if kolom_num:
        # LightGBM menangani NaN sendiri, jadi tidak perlu imputasi - dan itu
        # justru lebih baik: "tidak ada data" di sini memang informatif.
        bagian.append(("num", "passthrough", kolom_num))
    if pakai_tfidf:
        bagian.append((
            "txt",
            Pipeline([
                ("tfidf", TfidfVectorizer(min_df=3, ngram_range=(1, 2),
                                          sublinear_tf=True, strip_accents="unicode")),
                # 803 baris vs ribuan istilah - tanpa reduksi dimensi, pasti overfit.
                ("svd", TruncatedSVD(n_components=50, random_state=SEED)),
            ]),
            kolom_teks,
        ))
    if pakai_uploader:
        # TargetEncoder sklearn melakukan cross-fitting internal, jadi label
        # baris itu sendiri tidak dipakai untuk meng-encode dirinya.
        bagian.append(("upl", TargetEncoder(target_type="multiclass", random_state=SEED),
                       ["uploader"]))

    return Pipeline([
        ("fitur", ColumnTransformer(bagian, remainder="drop", sparse_threshold=0.0)),
        # `deterministic` + `force_row_wise` + n_jobs=1 wajib di sini: dengan
        # multi-thread, LightGBM menghasilkan skor yang bergeser antar-jalankan
        # (terpantau 29,6% vs 29,1% pada tahap yang sama). Tabel ablation yang
        # tidak bisa direproduksi tidak layak masuk laporan. Sedikit lebih lambat,
        # tapi angkanya jadi bisa dipertanggungjawabkan.
        ("clf", LGBMClassifier(
            n_estimators=400, learning_rate=0.05, num_leaves=15,
            min_child_samples=10, subsample=0.8, subsample_freq=1,
            colsample_bytree=0.8, reg_lambda=1.0,
            class_weight="balanced", random_state=SEED, verbose=-1,
            n_jobs=1, deterministic=True, force_row_wise=True,
        )),
    ])


def jalankan_cv(df: pd.DataFrame, pipe: Pipeline, nama: str,
                alpha_bobot: float | None = None, kolom_teks: str | None = None) -> dict:
    """Kalau `kolom_teks` diisi, pipeline dianggap model teks murni dan hanya
    menerima satu kolom Series - bukan seluruh DataFrame."""
    X = df.drop(columns=["emotion"])
    y = df["emotion"].to_numpy()
    grup = df["key"].to_numpy()

    oof = np.empty(len(df), dtype=object)
    cv = StratifiedGroupKFold(n_splits=N_SPLITS, shuffle=True, random_state=SEED)
    for tr, va in cv.split(X, y, groups=grup):
        p = buat_pipeline_dari(pipe)
        if alpha_bobot is not None:
            p.set_params(clf__class_weight=bobot_kelas(y[tr], alpha_bobot))
        if kolom_teks:
            # ColumnTransformer butuh DataFrame; pipeline teks-saja butuh Series.
            # Kolom yang tidak dipakai pipeline akan di-drop sendiri oleh
            # ColumnTransformer (remainder='drop'), jadi aman mengirim semuanya.
            pakai_konsep = any(s_ == "fitur" for s_, _ in p.steps)
            kol = ([kolom_teks] + kolom_konsep(df) + kolom_emosi(df)
                   if pakai_konsep else kolom_teks)
            Xtr, Xva = df[kol].iloc[tr], df[kol].iloc[va]
        else:
            Xtr, Xva = X.iloc[tr], X.iloc[va]
        p.fit(Xtr, y[tr])
        oof[va] = p.predict(Xva)

    return {
        "tahap": nama,
        "accuracy": accuracy_score(y, oof),
        "macro_f1": f1_score(y, oof, average="macro", zero_division=0),
        "weighted_f1": f1_score(y, oof, average="weighted", zero_division=0),
        "_oof": oof,
        "_y": y,
    }


def buat_pipeline_dari(pipe: Pipeline) -> Pipeline:
    """Salinan bersih supaya tiap fold mulai dari nol."""
    from sklearn.base import clone
    return clone(pipe)


def main() -> None:
    df = pd.read_parquet(FEATURES_DIR / "dataset.parquet")
    tr = df[df["split"] == "train"].reset_index(drop=True)
    grup = kelompok_fitur(tr)
    print(f"train {len(tr)} baris | fitur meta {len(grup['meta'])}, teks {len(grup['text'])}")
    print(f"CV: StratifiedGroupKFold {N_SPLITS} fold, grup = key ({tr['key'].nunique()} grup)\n")

    dasar = grup["meta"] + grup["text"]
    # (nama, kolom numerik, pakai tfidf, pakai uploader, dummy, kolom teks)
    tahapan = [
        ("1. baseline (kelas mayoritas)", [], False, False, True, "text_all"),
        ("2. + fitur metadata",           grup["meta"], False, False, False, "text_all"),
        ("3. + fitur teks (hitungan)",    dasar, False, False, False, "text_all"),
        ("4. + TF-IDF caption/komentar",  dasar, True, False, False, "text_all"),
        ("5. + uploader (target enc.)",   dasar, True, True, False, "text_all"),
    ]
    if grup["audio"]:
        tahapan += [
            # 6 dan 7 beda pada KOLOM TEKS yang divektorkan, bukan pada fitur
            # numeriknya - supaya sumbangan transkrip terisolasi dan terukur.
            ("6. + fitur audio (numerik)", dasar + grup["audio"], True, True, False, "text_all"),
            ("7. + transkrip ke TF-IDF",   dasar + grup["audio"], True, True, False, "text_combined"),
        ]

    hasil = []
    for nama, kol, tfidf, upl, dummy, kteks in tahapan:
        pipe = buat_pipeline(kol, tfidf, upl, dummy, kolom_teks=kteks)
        r = jalankan_cv(tr, pipe, nama)
        hasil.append(r)
        print(f"{nama:36} acc {r['accuracy']*100:5.1f}%   "
              f"macroF1 {r['macro_f1']:.3f}   weightedF1 {r['weighted_f1']:.3f}")

    # --- Jalur kedua: buang seluruh fitur numerik, TF-IDF langsung ke LogReg.
    #     Ternyata inilah yang terbaik - lihat RENCANA.md §13. ---
    print()
    for alpha in (0.0, 0.5, 1.0):
        nama = f"8. teks saja (TF-IDF+LogReg), bobot^{alpha}"
        r = jalankan_cv(tr, buat_pipeline_teks(alpha), nama,
                        alpha_bobot=alpha, kolom_teks=KOLOM_TEKS)
        hasil.append(r)
        print(f"{nama:36} acc {r['accuracy']*100:5.1f}%   "
              f"macroF1 {r['macro_f1']:.3f}   weightedF1 {r['weighted_f1']:.3f}")

    # --- Tahap 9: MODEL FINAL. Menambahkan 9 fitur konsep semantik di atas
    #     TF-IDF. Idenya datang dari menelaah video Sad yang isinya "mobil
    #     ditinggal, tangki gak kepake" - tidak ada kata emosi sama sekali,
    #     tapi idenya jelas. Lihat RENCANA.md §17. ---
    kk = kolom_konsep(tr)
    if kk:
        print()
        for alpha in (0.5,):
            nama = f"9. + fitur konsep semantik, bobot^{alpha}"
            r = jalankan_cv(tr, buat_pipeline_teks(alpha, kol_konsep=kk), nama,
                            alpha_bobot=alpha, kolom_teks=KOLOM_TEKS)
            hasil.append(r)
            print(f"{nama:36} acc {r['accuracy']*100:5.1f}%   "
                  f"macroF1 {r['macro_f1']:.3f}   weightedF1 {r['weighted_f1']:.3f}")

    # --- Tahap 10: lapisan emosi (§19). Probabilitas dari dua model emosi
    #     Bahasa Indonesia eksternal, dihitung pada teks yang sudah dipadatkan
    #     ke kalimat-kalimat paling emosional. Menang macro-F1 di 5 dari 5 seed.
    #     Kalibrasi prior (tau) dipasang di predict.py, bukan di sini, supaya
    #     tangga ini tetap mengukur sumbangan FITUR saja. ---
    ke = kolom_emosi(tr)
    if ke:
        nama = "10. + lapisan emosi"
        r = jalankan_cv(tr, buat_pipeline_teks(0.5, kol_konsep=kk, kol_emosi=ke),
                        nama, alpha_bobot=0.5, kolom_teks=KOLOM_TEKS)
        hasil.append(r)
        print(f"{nama:36} acc {r['accuracy']*100:5.1f}%   "
              f"macroF1 {r['macro_f1']:.3f}   weightedF1 {r['weighted_f1']:.3f}")

    # --- Tahap 11: MODEL FINAL. Stemming Sastrawi + unigram saja (§21).
    #     Keduanya harus berpasangan - stemming pada bigram tidak berpengaruh.
    #     Menang macro-F1 di 5 dari 5 seed. Kalibrasi prior dipasang di
    #     predict.py, jadi tangga ini mengukur sumbangan FITUR saja. ---
    if ke and KOLOM_TEKS_FINAL in tr.columns:
        nama = "11. + stemming, unigram"
        r = jalankan_cv(tr, buat_pipeline_teks(0.5, kol_konsep=kk, kol_emosi=ke,
                                               kolom_teks=KOLOM_TEKS_FINAL,
                                               ngram=NGRAM_FINAL),
                        nama, alpha_bobot=0.5, kolom_teks=KOLOM_TEKS_FINAL)
        hasil.append(r)
        print(f"{nama:36} acc {r['accuracy']*100:5.1f}%   "
              f"macroF1 {r['macro_f1']:.3f}   weightedF1 {r['weighted_f1']:.3f}")

        # --- Tahap 12: MODEL FINAL. Ensemble LogReg + ComplementNB (§24).
        #     Satu-satunya perubahan sepanjang proyek yang memperbaiki KEDUA
        #     metrik secara konsisten: macro-F1 menang 5/5 seed, akurasi 4/5. ---
        nama = "12. + ensemble LogReg+CNB (MODEL FINAL)"
        r = jalankan_cv(tr, buat_pipeline_teks(0.5, kol_konsep=kk, kol_emosi=ke,
                                               kolom_teks=KOLOM_TEKS_FINAL,
                                               ngram=NGRAM_FINAL, ensemble=True),
                        nama, alpha_bobot=0.5, kolom_teks=KOLOM_TEKS_FINAL)
        hasil.append(r)
        print(f"{nama:36} acc {r['accuracy']*100:5.1f}%   "
              f"macroF1 {r['macro_f1']:.3f}   weightedF1 {r['weighted_f1']:.3f}")

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    tabel = pd.DataFrame([{k: v for k, v in r.items() if not k.startswith("_")}
                          for r in hasil])
    tabel.to_csv(OUTPUTS_DIR / "ablation.csv", index=False)

    terbaik = max(hasil, key=lambda r: r["macro_f1"])
    print(f"\nterbaik menurut macro-F1: {terbaik['tahap']}")
    print("\nlaporan per kelas (tahap terbaik):")
    print(classification_report(terbaik["_y"], terbaik["_oof"], zero_division=0))

    pd.DataFrame({"id": tr["id"], "y": terbaik["_y"], "oof": terbaik["_oof"]}) \
        .to_csv(OUTPUTS_DIR / "oof_terbaik.csv", index=False)

    # `oof_terbaik.csv` adalah tahap dengan macro-F1 tertinggi di tangga ini,
    # yang BELUM TENTU model final - tahap 8 sering menang di sini karena
    # bobot^1,0 mengorbankan akurasi. OOF model final disimpan terpisah supaya
    # analisis galat di laporan memakai model yang benar-benar dikirim.
    final = next((r for r in hasil if "MODEL FINAL" in r["tahap"]), None)
    if final is not None:
        pd.DataFrame({"id": tr["id"], "y": final["_y"], "oof": final["_oof"]}) \
            .to_csv(OUTPUTS_DIR / "oof_final.csv", index=False)
        print(f"OOF model final -> {OUTPUTS_DIR / 'oof_final.csv'}")
    (OUTPUTS_DIR / "tahap_terbaik.json").write_text(
        json.dumps({"tahap": terbaik["tahap"], "macro_f1": terbaik["macro_f1"],
                    "accuracy": terbaik["accuracy"]}, indent=2), encoding="utf-8")
    print(f"\ndisimpan: {OUTPUTS_DIR/'ablation.csv'}, {OUTPUTS_DIR/'oof_terbaik.csv'}")


if __name__ == "__main__":
    main()
