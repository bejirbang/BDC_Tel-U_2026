"""
Menggabungkan tabel fitur jadi satu dataset siap latih.

Jalankan:  python src/build_dataset.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES_DIR, describe, save_features  # noqa: E402

KUNCI = ["split", "id", "key", "emotion"]
TEKS = ["text_caption", "text_all", "text_transcript", "text_combined", "text_stem"]


def stem_kolom(teks: pd.Series) -> pd.Series:
    """Stem tiap dokumen dengan Sastrawi.

    Cache dipasang per KATA, bukan per dokumen: 207 ribu token hanya berisi 41
    ribu kata unik, jadi memoisasi memangkas waktu dari menitan jadi ~9 detik.

    Kalau Sastrawi tidak terpasang, kolomnya diisi `text_combined` apa adanya
    supaya pipeline tetap jalan - hanya kehilangan sumbangan macro-F1 dari §21.
    """
    try:
        from Sastrawi.Stemmer.StemmerFactory import StemmerFactory
    except ImportError:
        print("Sastrawi tidak terpasang - text_stem diisi teks asli "
              "(pip install PySastrawi untuk mengaktifkan, lihat §21)")
        return teks.fillna("")

    stemmer = StemmerFactory().create_stemmer()
    memo: dict[str, str] = {}

    def satu(t: str) -> str:
        keluar = []
        for k in (t or "").lower().split():
            if k not in memo:
                memo[k] = stemmer.stem(k)
            keluar.append(memo[k])
        return " ".join(keluar)

    hasil = teks.fillna("").map(satu)
    print(f"stemming Sastrawi selesai ({len(memo)} kata unik di-cache)")
    return hasil


def build() -> pd.DataFrame:
    meta = pd.read_parquet(FEATURES_DIR / "meta.parquet")
    text = pd.read_parquet(FEATURES_DIR / "text.parquet")

    # Digabung lewat (split, id), bukan lewat `key`: satu key bisa dipakai
    # banyak baris CSV (URL duplikat), jadi merge pada key akan menggandakan baris.
    df = meta.merge(text.drop(columns=["key", "emotion"]), on=["split", "id"],
                    how="inner", validate="one_to_one")
    assert len(df) == len(meta) == len(text), "jumlah baris berubah saat merge"

    # Audio bersifat opsional supaya pipeline tetap jalan sebelum Whisper
    # selesai - berguna saat pengembangan, dan membuat paket tidak rusak kalau
    # panitia melewati tahap transkripsi.
    audio_path = FEATURES_DIR / "audio.parquet"
    if audio_path.exists():
        audio = pd.read_parquet(audio_path)
        df = df.merge(audio.drop(columns=["key", "emotion"]), on=["split", "id"],
                      how="left", validate="one_to_one")
        print(f"audio.parquet digabung ({audio['has_transcript'].sum()} transkrip)")
    else:
        print("audio.parquet belum ada - lewati (jalankan transcribe.py lalu features_audio.py)")
        df["text_transcript"] = ""
        df["has_transcript"] = 0

    # Caption dan transkrip disatukan untuk TF-IDF: keduanya sumber topik yang
    # sama-sama valid, dan cakupannya saling menutupi (caption 63%, transkrip 88%).
    df["text_combined"] = (
        df["text_all"].fillna("") + " " + df["text_transcript"].fillna("")
    ).str.strip()

    # Fitur konsep dihitung DI SINI, bukan di features_text.py, karena harus
    # membaca caption DAN transkrip sekaligus - dan `text_combined` baru ada
    # setelah keduanya digabung.
    from features_text import hitung_konsep
    konsep = pd.DataFrame([hitung_konsep(t) for t in df["text_combined"]],
                          index=df.index)
    df = pd.concat([df, konsep], axis=1)
    print(f"{konsep.shape[1]} fitur konsep ditambahkan "
          f"(rata-rata {konsep.sum(axis=1).mean():.2f} kemunculan per baris)")

    # Stemming Sastrawi (§21). Dipakai model final BERSAMA unigram-saja: pada
    # unigram, stemming menyusutkan kosakata 17,4% dan menaikkan macro-F1 di 5
    # dari 5 seed. Pada bigram efeknya hilang (kosakata cuma menyusut 1,6%),
    # karena bigram jarang bertabrakan setelah di-stem.
    df["text_stem"] = stem_kolom(df["text_combined"])

    # Fitur emosi (§19) bersifat opsional dengan alasan yang sama seperti audio:
    # membangunnya butuh dua model HuggingFace, jadi kalau `emosi.parquet` tidak
    # ikut dikirim dan panitia offline, pipeline harus tetap jalan - hanya saja
    # tanpa sumbangan macro-F1 dari lapisan emosi.
    emosi_path = FEATURES_DIR / "emosi.parquet"
    if emosi_path.exists():
        emosi = pd.read_parquet(emosi_path)
        df = df.merge(emosi.drop(columns=["key", "emotion"]), on=["split", "id"],
                      how="left", validate="one_to_one")
        n_emo = len([c for c in emosi.columns if c.startswith("emo_")])
        print(f"emosi.parquet digabung ({n_emo} probabilitas emosi + leksikon InSet)")
    else:
        print("emosi.parquet belum ada - lewati (jalankan features_emosi.py)")
    return df


def kolom_fitur(df: pd.DataFrame) -> list[str]:
    """Kolom yang boleh masuk model: semua kecuali kunci dan teks mentah."""
    return [c for c in df.columns if c not in KUNCI + TEKS]


if __name__ == "__main__":
    df = build()
    save_features(df, "dataset")
    describe(df, "dataset")

    tr = df[df["split"] == "train"]
    te = df[df["split"] == "test"]
    print(f"\ntrain {len(tr)} | test {len(te)}")
    print(f"grup unik (untuk GroupKFold): train {tr['key'].nunique()}, test {te['key'].nunique()}")

    kosong = tr[kolom_fitur(tr)].select_dtypes("number").isna().all(axis=1)
    print(f"baris train tanpa fitur numerik sama sekali: {kosong.sum()}")
    print(f"baris test  tanpa fitur numerik sama sekali: "
          f"{te[kolom_fitur(te)].select_dtypes('number').isna().all(axis=1).sum()}")
    print("\ndistribusi label train:")
    print(tr["emotion"].value_counts().to_string())
