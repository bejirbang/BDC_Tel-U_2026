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
TEKS = ["text_caption", "text_all", "text_transcript", "text_combined"]


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
