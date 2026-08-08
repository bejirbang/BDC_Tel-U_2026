"""
Fitur dari transkrip Whisper.

Menghasilkan dua macam keluaran, sejalan dengan `features_text.py`:

1. **Fitur numerik** - ada/tidaknya ucapan, kecepatan bicara, rasio bagian
   berucap, bahasa terdeteksi. Semuanya bisa dijelaskan ke juri.
2. **Kolom teks `text_transcript`** untuk divektorkan di dalam Pipeline CV.

Perhatikan: transkrip kosong itu HASIL YANG SAH, bukan kegagalan. Banyak reel
hanya berisi musik dengan teks tertempel di layar. `has_speech` justru salah
satu fitur yang paling mungkin membedakan - konten berbicara (review, edukasi)
punya karakter berbeda dari konten musik (montase produk).

Prasyarat:  python src/transcribe.py
Jalankan :  python src/features_audio.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, describe, load_rows, save_features  # noqa: E402

TRANSCRIPT_DIR = CACHE_DIR / "transcript"


def muat_transkrip() -> dict[str, dict]:
    out = {}
    if not TRANSCRIPT_DIR.exists():
        return out
    for f in TRANSCRIPT_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("key"):
            out[d["key"]] = d
    return out


def build() -> pd.DataFrame:
    rows = load_rows()
    tr = muat_transkrip()

    recs = []
    for r in rows.itertuples(index=False):
        d = tr.get(r.key)
        rec = {"split": r.split, "id": r.id, "key": r.key, "emotion": r.emotion}

        if not d:
            rec.update(
                text_transcript="", has_transcript=0, has_speech=np.nan,
                trans_len=np.nan, trans_words=np.nan, trans_segments=np.nan,
                speech_ratio=np.nan, words_per_sec=np.nan, avg_seg_dur=np.nan,
                avg_seg_words=np.nan, lang_prob=np.nan, is_indonesian=np.nan,
            )
            recs.append(rec)
            continue

        teks = d.get("text") or ""
        segs = d.get("segments") or []
        dur = float(d.get("duration") or 0)
        kata = teks.split()
        durasi_ucap = sum(max(0.0, s["end"] - s["start"]) for s in segs)

        rec.update(
            text_transcript=teks,
            has_transcript=1,
            has_speech=int(bool(kata)),
            trans_len=len(teks),
            trans_words=len(kata),
            trans_segments=len(segs),
            # Rasio bagian berucap: montase musik mendekati 0, konten edukasi mendekati 1
            speech_ratio=(durasi_ucap / dur) if dur > 0 else np.nan,
            # Kecepatan bicara - proksi intensitas/energi penyampaian
            words_per_sec=(len(kata) / durasi_ucap) if durasi_ucap > 0 else np.nan,
            avg_seg_dur=(durasi_ucap / len(segs)) if segs else np.nan,
            avg_seg_words=(len(kata) / len(segs)) if segs else np.nan,
            lang_prob=d.get("language_prob"),
            is_indonesian=int(d.get("language") == "id"),
        )
        recs.append(rec)

    return pd.DataFrame(recs)


if __name__ == "__main__":
    df = build()
    save_features(df, "audio")
    describe(df, "audio")

    ada = df["has_transcript"] == 1
    bicara = df["has_speech"] == 1
    print(f"\npunya transkrip : {ada.sum()} / {len(df)} ({ada.mean() * 100:.1f}%)")
    print(f"ada ucapan      : {bicara.sum()} ({bicara.mean() * 100:.1f}%)")
    print(f"musik saja      : {(ada & ~bicara).sum()}")

    tr = df[(df["split"] == "train") & bicara]
    if len(tr) > 20:
        print("\nrata-rata per emosi (baris yang ada ucapan):")
        kol = ["trans_words", "words_per_sec", "speech_ratio", "trans_segments"]
        ring = tr.groupby("emotion")[kol].mean().round(2)
        ring["n"] = tr.groupby("emotion").size()
        print(ring.sort_values("n", ascending=False).to_string())
