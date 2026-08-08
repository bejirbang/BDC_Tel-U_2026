"""
Hal-hal yang dipakai bersama oleh semua tahap pipeline.

Sengaja mengimpor dari `acquire` alih-alih menyalin ulang logikanya, supaya
normalisasi URL dan lokasi cache hanya punya SATU definisi. Kalau `classify()`
berubah, seluruh pipeline ikut berubah otomatis - tidak ada risiko fitur dan
akuisisi memakai kunci yang berbeda.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from acquire import (  # noqa: F401 - sengaja di-re-export
    CACHE_DIR, META_DIR, MANIFEST, ROOT, VIDEO_DIR, classify,
)

FEATURES_DIR = ROOT / "data" / "features"
OUTPUTS_DIR = ROOT / "outputs"
SEED = 42

EMOTIONS = [
    "Surprise", "Trust", "Proud", "Joy", "Anger",
    "Sad", "Fear", "Neutral", "Love", "Loyalty",
]


def load_rows() -> pd.DataFrame:
    """Gabungan datatrain + datatest, satu baris per baris CSV asli.

    Kolom: split, id, url, key, emotion (kosong untuk test).
    `key` adalah kunci ternormalisasi yang sama dengan yang dipakai cache,
    sehingga baris yang URL-nya duplikat akan berbagi key - ini yang nanti
    dipakai sebagai grup pada GroupKFold.
    """
    frames = []
    for split, fname in (("train", "datatrain.csv"), ("test", "datatest.csv")):
        path = ROOT / fname
        df = pd.read_csv(path, encoding="utf-8-sig", dtype={"id": str})
        df["split"] = split
        if "emotion" not in df.columns:
            df["emotion"] = pd.NA
        df = df.rename(columns={"video": "url"})
        df["key"] = df["url"].map(lambda u: classify(str(u))[1])
        df["kind"] = df["url"].map(lambda u: classify(str(u))[0])
        frames.append(df[["split", "id", "url", "key", "kind", "emotion"]])
    return pd.concat(frames, ignore_index=True)


def load_meta() -> dict[str, dict]:
    """Semua metadata hasil scraping, dikunci per `key`."""
    out = {}
    for f in META_DIR.glob("*.json"):
        try:
            d = json.loads(f.read_text(encoding="utf-8"))
        except Exception:
            continue
        if d.get("key"):
            out[d["key"]] = d
    return out


def video_path(key: str) -> Path | None:
    for ext in (".mp4", ".mkv", ".webm", ".mov"):
        p = VIDEO_DIR / f"{key}{ext}"
        if p.exists() and p.stat().st_size > 0:
            return p
    return None


def save_features(df: pd.DataFrame, name: str) -> Path:
    FEATURES_DIR.mkdir(parents=True, exist_ok=True)
    path = FEATURES_DIR / f"{name}.parquet"
    df.to_parquet(path, index=False)
    return path


def describe(df: pd.DataFrame, name: str) -> None:
    """Ringkasan singkat supaya tiap tahap kelihatan hasilnya saat dijalankan."""
    fitur = [c for c in df.columns if c not in ("split", "id", "key", "kind", "emotion")]
    print(f"\n{name}: {len(df)} baris x {len(fitur)} fitur -> {FEATURES_DIR / (name + '.parquet')}")
    kosong = df[fitur].isna().mean().sort_values(ascending=False)
    banyak = kosong[kosong > 0.5]
    if len(banyak):
        print(f"  fitur dengan >50% kosong ({len(banyak)}): {list(banyak.index)[:8]}")
    print(f"  rata-rata kekosongan: {df[fitur].isna().mean().mean() * 100:.1f}%")
