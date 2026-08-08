"""
Fitur metadata - tanpa menyentuh teks sama sekali.

Dua sumber:
  1. JSON hasil scraping : like, komentar, uploader, waktu posting
  2. File video langsung : durasi, resolusi, ada/tidaknya audio, bitrate

Sumber kedua penting karena `duration` hanya terisi di 12 dari 621 metadata
Instagram (field itu cuma diberikan kalau scraping dilakukan dalam keadaan
login). Membacanya dari file video jauh lebih murah daripada scraping ulang:
~30 detik untuk 842 file, offline, tanpa risiko rate limit - dan ini SATU-SATUNYA
cara untuk 221 video Google Drive yang metadata-nya kosong total.

Jalankan:  python src/features_meta.py
"""

from __future__ import annotations

import json
import re
import subprocess
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (  # noqa: E402
    CACHE_DIR, describe, load_meta, load_rows, save_features, video_path,
)

PROBE_CACHE = CACHE_DIR / "probe_video.json"

RE_DUR = re.compile(r"Duration: (\d+):(\d+):([\d.]+)")
RE_VID = re.compile(r"Video:.*?(\d{2,4})x(\d{2,4})")
RE_FPS = re.compile(r"([\d.]+) fps")
RE_BR = re.compile(r"bitrate: (\d+) kb/s")


def _ffmpeg() -> str:
    import imageio_ffmpeg
    return imageio_ffmpeg.get_ffmpeg_exe()


def probe_videos(keys: list[str], force: bool = False) -> dict[str, dict]:
    """Baca durasi/resolusi/audio dari tiap file video.

    Hasilnya di-cache ke disk karena ini murni fungsi dari isi file - tidak ada
    gunanya mengulang tiap kali pipeline dijalankan.
    """
    cache: dict[str, dict] = {}
    if PROBE_CACHE.exists() and not force:
        cache = json.loads(PROBE_CACHE.read_text(encoding="utf-8"))

    todo = [k for k in keys if k not in cache and video_path(k)]
    if todo:
        exe = _ffmpeg()
        print(f"membaca {len(todo)} file video dengan ffmpeg...")
        for i, k in enumerate(todo, 1):
            p = video_path(k)
            r = subprocess.run([exe, "-i", str(p)], capture_output=True, text=True,
                               encoding="utf-8", errors="replace")
            err = r.stderr or ""
            info: dict = {}
            if (m := RE_DUR.search(err)):
                info["duration"] = int(m[1]) * 3600 + int(m[2]) * 60 + float(m[3])
            if (m := RE_VID.search(err)):
                info["width"], info["height"] = int(m[1]), int(m[2])
            if (m := RE_FPS.search(err)):
                info["fps"] = float(m[1])
            if (m := RE_BR.search(err)):
                info["bitrate_kbps"] = int(m[1])
            info["has_audio"] = "Audio:" in err
            info["filesize_mb"] = round(p.stat().st_size / 1e6, 3)
            cache[k] = info
            if i % 100 == 0:
                print(f"  {i}/{len(todo)}")
        PROBE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        PROBE_CACHE.write_text(json.dumps(cache, ensure_ascii=False), encoding="utf-8")
    return cache


def build() -> pd.DataFrame:
    rows = load_rows()
    meta = load_meta()
    probe = probe_videos(rows["key"].unique().tolist())

    recs = []
    for r in rows.itertuples(index=False):
        m = meta.get(r.key, {})
        p = probe.get(r.key, {})
        cap = m.get("caption") or ""

        like = m.get("like_count")
        komen = m.get("comment_count")
        ts = m.get("timestamp")

        rec = {
            "split": r.split, "id": r.id, "key": r.key,
            "emotion": r.emotion,

            # --- ketersediaan: kekosongan itu sendiri informatif, dan
            #     sumber data (Instagram vs Drive) berkorelasi dengan konten ---
            "has_video": int(bool(p)),
            "has_meta": int(bool(m)),
            "has_caption": int(len(cap) > 0),
            "src_instagram": int(r.kind == "instagram"),
            "src_gdrive": int(r.kind == "gdrive"),

            # --- engagement ---
            "like_count": like,
            "comment_count": komen,
            "log_like": np.log1p(like) if like is not None else np.nan,
            "log_comment": np.log1p(komen) if komen is not None else np.nan,
            "like_per_comment": (like / komen) if (like and komen) else np.nan,

            # --- properti video ---
            "duration": p.get("duration"),
            "width": p.get("width"),
            "height": p.get("height"),
            "fps": p.get("fps"),
            "bitrate_kbps": p.get("bitrate_kbps"),
            "filesize_mb": p.get("filesize_mb"),
            "has_audio": int(p["has_audio"]) if "has_audio" in p else np.nan,
            "is_portrait": (int(p["height"] > p["width"])
                            if p.get("width") and p.get("height") else np.nan),
            "aspect_ratio": (p["width"] / p["height"]
                             if p.get("width") and p.get("height") else np.nan),

            # --- uploader: hipotesis prediktor terkuat (akun brand vs influencer).
            #     Disimpan MENTAH sebagai string. Target encoding-nya WAJIB
            #     dilakukan di dalam fold CV, bukan di sini - kalau di sini,
            #     label ikut bocor ke fitur dan skor CV jadi optimistis palsu. ---
            "uploader": m.get("uploader") or "",
            "uploader_id": str(m.get("uploader_id") or ""),
        }

        # --- waktu posting ---
        if ts:
            t = pd.to_datetime(ts, unit="s", utc=True).tz_convert("Asia/Jakarta")
            rec.update(
                post_hour=t.hour,
                post_dayofweek=t.dayofweek,
                post_month=t.month,
                post_is_weekend=int(t.dayofweek >= 5),
                post_age_days=(pd.Timestamp.now(tz="Asia/Jakarta") - t).days,
            )
        else:
            rec.update(post_hour=np.nan, post_dayofweek=np.nan, post_month=np.nan,
                       post_is_weekend=np.nan, post_age_days=np.nan)

        recs.append(rec)

    df = pd.DataFrame(recs)

    # Durasi punya sebaran ekor panjang (15 detik sampai >2 menit); versi log
    # dan bucket lebih mudah dipakai model berbasis pohon maupun linear.
    df["log_duration"] = np.log1p(df["duration"])
    df["dur_bucket"] = pd.cut(df["duration"], bins=[0, 15, 30, 60, 120, 1e9],
                              labels=[0, 1, 2, 3, 4]).astype("float")
    return df


if __name__ == "__main__":
    df = build()
    save_features(df, "meta")
    describe(df, "meta")

    print("\ncakupan fitur kunci:")
    for c in ("duration", "like_count", "uploader", "post_hour", "has_audio"):
        isi = df[c].replace("", np.nan).notna().mean() * 100
        print(f"  {c:16} terisi {isi:5.1f}%")
    print(f"\nuploader unik: {df.loc[df['uploader'] != '', 'uploader'].nunique()}")
