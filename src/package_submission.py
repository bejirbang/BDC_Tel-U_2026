"""
Menyiapkan paket kiriman untuk panitia.

Kuncinya: **video tidak ikut dikirim.** Model final hanya memakai teks, jadi
yang dibutuhkan untuk mereproduksi hasil cuma metadata dan transkrip - sekitar
8 MB, bukan 5,8 GB. Video hanya diperlukan kalau seseorang ingin menghitung
ulang durasi/resolusi, dan nilainya sudah tersimpan di `probe_video.json`.

Skrip ini menyalin cache ringan dari lokasi kerja (yang mungkin di luar repo
lewat `cache_dir.txt`) ke `data/cache/` di dalam repo, supaya paket bisa
dijalankan apa adanya tanpa konfigurasi tambahan.

Jalankan:  python src/package_submission.py
"""

from __future__ import annotations

import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, ROOT  # noqa: E402

TUJUAN = ROOT / "data" / "cache"
# Yang WAJIB ikut supaya `run_all.py` jalan tanpa internet.
ISI = ["meta", "transcript", "manifest.csv", "probe_video.json"]


def ukuran(p: Path) -> float:
    if p.is_file():
        return p.stat().st_size / 1e6
    return sum(f.stat().st_size for f in p.rglob("*") if f.is_file()) / 1e6


def main() -> int:
    if CACHE_DIR.resolve() == TUJUAN.resolve():
        print(f"cache sudah berada di dalam repo ({TUJUAN}) - tidak perlu disalin")
    else:
        print(f"menyalin cache ringan\n  dari : {CACHE_DIR}\n  ke   : {TUJUAN}")
        TUJUAN.mkdir(parents=True, exist_ok=True)
        for nama in ISI:
            src = CACHE_DIR / nama
            if not src.exists():
                print(f"  ! {nama} tidak ada - dilewati")
                continue
            dst = TUJUAN / nama
            if src.is_dir():
                shutil.copytree(src, dst, dirs_exist_ok=True)
            else:
                shutil.copy2(src, dst)
            print(f"  + {nama:18} {ukuran(dst):7.2f} MB")

    total = ukuran(TUJUAN)
    print(f"\ntotal cache dalam repo: {total:.1f} MB")

    print("\nyang TIDAK ikut dikirim:")
    vid = CACHE_DIR / "video"
    if vid.exists():
        print(f"  video/  {ukuran(vid) / 1000:.2f} GB - tidak dibutuhkan model final;"
              f" durasi & resolusi sudah tersimpan di probe_video.json")

    print("\nlangkah terakhir sebelum mengirim:")
    print("  1. hapus cache_dir.txt (supaya paket memakai data/cache bawaan)")
    print("  2. python run_all.py        # uji ulang dari nol")
    print("  3. periksa outputs/submission.csv berisi 200 baris")
    return 0


if __name__ == "__main__":
    sys.exit(main())
