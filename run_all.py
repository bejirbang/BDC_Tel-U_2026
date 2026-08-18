"""
Menjalankan seluruh pipeline dari cache yang disertakan sampai submission.csv.

    python run_all.py

Tidak butuh internet dan tidak butuh GPU. Tahap yang mahal - mengunduh 842
video dan mentranskripsinya dengan Whisper - hasilnya sudah disertakan di
`data/cache/`, jadi yang dijalankan di sini hanya ekstraksi fitur dan pemodelan.

Untuk membangun ulang cache dari nol (butuh internet, GPU, dan berjam-jam):

    pip install yt-dlp gdown imageio-ffmpeg faster-whisper
    python src/acquire.py                 # ~4 jam, unduh 842 video (~6 GB)
    python src/transcribe.py --model small  # ~50 menit di RTX 4050

Perlu diketahui: sebagian tautan sumber sudah mati sejak dataset dibuat, jadi
membangun ulang dari nol kemungkinan menghasilkan cakupan yang LEBIH RENDAH
daripada cache yang disertakan. Rincian di RENCANA.md §11.
"""

from __future__ import annotations

import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent

TAHAP = [
    ("features_meta.py",  "fitur metadata (+ durasi & resolusi dari file video)"),
    ("features_text.py",  "fitur teks dari caption, hashtag, komentar"),
    ("features_audio.py", "fitur dari transkrip Whisper"),
    # Melewati diri sendiri kalau emosi.parquet sudah ada, sehingga tahap ini
    # tidak butuh internet saat cache-nya ikut dikirim.
    ("features_emosi.py", "lapisan emosi: leksikon InSet + model emosi eksternal"),
    ("build_dataset.py",  "menggabungkan seluruh tabel fitur"),
    ("train.py",          "validasi silang + tabel ablation"),
    ("predict.py",        "melatih model final -> outputs/submission.csv"),
]


def jalankan(skrip: str, judul: str) -> None:
    print("\n" + "=" * 72)
    print(f"  {skrip}  -  {judul}")
    print("=" * 72)
    t0 = time.time()
    r = subprocess.run([sys.executable, str(ROOT / "src" / skrip)], cwd=ROOT)
    if r.returncode != 0:
        print(f"\nGAGAL pada {skrip} (kode {r.returncode}). Pipeline dihentikan.")
        sys.exit(r.returncode)
    print(f"  selesai dalam {time.time() - t0:.1f} detik")


def main() -> int:
    print("BDC Tel-U 2026 - pipeline lengkap")

    # features_audio.py tetap jalan tanpa transkrip (semua fiturnya jadi kosong),
    # tapi lebih baik memberi peringatan jelas daripada diam-diam menghasilkan
    # model yang jauh lebih lemah.
    sys.path.insert(0, str(ROOT / "src"))
    from common import CACHE_DIR  # noqa: E402
    n_meta = len(list((CACHE_DIR / "meta").glob("*.json"))) if (CACHE_DIR / "meta").exists() else 0
    n_trans = (len(list((CACHE_DIR / "transcript").glob("*.json")))
               if (CACHE_DIR / "transcript").exists() else 0)
    print(f"cache: {CACHE_DIR}")
    print(f"  metadata  : {n_meta} berkas")
    print(f"  transkrip : {n_trans} berkas")
    if n_meta == 0:
        print("\nPERINGATAN: cache metadata kosong. Jalankan src/acquire.py dulu.")
        return 1
    if n_trans == 0:
        print("\nPERINGATAN: cache transkrip kosong - model akan jauh lebih lemah.")
        print("            Jalankan src/transcribe.py untuk hasil penuh.")

    t0 = time.time()
    for skrip, judul in TAHAP:
        jalankan(skrip, judul)

    print("\n" + "=" * 72)
    print(f"  SELESAI dalam {(time.time() - t0) / 60:.1f} menit")
    print(f"  submission : {ROOT / 'outputs' / 'submission.csv'}")
    print(f"  ablation   : {ROOT / 'outputs' / 'ablation.csv'}")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    sys.exit(main())
