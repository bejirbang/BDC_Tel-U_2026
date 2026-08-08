"""
Transkripsi audio video memakai Whisper (faster-whisper / CTranslate2).

Tahap ini dipisah dari ekstraksi fitur karena sifatnya sama dengan akuisisi:
mahal, sekali jalan, dan hasilnya di-cache. `features_audio.py` nanti tinggal
membaca cache-nya.

Kenapa ini prioritas tertinggi setelah metadata+teks: soal CAKUPAN, bukan
kecanggihan. Caption hanya ada di 62,6% baris train, sedangkan video ada di
88,3%. Transkrip menutup ~26% baris yang punya video tapi tidak punya caption -
lompatan cakupan terbesar yang tersisa.

Bahasa sengaja dideteksi otomatis, tidak dipaksa "id". Selain lebih aman untuk
konten campur, bahasa terdeteksi + keyakinannya ikut jadi fitur.

Transkrip kosong BUKAN kegagalan: banyak reel hanya berisi musik tanpa ucapan.
Itu justru sinyal (`has_speech`), dan dicatat sebagai hasil yang sah.

Jalankan:
    python src/transcribe.py --model small        # bawaan
    python src/transcribe.py --limit 20           # uji coba dulu
    python src/transcribe.py --device cpu         # kalau VRAM bermasalah
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import CACHE_DIR, VIDEO_DIR  # noqa: E402

TRANSCRIPT_DIR = CACHE_DIR / "transcript"


def daftar_video() -> list[Path]:
    return sorted(VIDEO_DIR.glob("*.mp4"))


def sudah_ada(key: str) -> bool:
    return (TRANSCRIPT_DIR / f"{key}.json").exists()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Transkripsi Whisper untuk BDC Tel-U 2026")
    ap.add_argument("--model", default="small",
                    help="tiny | base | small | medium (bawaan: small)")
    ap.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    ap.add_argument("--compute-type", default=None,
                    help="bawaan: float16 di cuda, int8 di cpu")
    ap.add_argument("--limit", type=int, default=0, help="proses N file saja")
    ap.add_argument("--force", action="store_true", help="timpa transkrip yang sudah ada")
    args = ap.parse_args(argv)

    from faster_whisper import WhisperModel

    compute = args.compute_type or ("float16" if args.device == "cuda" else "int8")
    print(f"memuat Whisper '{args.model}' di {args.device} ({compute})...")
    model = WhisperModel(args.model, device=args.device, compute_type=compute)

    TRANSCRIPT_DIR.mkdir(parents=True, exist_ok=True)
    semua = daftar_video()
    todo = [p for p in semua if args.force or not sudah_ada(p.stem)]
    if args.limit:
        todo = todo[: args.limit]

    print(f"total video {len(semua)} | akan diproses {len(todo)} "
          f"(sisanya sudah ada di cache)")
    if not todo:
        return 0
    print("-" * 74)

    t0 = time.time()
    durasi_total = 0.0
    ok = kosong = gagal = 0
    try:
        for n, p in enumerate(todo, 1):
            try:
                segs, info = model.transcribe(
                    str(p),
                    beam_size=1,          # greedy: ~2x lebih cepat, beda akurasi kecil
                    vad_filter=True,      # lewati bagian tanpa suara
                    vad_parameters={"min_silence_duration_ms": 500},
                )
                potongan = [{"start": round(s.start, 2), "end": round(s.end, 2),
                             "text": s.text.strip()} for s in segs]
                teks = " ".join(s["text"] for s in potongan).strip()
                data = {
                    "key": p.stem,
                    "language": info.language,
                    "language_prob": round(float(info.language_probability), 4),
                    "duration": round(float(info.duration), 2),
                    "n_segments": len(potongan),
                    "text": teks,
                    "segments": potongan,
                }
                (TRANSCRIPT_DIR / f"{p.stem}.json").write_text(
                    json.dumps(data, ensure_ascii=False), encoding="utf-8")
                durasi_total += data["duration"]
                if teks:
                    ok += 1
                else:
                    kosong += 1
            except Exception as e:  # noqa: BLE001 - satu file rusak tidak boleh menghentikan batch
                gagal += 1
                print(f"[{n}/{len(todo)}] GAGAL {p.stem}: {type(e).__name__}: {str(e)[:70]}")
                continue

            if n % 25 == 0 or n == len(todo):
                el = time.time() - t0
                laju = durasi_total / el if el else 0
                sisa = (len(todo) - n) * (el / n) / 60
                print(f"[{n}/{len(todo)}] ada-ucapan {ok} | tanpa-ucapan {kosong} | "
                      f"gagal {gagal} | {laju:.1f}x realtime | sisa ~{sisa:.0f} menit")
    except KeyboardInterrupt:
        print("\ndihentikan pengguna - transkrip yang sudah jadi tetap tersimpan")

    print("-" * 74)
    print(f"selesai: {ok} ada ucapan, {kosong} tanpa ucapan, {gagal} gagal")
    print(f"cache: {TRANSCRIPT_DIR}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
