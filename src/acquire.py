"""
Akuisisi konten untuk BDC Tel-U 2026.

Mengambil metadata dan video dari URL di datatrain.csv / datatest.csv, lalu
menyimpannya ke data/cache/ supaya pipeline berikutnya bisa jalan tanpa internet.

Dijalankan berulang kali dengan aman: item yang sudah berhasil akan dilewati.

Contoh pemakaian
----------------
    # Rekonesans: hitung sebaran URL, tidak ada request jaringan
    python src/acquire.py --dry-run

    # Tahap 1 - metadata saja, cepat, untuk mengukur berapa URL yang masih hidup
    python src/acquire.py --meta-only

    # Tahap 2 - unduh video
    python src/acquire.py

    # Instagram sering minta login. Pakai cookies dari browser:
    python src/acquire.py --cookies-from-browser chrome

    # Ulangi hanya yang gagal
    python src/acquire.py --retry-failed
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import os
import random
import re
import sys
import time
from dataclasses import dataclass, asdict
from pathlib import Path

import requests

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"

def _resolve_cache_dir() -> Path:
    """Lokasi cache, berurutan:

    1. env var BDC_CACHE_DIR
    2. isi file `cache_dir.txt` di root repo (tidak masuk git)
    3. bawaan: data/cache di dalam repo

    Opsi 1 dan 2 ada supaya cache bisa ditaruh di luar folder OneDrive/Dropbox
    - video 6 GB tidak perlu ikut tersinkronisasi ke cloud. Bawaannya tetap
    di dalam repo supaya paket yang dikirim ke panitia jalan apa adanya.
    """
    env = os.environ.get("BDC_CACHE_DIR")
    if env:
        return Path(env)
    marker = ROOT / "cache_dir.txt"
    if marker.exists():
        loc = marker.read_text(encoding="utf-8").strip()
        if loc:
            return Path(loc)
    return ROOT / "data" / "cache"


CACHE_DIR = _resolve_cache_dir()
VIDEO_DIR = CACHE_DIR / "video"
META_DIR = CACHE_DIR / "meta"
MANIFEST = CACHE_DIR / "manifest.csv"

# Sumber CSV: pakai data/raw kalau ada, kalau tidak jatuh ke root repo.
CSV_CANDIDATES = [RAW_DIR, ROOT]

UA = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/126.0 Safari/537.36"
)


def _ffmpeg_path() -> str | None:
    """ffmpeg portabel dari paket pip imageio-ffmpeg, supaya tidak perlu
    instalasi sistem.

    Kembalikan path LENGKAP ke exe, bukan direktorinya: binary imageio bernama
    'ffmpeg-win-x86_64-v7.1.exe', jadi kalau yang diberikan direktori, yt-dlp
    mencari 'ffmpeg.exe', tidak menemukannya, lalu DIAM-DIAM jatuh ke format
    progresif kualitas penuh - file jadi ~8x lebih besar tanpa pesan error.
    """
    try:
        import imageio_ffmpeg
        return imageio_ffmpeg.get_ffmpeg_exe()
    except Exception:
        return None


FFMPEG = _ffmpeg_path()

MANIFEST_FIELDS = [
    "key", "kind", "url", "status", "video_path", "meta_path",
    "duration", "like_count", "comment_count", "view_count",
    "caption_len", "error", "fetched_at",
]


# --------------------------------------------------------------------------
# Normalisasi URL
# --------------------------------------------------------------------------

# Bentuk yang muncul di dataset:
#   instagram.com/reel/SHORTCODE/...
#   instagram.com/USERNAME/reel/SHORTCODE/...
RE_IG = re.compile(
    r"instagram\.com/(?:[A-Za-z0-9_.]+/)?(?:reel|reels|p|tv)/([A-Za-z0-9_-]+)"
)
RE_GD = re.compile(r"drive\.google\.com/(?:file/d/|open\?id=|uc\?id=)([A-Za-z0-9_-]+)")
RE_GD_FOLDER = re.compile(r"drive\.google\.com/drive/(?:u/\d+/)?folders/([A-Za-z0-9_-]+)")


def classify(url: str) -> tuple[str, str]:
    """Kembalikan (kind, key). Key sengaja dinormalisasi supaya URL yang hanya
    beda parameter tracking (igsh, utm_source) atau beda prefiks username
    dianggap sebagai satu item."""
    u = url.strip()

    m = RE_IG.search(u)
    if m:
        return "instagram", f"ig_{m.group(1)}"

    m = RE_GD.search(u)
    if m:
        return "gdrive", f"gd_{m.group(1)}"

    # Link folder, bukan file tunggal - tidak bisa diunduh otomatis, tangani manual.
    m = RE_GD_FOLDER.search(u)
    if m:
        return "gdrive_folder", f"gdf_{m.group(1)}"

    if "cdninstagram.com" in u or "fbcdn.net" in u:
        return "cdn", f"cdn_{hashlib.sha1(u.encode()).hexdigest()[:12]}"

    return "other", f"ot_{hashlib.sha1(u.encode()).hexdigest()[:12]}"


# --------------------------------------------------------------------------
# Muat daftar URL
# --------------------------------------------------------------------------

@dataclass
class Item:
    key: str
    kind: str
    url: str
    ids: list[str]          # "train:12", "test:5" - satu key bisa dipakai banyak baris


def find_csv(name: str) -> Path:
    for d in CSV_CANDIDATES:
        p = d / name
        if p.exists():
            return p
    raise FileNotFoundError(f"{name} tidak ditemukan di {[str(d) for d in CSV_CANDIDATES]}")


def load_items() -> list[Item]:
    by_key: dict[str, Item] = {}
    for split, fname in (("train", "datatrain.csv"), ("test", "datatest.csv")):
        path = find_csv(fname)
        with path.open(encoding="utf-8-sig", newline="") as fh:
            for row in csv.DictReader(fh):
                url = (row.get("video") or "").strip()
                if not url:
                    continue
                kind, key = classify(url)
                tag = f"{split}:{row.get('id', '?')}"
                if key in by_key:
                    by_key[key].ids.append(tag)
                else:
                    by_key[key] = Item(key=key, kind=kind, url=url, ids=[tag])
    return list(by_key.values())


# --------------------------------------------------------------------------
# Manifest
# --------------------------------------------------------------------------

def load_manifest() -> dict[str, dict]:
    if not MANIFEST.exists():
        return {}
    with MANIFEST.open(encoding="utf-8", newline="") as fh:
        return {r["key"]: r for r in csv.DictReader(fh)}


def save_manifest(rows: dict[str, dict]) -> None:
    MANIFEST.parent.mkdir(parents=True, exist_ok=True)
    tmp = MANIFEST.with_suffix(".tmp")
    with tmp.open("w", encoding="utf-8", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=MANIFEST_FIELDS, extrasaction="ignore")
        w.writeheader()
        for key in sorted(rows):
            w.writerow(rows[key])
    tmp.replace(MANIFEST)


def existing_video(key: str) -> Path | None:
    for p in VIDEO_DIR.glob(f"{key}.*"):
        if p.suffix.lower() in {".mp4", ".mkv", ".webm", ".mov"} and p.stat().st_size > 0:
            return p
    return None


# --------------------------------------------------------------------------
# Handler per sumber
# --------------------------------------------------------------------------

def relpath(p: Path) -> str:
    """Path relatif ke ROOT kalau memungkinkan, absolut kalau cache dipindah
    keluar repo lewat BDC_CACHE_DIR."""
    try:
        return str(p.relative_to(ROOT))
    except ValueError:
        return str(p)


def write_meta(key: str, payload: dict) -> Path:
    META_DIR.mkdir(parents=True, exist_ok=True)
    path = META_DIR / f"{key}.json"
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def fetch_ytdlp(item: Item, args) -> dict:
    """Instagram (dan URL lain yang didukung yt-dlp)."""
    import yt_dlp

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    h = args.max_height
    opts = {
        "outtmpl": str(VIDEO_DIR / f"{item.key}.%(ext)s"),
        # Instagram menyajikan DASH terpisah: video 360x640 / 720x1280 + audio m4a.
        # Gabungkan video + audio terbaik (butuh ffmpeg).
        #
        # Plafon tbr hanya dipasang pada bagian VIDEO (bv*), tidak pada audio:
        # sebagian reel tidak punya varian 360p sama sekali, dan di antara
        # varian 720p tie-break bawaan memilih bitrate TERTINGGI - ada yang
        # sampai 3 Mbps / 57 MB. Audio sengaja dibiarkan kualitas terbaik
        # karena jadi masukan Whisper.
        "format": (
            f"bv*[tbr<={args.max_vbr}]+ba/bv*+ba/b" if args.max_vbr else "bv*+ba/b"
        ),
        # Konten reel itu POTRET, jadi filter [height<=480] salah - varian 360p
        # punya height 640 dan lolos filter. `format_sort` dengan "res:N" memakai
        # sisi TERPENDEK dan memilih yang paling dekat ke N, jadi benar untuk
        # potret maupun lanskap.
        **({"format_sort": [f"res:{h}"]} if h else {}),
        "merge_output_format": "mp4",
        "ffmpeg_location": FFMPEG,
        "quiet": True,
        "no_warnings": True,
        "noprogress": True,
        "retries": 2,
        "socket_timeout": 30,
        "ignoreerrors": False,
        "http_headers": {"User-Agent": UA},
    }
    if args.cookies_from_browser:
        opts["cookiesfrombrowser"] = (args.cookies_from_browser,)
    if args.cookies_file:
        opts["cookiefile"] = args.cookies_file
    if args.comments:
        opts["getcomments"] = True

    download = not args.meta_only
    with yt_dlp.YoutubeDL(opts) as ydl:
        info = ydl.extract_info(item.url, download=download)

    if info.get("_type") == "playlist" and info.get("entries"):
        info = info["entries"][0]

    caption = info.get("description") or ""
    comments = [
        {"text": c.get("text"), "like_count": c.get("like_count")}
        for c in (info.get("comments") or [])[:50]
    ]
    payload = {
        "key": item.key,
        "kind": item.kind,
        "url": item.url,
        "source_ids": item.ids,
        "caption": caption,
        "hashtags": re.findall(r"#(\w+)", caption),
        "mentions": re.findall(r"@(\w+)", caption),
        "duration": info.get("duration"),
        "view_count": info.get("view_count"),
        "like_count": info.get("like_count"),
        "comment_count": info.get("comment_count"),
        "repost_count": info.get("repost_count"),
        "uploader": info.get("uploader"),
        "uploader_id": info.get("uploader_id"),
        "timestamp": info.get("timestamp"),
        "upload_date": info.get("upload_date"),
        "width": info.get("width"),
        "height": info.get("height"),
        "fps": info.get("fps"),
        "track": info.get("track"),
        "artist": info.get("artist"),
        "comments": comments,
    }
    meta_path = write_meta(item.key, payload)
    vid = existing_video(item.key)
    return {
        "status": "ok",
        "video_path": relpath(vid) if vid else "",
        "meta_path": relpath(meta_path),
        "duration": payload["duration"],
        "like_count": payload["like_count"],
        "comment_count": payload["comment_count"],
        "view_count": payload["view_count"],
        "caption_len": len(caption),
    }


RE_TITLE = re.compile(r"<title>(.*?)</title>", re.S | re.I)


def _gdrive_direct(file_id: str, dest: Path) -> bool:
    """Unduh lewat endpoint drive.usercontent.

    Jalur `gdown` biasa memakai endpoint yang cepat kena throttle
    ("Cannot retrieve the public link of the file"), sementara endpoint ini
    melayani file yang sama tanpa masalah. Dipakai sebagai jalur utama.
    """
    url = (f"https://drive.usercontent.google.com/download"
           f"?id={file_id}&export=download&confirm=t")
    with requests.get(url, headers={"User-Agent": UA}, timeout=90, stream=True) as r:
        if r.status_code != 200:
            return False
        # Google membalas halaman HTML (bukan 4xx) untuk file mati / minta izin,
        # jadi tipe konten harus diperiksa - kalau tidak, HTML tersimpan sebagai .mp4.
        if "text/html" in r.headers.get("Content-Type", ""):
            return False
        with dest.open("wb") as fh:
            for chunk in r.iter_content(1 << 16):
                fh.write(chunk)
    if dest.stat().st_size < 10_000:      # terlalu kecil untuk sebuah video
        dest.unlink(missing_ok=True)
        return False
    return True


def fetch_gdrive(item: Item, args) -> dict:
    """Google Drive. Metadata sangat terbatas - hanya nama file dari halaman view."""
    name = ""
    try:
        r = requests.get(item.url, headers={"User-Agent": UA}, timeout=30)
        m = RE_TITLE.search(r.text)
        if m:
            name = m.group(1).replace(" - Google Drive", "").strip()
    except requests.RequestException:
        pass  # metadata nama file sifatnya opsional

    # Sengaja TIDAK menggagalkan item hanya karena halaman bertuliskan
    # "Halaman Tidak Ditemukan": Google kadang menyajikannya untuk file yang
    # sebenarnya masih bisa diunduh. Biar percobaan unduh yang memutuskan.

    payload = {
        "key": item.key, "kind": item.kind, "url": item.url,
        "source_ids": item.ids, "filename": name,
        "caption": "", "hashtags": [], "mentions": [], "comments": [],
    }
    meta_path = write_meta(item.key, payload)
    out = {
        "status": "ok",
        "meta_path": relpath(meta_path),
        "video_path": "",
        "caption_len": 0,
    }

    if not args.meta_only:
        VIDEO_DIR.mkdir(parents=True, exist_ok=True)
        dest = VIDEO_DIR / f"{item.key}.mp4"
        file_id = item.key[3:]          # buang prefiks "gd_"

        if not _gdrive_direct(file_id, dest):
            # Cadangan: gdown. Endpoint-nya berbeda dan lebih mudah kena
            # throttle, tapi menangani sebagian kasus yang jalur langsung tolak.
            import gdown
            try:
                # gdown >= 5 menghapus parameter `fuzzy` (jadi perilaku bawaan);
                # memakainya pada 6.x melempar TypeError.
                got = gdown.download(item.url, str(dest), quiet=True)
            except Exception as e:
                raise RuntimeError(f"jalur langsung & gdown gagal: {e}") from e
            if not got or not dest.exists() or dest.stat().st_size == 0:
                raise RuntimeError("file mati / tidak publik")
        out["video_path"] = relpath(dest)
    return out


def fetch_direct(item: Item, args) -> dict:
    """Link CDN mentah - unduh langsung."""
    payload = {
        "key": item.key, "kind": item.kind, "url": item.url,
        "source_ids": item.ids,
        "caption": "", "hashtags": [], "mentions": [], "comments": [],
    }
    meta_path = write_meta(item.key, payload)
    out = {
        "status": "ok",
        "meta_path": relpath(meta_path),
        "video_path": "",
        "caption_len": 0,
    }
    if args.meta_only:
        r = requests.head(item.url, headers={"User-Agent": UA}, timeout=30, allow_redirects=True)
        if r.status_code >= 400:
            raise RuntimeError(f"HTTP {r.status_code}")
        return out

    VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    dest = VIDEO_DIR / f"{item.key}.mp4"
    with requests.get(item.url, headers={"User-Agent": UA}, timeout=60, stream=True) as r:
        r.raise_for_status()
        with dest.open("wb") as fh:
            for chunk in r.iter_content(1 << 16):
                fh.write(chunk)
    if dest.stat().st_size == 0:
        dest.unlink(missing_ok=True)
        raise RuntimeError("file kosong")
    out["video_path"] = relpath(dest)
    return out


def fetch_manual(item: Item, args) -> dict:
    """Link yang tidak bisa diotomatiskan (mis. folder Drive). Dicatat supaya
    terlihat di manifest dan bisa ditangani manual."""
    raise RuntimeError("perlu penanganan manual (link folder, bukan file video)")


HANDLERS = {
    "instagram": fetch_ytdlp,
    "gdrive": fetch_gdrive,
    "gdrive_folder": fetch_manual,
    "cdn": fetch_direct,
    "other": fetch_ytdlp,
}


# --------------------------------------------------------------------------
# Main
# --------------------------------------------------------------------------

def summarize(items: list[Item]) -> None:
    from collections import Counter
    kinds = Counter(i.kind for i in items)
    n_rows = sum(len(i.ids) for i in items)
    print(f"Total baris CSV     : {n_rows}")
    print(f"URL unik (ternormal): {len(items)}")
    print(f"Duplikat terlipat   : {n_rows - len(items)}")
    print("Sebaran sumber      :")
    for k, v in kinds.most_common():
        print(f"  {k:10s} {v:4d}")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="Akuisisi konten BDC Tel-U 2026")
    ap.add_argument("--dry-run", action="store_true", help="hanya tampilkan sebaran URL")
    ap.add_argument("--meta-only", action="store_true", help="ambil metadata saja, jangan unduh video")
    ap.add_argument("--kind", default="all",
                    choices=["all", "instagram", "gdrive", "gdrive_folder", "cdn", "other"])
    ap.add_argument("--limit", type=int, default=0, help="proses N item saja (0 = semua)")
    ap.add_argument("--max-height", type=int, default=480,
                    help="resolusi video yang dituju (sisi terpendek). 0 = kualitas penuh")
    ap.add_argument("--max-vbr", type=int, default=1000,
                    help="plafon bitrate video kbps. 0 = tanpa plafon")
    ap.add_argument("--sleep", type=float, default=3.0, help="jeda antar request, detik")
    ap.add_argument("--jitter", type=float, default=2.0, help="jitter acak ditambahkan ke jeda")
    ap.add_argument("--retry-failed", action="store_true", help="ulangi item yang statusnya gagal")
    ap.add_argument("--comments", action="store_true", help="ambil komentar (lebih lambat)")
    ap.add_argument("--cookies-from-browser", default=None,
                    help="chrome | firefox | edge - dipakai kalau Instagram minta login")
    ap.add_argument("--cookies-file", default=None, help="path cookies.txt")
    args = ap.parse_args(argv)

    items = load_items()
    if args.dry_run:
        summarize(items)
        return 0

    if args.kind != "all":
        items = [i for i in items if i.kind == args.kind]

    manifest = load_manifest()

    todo = []
    for it in items:
        prev = manifest.get(it.key)
        if prev and prev.get("status") == "ok":
            if args.meta_only:
                continue                      # metadata sudah tersimpan
            if existing_video(it.key):
                continue                      # video sudah ada di cache
            # status ok tapi video belum ada -> proses ulang untuk ambil videonya
        elif prev and prev.get("status") == "failed" and not args.retry_failed:
            continue
        todo.append(it)

    if args.limit:
        todo = todo[: args.limit]

    mode = "METADATA" if args.meta_only else "METADATA + VIDEO"
    print(f"Mode        : {mode}")
    print(f"Total unik  : {len(items)}")
    print(f"Akan diproses: {len(todo)}  (sisanya sudah ada di cache)")
    if not todo:
        return 0
    print("-" * 70)

    ok = fail = 0
    try:
        for n, it in enumerate(todo, 1):
            handler = HANDLERS[it.kind]
            row = {
                "key": it.key, "kind": it.kind, "url": it.url,
                "fetched_at": time.strftime("%Y-%m-%d %H:%M:%S"),
                "status": "", "video_path": "", "meta_path": "", "error": "",
                "duration": "", "like_count": "", "comment_count": "",
                "view_count": "", "caption_len": "",
            }
            try:
                row.update(handler(it, args))
                ok += 1
                tail = f"cap={row.get('caption_len', 0)} dur={row.get('duration') or '-'}"
                print(f"[{n}/{len(todo)}] OK   {it.key:24s} {tail}")
            except Exception as e:  # noqa: BLE001 - kegagalan per item tidak boleh menghentikan batch
                prev = manifest.get(it.key)
                if prev and prev.get("status") == "ok" and existing_video(it.key):
                    # Percobaan ulang gagal TAPI artefak lama masih utuh di disk.
                    # Jangan turunkan status jadi failed - tanpa penjagaan ini,
                    # sebuah retry yang gagal menghapus jejak unduhan yang sukses
                    # dan datanya terhitung hilang padahal filenya ada.
                    ok += 1
                    print(f"[{n}/{len(todo)}] SKIP {it.key:24s} retry gagal, pakai cache lama")
                    continue
                row["status"] = "failed"
                row["error"] = f"{type(e).__name__}: {e}"[:300]
                fail += 1
                print(f"[{n}/{len(todo)}] FAIL {it.key:24s} {row['error'][:80]}")

            manifest[it.key] = row
            if n % 10 == 0:
                save_manifest(manifest)
            if n < len(todo):
                time.sleep(args.sleep + random.uniform(0, args.jitter))
    except KeyboardInterrupt:
        print("\nDihentikan pengguna - menyimpan progres...")
    finally:
        save_manifest(manifest)

    total = ok + fail
    rate = (ok / total * 100) if total else 0.0
    print("-" * 70)
    print(f"Berhasil {ok} / {total}  ({rate:.1f}%)   gagal {fail}")
    print(f"Manifest: {MANIFEST}")
    if rate < 60 and total >= 20:
        print("\n!! Tingkat keberhasilan di bawah 60%.")
        print("   Lihat RENCANA.md §7 'Gate hari 1' - pertimbangkan --cookies-from-browser chrome")
    return 0


if __name__ == "__main__":
    sys.exit(main())
