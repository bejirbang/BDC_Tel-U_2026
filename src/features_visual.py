"""
Fitur visual CLIP - satu-satunya modalitas yang belum pernah dipakai.

842 video (12 GB) sudah diunduh sejak §11 dan tidak pernah masuk model. §14
membuang fitur visual, tapi yang dibuang itu **fitur numerik buatan tangan**:
cut rate, saturasi, histogram warna. CLIP jenisnya berbeda - ia semantik.

**Keputusan desain yang menentukan: BUKAN embedding mentah.**

Embedding CLIP 512 dimensi untuk 803 baris akan overfit, persis peringatan §5B
soal IndoBERT 768 dimensi. Yang dipakai di sini adalah **skor kemiripan
zero-shot terhadap prompt yang dirancang**, menghasilkan ~18 fitur yang bisa
dijelaskan satu per satu di laporan.

**Kenapa ini berpeluang padahal aturan yang sama gagal di teks (§23).**

Aturan pembeda manusia - Trust = mengajari/menjelaskan, Proud = pencapaian atau
spesifikasi, Surprise = produk baru dan kemegahan - diterjemahkan jadi fitur
teks dan DITOLAK, karena 17 dari 20 katanya sudah ada di kosakata TF-IDF.
Fiturnya cuma mengulang yang model sudah punya.

Di ranah visual redundansi itu **tidak ada**. Tidak ada satu pun fitur di model
sekarang yang tahu apakah layarnya menampilkan orang memegang piala, mobil di
atas panggung pameran, atau orang di gym. Aturan yang sama, ranah yang belum
tersentuh.

Prompt ditulis dalam Bahasa Inggris karena CLIP dilatih pada teks Inggris;
menerjemahkannya ke Bahasa Indonesia justru menurunkan kualitas kecocokan.

Jalankan:  python src/features_visual.py
"""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES_DIR, describe, save_features, video_path  # noqa: E402

MODEL = "openai/clip-vit-base-patch32"
N_FRAME = 8
BATCH = 64
CACHE_FRAME = FEATURES_DIR / "clip_frames.npz"

# Prompt dirancang dari aturan pembeda manusia (§23), plus beberapa penanda
# adegan umum. Tiap prompt jadi satu fitur - semuanya bisa dijelaskan.
PROMPT = {
    # Trust: mengajari / menjelaskan / menunjukkan cara
    "vis_mengajari": "a person demonstrating how to do something step by step",
    "vis_olahraga": "a person exercising in a gym with weights",
    "vis_menjelaskan": "a person talking to the camera explaining something",
    "vis_tangan_kerja": "close-up of hands working on or repairing something",

    # Proud: pencapaian / spesifikasi
    "vis_piala": "a person holding a trophy or medal at an award ceremony",
    "vis_podium": "athletes celebrating victory on a podium",
    "vis_spesifikasi": "a screen showing product specifications and numbers",
    "vis_seragam": "people wearing team uniforms or national colors",

    # Surprise: produk baru / kemegahan / pameran
    "vis_pameran": "a new car displayed on a stage at an auto show",
    "vis_mobil_mewah": "a luxury car exterior shot",
    "vis_produk_baru": "a new product being revealed or unboxed",
    "vis_harga": "a price tag or price displayed on screen",

    # penanda adegan umum
    "vis_wajah_dekat": "a close-up of a human face",
    "vis_banyak_orang": "a crowd of many people",
    "vis_makanan": "food being prepared or eaten",
    "vis_hewan": "a pet animal such as a cat, dog or fish",
    "vis_luar_ruang": "an outdoor landscape or street scene",
    "vis_teks_layar": "a screen with large text overlay",
}


def ambil_frame(path: Path, n: int = N_FRAME) -> list[np.ndarray]:
    """Ambil n keyframe merata. Frame merata lebih baik daripada n frame
    pertama: video Reel sering dibuka logo atau hook yang sama untuk semua
    konten, jadi awalnya justru paling tidak membedakan."""
    import cv2

    cap = cv2.VideoCapture(str(path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    keluar = []
    if total > 0:
        for i in np.linspace(0, total - 1, n).astype(int):
            cap.set(cv2.CAP_PROP_POS_FRAMES, int(i))
            ok, fr = cap.read()
            if ok:
                keluar.append(cv2.cvtColor(fr, cv2.COLOR_BGR2RGB))
    cap.release()
    return keluar


def build(force: bool = False) -> pd.DataFrame:
    import torch
    from PIL import Image
    from transformers import CLIPModel, CLIPProcessor

    ds = pd.read_parquet(FEATURES_DIR / "dataset.parquet")
    kunci = ds[["split", "id", "key", "emotion"]].copy()
    nama = list(PROMPT)
    teks = [PROMPT[k] for k in nama]

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    proc = CLIPProcessor.from_pretrained(MODEL)
    model = CLIPModel.from_pretrained(MODEL).to(dev).eval()

    def vektor(keluaran):
        """transformers 5.x mengembalikan objek, bukan tensor - ambil pooler-nya.
        Ditulis toleran supaya kode ini tetap jalan di kedua versi API."""
        return getattr(keluaran, "pooler_output", keluaran)

    # embedding teks dihitung sekali
    with torch.inference_mode():
        t = proc(text=teks, return_tensors="pt", padding=True).to(dev)
        emb_teks = vektor(model.get_text_features(**t))
        emb_teks = emb_teks / emb_teks.norm(dim=-1, keepdim=True)

    # satu key bisa dipakai beberapa baris (URL duplikat) - hitung sekali per key
    key_unik = kunci["key"].dropna().unique().tolist()
    hasil: dict[str, np.ndarray] = {}
    t0 = time.time()
    ada = 0

    for i, key in enumerate(key_unik):
        p = video_path(key)
        if p is None:
            continue
        frames = ambil_frame(p)
        if not frames:
            continue
        ada += 1
        with torch.inference_mode():
            im = proc(images=[Image.fromarray(f) for f in frames],
                      return_tensors="pt").to(dev)
            e = vektor(model.get_image_features(**im))
            e = e / e.norm(dim=-1, keepdim=True)
            sim = (e @ emb_teks.T).float().cpu().numpy()   # (n_frame, n_prompt)
        # Dua ringkasan per prompt: MAKS menangkap adegan puncak (satu frame
        # piala sudah cukup menandakan pencapaian), RATA-RATA menangkap nada
        # keseluruhan video.
        hasil[key] = np.concatenate([sim.max(0), sim.mean(0)])
        if i % 100 == 0:
            lewat = time.time() - t0
            print(f"  {i}/{len(key_unik)}  ({lewat/60:.1f} menit, "
                  f"sisa ~{(len(key_unik)-i)*lewat/max(i,1)/60:.0f} menit)", flush=True)

    print(f"video terproses: {ada}/{len(key_unik)} key unik")
    kolom = [f"{k}_maks" for k in nama] + [f"{k}_rata" for k in nama]
    M = np.full((len(kunci), len(kolom)), np.nan, dtype=np.float32)
    for j, key in enumerate(kunci["key"]):
        if key in hasil:
            M[j] = hasil[key]
    return pd.concat([kunci.reset_index(drop=True),
                      pd.DataFrame(M, columns=kolom)], axis=1)


if __name__ == "__main__":
    keluaran = FEATURES_DIR / "visual.parquet"
    if keluaran.exists() and "--force" not in sys.argv:
        print(f"memakai cache: {keluaran.name}. Pakai --force untuk membangun ulang.")
        sys.exit(0)
    df = build()
    save_features(df, "visual")
    describe(df, "visual")
