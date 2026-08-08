"""
Embedding IndoBERT sebagai alternatif TF-IDF.

Dipakai dalam mode **beku** (frozen): IndoBERT hanya menghasilkan angka,
classifier-nya tetap Logistic Regression. Ini sengaja dipilih sebagai langkah
pertama - kalau langsung fine-tune, kegagalan karena overfitting akan tercampur
dengan pertanyaan sebenarnya, yaitu apakah representasi IndoBERT memang membawa
sinyal lebih banyak daripada hitungan kata.

Dua keputusan teknis yang perlu dicatat di laporan:

**1. Chunking, bukan pemotongan.** Batas BERT 512 token, sementara 22,5% teks
gabungan (caption + transkrip) melebihi itu - sampai 2.632 token. Memotong
begitu saja membuang seperlima isi. Di sini teks dipecah jadi potongan 512
token lalu embedding-nya dirata-rata, sehingga seluruh isi ikut terwakili.

**2. Mean pooling, bukan token [CLS].** Untuk model beku (tanpa fine-tuning),
[CLS] belum dilatih menjadi ringkasan kalimat pada tugas ini, sedangkan
rata-rata seluruh token yang bermakna secara empiris lebih stabil.

Jalankan:  python src/features_bert.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES_DIR, load_rows  # noqa: E402

MODEL = "indobenchmark/indobert-base-p1"
MAX_LEN = 512
STRIDE = 128          # potongan saling tumpang tindih supaya konteks di batas tidak hilang
BATCH = 16
CACHE = FEATURES_DIR / "bert_embeddings.npz"


def potong(ids: list[int], maks: int, stride: int) -> list[list[int]]:
    """Pecah daftar token jadi potongan yang saling tumpang tindih."""
    if len(ids) <= maks:
        return [ids]
    langkah = maks - stride
    return [ids[i:i + maks] for i in range(0, len(ids), langkah)
            if i == 0 or len(ids[i:i + maks]) > stride // 2]


def encode(teks: list[str], device: str = "cuda") -> np.ndarray:
    import torch
    from transformers import AutoModel, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModel.from_pretrained(MODEL).to(device).eval()
    if device == "cuda":
        model = model.half()

    # Satukan seluruh potongan dari semua dokumen supaya batch-nya padat,
    # lalu kembalikan ke dokumen asalnya setelah selesai.
    potongan, pemilik = [], []
    for i, t in enumerate(teks):
        ids = tok.encode(t or "", add_special_tokens=False)
        if not ids:
            potongan.append([tok.cls_token_id, tok.sep_token_id]); pemilik.append(i)
            continue
        for p in potong(ids, MAX_LEN - 2, STRIDE):
            potongan.append([tok.cls_token_id] + p + [tok.sep_token_id])
            pemilik.append(i)

    print(f"{len(teks)} dokumen -> {len(potongan)} potongan "
          f"({len(potongan)/len(teks):.2f} potongan/dokumen)")

    hasil = np.zeros((len(potongan), model.config.hidden_size), dtype=np.float32)
    t0 = time.time()
    with torch.no_grad():
        for a in range(0, len(potongan), BATCH):
            b = potongan[a:a + BATCH]
            n = max(len(x) for x in b)
            ids = torch.full((len(b), n), tok.pad_token_id, dtype=torch.long)
            mask = torch.zeros((len(b), n), dtype=torch.long)
            for j, x in enumerate(b):
                ids[j, :len(x)] = torch.tensor(x); mask[j, :len(x)] = 1
            out = model(input_ids=ids.to(device),
                        attention_mask=mask.to(device)).last_hidden_state
            # mean pooling dengan bobot attention mask - token padding diabaikan
            m = mask.to(device).unsqueeze(-1).to(out.dtype)
            vek = (out * m).sum(1) / m.sum(1).clamp(min=1)
            hasil[a:a + len(b)] = vek.float().cpu().numpy()
            if (a // BATCH) % 20 == 0:
                lewat = time.time() - t0
                print(f"  {a + len(b)}/{len(potongan)}  "
                      f"sisa ~{(len(potongan) - a) * lewat / max(a + len(b), 1) / 60:.1f} menit")

    # rata-ratakan potongan kembali ke dokumen asalnya
    emb = np.zeros((len(teks), hasil.shape[1]), dtype=np.float32)
    pemilik = np.array(pemilik)
    for i in range(len(teks)):
        emb[i] = hasil[pemilik == i].mean(0)
    return emb


def build(force: bool = False) -> tuple[np.ndarray, pd.DataFrame]:
    rows = load_rows()
    ds = pd.read_parquet(FEATURES_DIR / "dataset.parquet")
    teks = ds["text_combined"].fillna("").tolist()

    if CACHE.exists() and not force:
        d = np.load(CACHE)
        if len(d["emb"]) == len(teks):
            print(f"memakai cache: {CACHE.name}")
            return d["emb"], ds[["split", "id", "key", "emotion"]]

    import torch
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"menghitung embedding IndoBERT di {dev} ...")
    emb = encode(teks, dev)
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(CACHE, emb=emb)
    print(f"disimpan: {CACHE} ({CACHE.stat().st_size/1e6:.1f} MB)")
    return emb, ds[["split", "id", "key", "emotion"]]


if __name__ == "__main__":
    emb, meta = build()
    print(f"\nembedding: {emb.shape}")
    print(f"norma rata-rata: {np.linalg.norm(emb, axis=1).mean():.2f}")
    kosong = (np.abs(emb).sum(1) == 0).sum()
    print(f"vektor nol: {kosong}")
