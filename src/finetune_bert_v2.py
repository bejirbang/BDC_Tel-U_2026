"""
Fine-tuning IndoBERT, percobaan kedua - memakai resep yang terbukti di proyek
lain (klasifikasi tabel data governance: 347 sampel, 4 kelas, test acc 71%).

`finetune_bert.py` yang pertama kalah (37,4% / 0,129) dengan resep polos:
epoch tetap 4, tanpa validasi internal, tanpa augmentasi, padding penuh 512.
Empat hal dari resep pembanding itu belum pernah dicoba di sini, dan tiga di
antaranya justru menyerang persis penyakit dataset ini - 803 baris untuk 124
juta parameter:

1. **Augmentasi subsampling kalimat.** Di proyek pembanding, satu tabel
   dijadikan 5 varian dengan mengambil 50-100% atributnya secara acak; 347
   sampel jadi 1.337. Padanannya di sini: teks kita adalah caption + transkrip
   Whisper, yaitu deretan kalimat - jadi varian dibuat dengan mengambil 50-100%
   kalimat dalam urutan acak. Efek sampingnya kebetulan mengobati masalah lain:
   22,5% teks melebihi batas 512 token (lihat features_bert.py), dan varian
   yang lebih pendek membuat model melihat bagian transkrip yang berbeda-beda
   alih-alih selalu 512 token pertama.

2. **Early stopping pada macro-F1 validasi internal.** Epoch tetap 4 itu
   tebakan. Di sini tiap fold luar memotong lagi ~20% sebagai validasi dalam
   (tetap dikelompokkan per `key`), dan epoch terbaik dipilih dari situ. Fold
   luar tidak pernah dilihat sampai penilaian - jadi angkanya tetap jujur dan
   sebanding dengan seluruh tahap di train.py.

3. **`min_epochs` guard.** Dengan validasi dalam yang cuma ~100 baris untuk 10
   kelas, epoch awal bisa mendapat skor "beruntung". Pemilihan checkpoint baru
   aktif setelah epoch ke-2.

4. **Learning rate 1e-5 (dari 2e-5) + dynamic padding.** LR kecil untuk data
   kecil. Dynamic padding membuat tiap batch hanya selebar sekuens
   terpanjangnya, bukan selalu 512 - lebih cepat dan muat batch lebih besar.

Skema validasinya sengaja identik dengan train.py (StratifiedGroupKFold 5 fold,
`groups=key`, seed sama), supaya angkanya bisa ditaruh berdampingan di tabel
ablation tanpa catatan kaki.

Jalankan:  python src/finetune_bert_v2.py
           python src/finetune_bert_v2.py --n-augment 0   # isolasi efek augmentasi
"""

from __future__ import annotations

import argparse
import contextlib
import re
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES_DIR, OUTPUTS_DIR, SEED  # noqa: E402

MODEL = "indobenchmark/indobert-base-p1"
MAX_LEN = 512
BATCH = 16
EVAL_BATCH = 32
EPOCHS = 8
MIN_EPOCHS = 2        # skor epoch awal diabaikan - model belum matang
PATIENCE = 2
LR = 1e-5
WEIGHT_DECAY = 0.01
WARMUP_RATIO = 0.1
ALPHA_BOBOT = 0.5     # sama dengan model final di train.py
N_AUGMENT = 4         # jumlah varian tambahan per baris (train dalam saja)
INNER_SPLITS = 5      # 1 dari 5 jadi validasi dalam -> ~20%

# Pemecah kalimat: akhir kalimat atau pergantian baris. Transkrip Whisper sering
# tanpa tanda baca, jadi baris baru ikut dihitung sebagai batas.
PISAH_KALIMAT = re.compile(r"(?<=[.!?])\s+|\n+")


def kalimat(teks: str) -> list[str]:
    return [s.strip() for s in PISAH_KALIMAT.split(teks or "") if s.strip()]


def varian(kal: list[str], rng: np.random.Generator) -> str:
    """Ambil 50-100% kalimat secara acak, urutannya ikut diacak.

    Mengacak urutan disengaja: label di dataset ini melacak topik dan konsep
    (RENCANA.md §17), bukan alur narasi - jadi urutan kalimat bukan sinyal yang
    perlu dipertahankan, dan mengacaknya mencegah model menghafal pembukaan
    video yang seragam ("halo guys, balik lagi di channel...").
    """
    n = len(kal)
    lo = max(1, int(np.ceil(n * 0.5)))
    k = int(rng.integers(lo, n + 1))
    pilih = rng.permutation(n)[:k]
    return " ".join(kal[i] for i in pilih)


def augmentasi(teks: list[str], label: np.ndarray, n_aug: int,
               seed: int) -> tuple[list[str], np.ndarray]:
    """Kembangkan train set jadi beberapa varian per baris.

    Teks asli selalu ikut. Varian dikumpulkan lewat `set` supaya duplikat gugur
    sendiri - baris berkalimat sedikit otomatis menghasilkan varian lebih
    sedikit, dan itu memang yang diinginkan.
    """
    if n_aug <= 0:
        return list(teks), label.copy()

    rng = np.random.default_rng(seed)
    keluar_teks: list[str] = []
    keluar_label: list[int] = []
    for t, y in zip(teks, label):
        kal = kalimat(t)
        v = {t}
        if len(kal) >= 2:                       # <2 kalimat: tidak ada yang bisa disubsample
            for _ in range(n_aug):
                v.add(varian(kal, rng))
        v = sorted(v)                           # sorted -> urutan deterministik
        keluar_teks += v
        keluar_label += [int(y)] * len(v)
    return keluar_teks, np.array(keluar_label, dtype=np.int64)


def main(argv: list[str] | None = None) -> int:
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import LabelEncoder
    from torch.utils.data import DataLoader, Dataset
    from transformers import (AutoModelForSequenceClassification, AutoTokenizer,
                              DataCollatorWithPadding,
                              get_linear_schedule_with_warmup)

    import train as T

    ap = argparse.ArgumentParser()
    ap.add_argument("--n-augment", type=int, default=N_AUGMENT,
                    help="varian augmentasi per baris; 0 = tanpa augmentasi")
    ap.add_argument("--epochs", type=int, default=EPOCHS)
    ap.add_argument("--lr", type=float, default=LR)
    ap.add_argument("--alpha", type=float, default=ALPHA_BOBOT)
    # text_combined menang untuk TF-IDF (§14), tapi kesimpulan itu belum tentu
    # berlaku di sini: TF-IDF tidak punya batas panjang, sedangkan menambahkan
    # caption ke transkrip menaikkan pemotongan di 512 token dari 5,7% ke 25,0%
    # demi 1,7 poin cakupan. Kerugian itu hanya diderita BERT.
    ap.add_argument("--kolom-teks", default="text_combined",
                    choices=["text_combined", "text_transcript", "text_caption"])
    ap.add_argument("--tag", default="v2", help="akhiran nama file keluaran")
    args = ap.parse_args(argv)

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    # BF16 punya rentang eksponen seluas FP32, jadi tidak perlu GradScaler dan
    # lebih stabil dari FP16 - penting karena loss di sini berbobot kelas.
    amp = (torch.bfloat16 if dev == "cuda" and torch.cuda.is_bf16_supported()
           else torch.float16 if dev == "cuda" else None)

    def autocast():
        return torch.autocast(dev, dtype=amp) if amp else contextlib.nullcontext()

    ds = pd.read_parquet(FEATURES_DIR / "dataset.parquet")
    tr = ds[ds.split == "train"].reset_index(drop=True)
    teks = tr[args.kolom_teks].fillna("").to_numpy()
    y = tr["emotion"].to_numpy()
    grup = tr["key"].to_numpy()

    le = LabelEncoder().fit(y)
    # int64 wajib: di Windows LabelEncoder mengembalikan int32, sedangkan
    # cross_entropy hanya menerima Long/Byte.
    yi = le.transform(y).astype(np.int64)
    n_kelas = len(le.classes_)

    kosong = (pd.Series(teks).str.strip() == "").sum()
    print(f"device {dev} | amp {amp} | {len(tr)} baris | {n_kelas} kelas")
    print(f"kolom teks {args.kolom_teks} | {len(tr)-kosong} berisi "
          f"({100*(len(tr)-kosong)/len(tr):.1f}% cakupan)")
    print(f"augmentasi {args.n_augment} varian | lr {args.lr} | alpha {args.alpha} "
          f"| maks {args.epochs} epoch (early stop patience {PATIENCE})\n")

    tok = AutoTokenizer.from_pretrained(MODEL)
    collator = DataCollatorWithPadding(tokenizer=tok, return_tensors="pt")

    class DS(Dataset):
        """Tokenisasi TANPA padding - padding dilakukan per batch oleh collator,
        jadi tiap batch hanya selebar sekuens terpanjangnya."""

        def __init__(self, txt, lab):
            enc = tok(list(txt), truncation=True, max_length=MAX_LEN)
            self.items = [{**{k: enc[k][i] for k in enc}, "labels": int(lab[i])}
                          for i in range(len(lab))]

        def __len__(self):
            return len(self.items)

        def __getitem__(self, i):
            return self.items[i]

    def prediksi(model, loader) -> np.ndarray:
        model.eval()
        out = []
        with torch.inference_mode():
            for b in loader:
                b = {k: v.to(dev, non_blocking=True) for k, v in b.items()}
                b.pop("labels", None)
                with autocast():
                    logit = model(**b).logits
                out.append(logit.float().argmax(-1).cpu().numpy())
        return np.concatenate(out)

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(tr), dtype=int)
    jejak = []
    t0 = time.time()

    for fold, (a, b) in enumerate(cv.split(tr, y, groups=grup), 1):
        torch.manual_seed(SEED)
        np.random.seed(SEED)

        # Validasi dalam untuk early stopping, dipotong dari fold-train saja.
        # Tetap dikelompokkan per key supaya URL duplikat tidak bocor ke sini.
        icv = StratifiedGroupKFold(n_splits=INNER_SPLITS, shuffle=True,
                                   random_state=SEED)
        ia, ib = next(icv.split(a, y[a], groups=grup[a]))
        dalam, vdalam = a[ia], a[ib]

        # Augmentasi HANYA pada train dalam. Validasi dalam dan fold luar tetap
        # murni - kalau varian dari baris yang sama muncul di keduanya, early
        # stopping akan memilih epoch berdasarkan hafalan.
        tx_tr, y_tr = augmentasi(teks[dalam], yi[dalam], args.n_augment, SEED + fold)

        # Bobot dihitung dari label SEBELUM augmentasi, konsisten dengan
        # train.py - augmentasi menggeser jumlah per kelas sedikit (baris
        # berkalimat sedikit menghasilkan varian lebih sedikit) dan pergeseran
        # itu bukan bagian dari keputusan pembobotan.
        w = T.bobot_kelas(y[dalam], args.alpha) or {}
        bobot = torch.tensor([w.get(c, 1.0) for c in le.classes_],
                             dtype=torch.float, device=dev)
        lossf = nn.CrossEntropyLoss(weight=bobot)

        dl_tr = DataLoader(DS(tx_tr, y_tr), batch_size=BATCH, shuffle=True,
                           collate_fn=collator, pin_memory=(dev == "cuda"),
                           generator=torch.Generator().manual_seed(SEED + fold))
        dl_vd = DataLoader(DS(teks[vdalam], yi[vdalam]), batch_size=EVAL_BATCH,
                           collate_fn=collator)
        dl_luar = DataLoader(DS(teks[b], yi[b]), batch_size=EVAL_BATCH,
                             collate_fn=collator)

        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=n_kelas).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr,
                                weight_decay=WEIGHT_DECAY)
        total = len(dl_tr) * args.epochs
        sched = get_linear_schedule_with_warmup(
            opt, int(total * WARMUP_RATIO), total)
        scaler = torch.amp.GradScaler("cuda") if amp == torch.float16 else None

        f1_terbaik, ep_terbaik, mandek, bobot_terbaik = -1.0, 0, 0, None
        for ep in range(1, args.epochs + 1):
            model.train()
            for batch in dl_tr:
                batch = {k: v.to(dev, non_blocking=True) for k, v in batch.items()}
                lab = batch.pop("labels")
                opt.zero_grad(set_to_none=True)
                with autocast():
                    loss = lossf(model(**batch).logits, lab)
                if scaler is not None:
                    scaler.scale(loss).backward()
                    scaler.unscale_(opt)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    scaler.step(opt); scaler.update()
                else:
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    opt.step()
                sched.step()

            pv = prediksi(model, dl_vd)
            f1v = f1_score(yi[vdalam], pv, average="macro", zero_division=0)
            accv = accuracy_score(yi[vdalam], pv)

            tanda = ""
            if ep >= MIN_EPOCHS:
                if f1v > f1_terbaik:
                    f1_terbaik, ep_terbaik, mandek = f1v, ep, 0
                    # Disimpan di RAM CPU (~0,5 GB) supaya VRAM tetap lega.
                    bobot_terbaik = {k: v.detach().cpu().clone()
                                     for k, v in model.state_dict().items()}
                    tanda = "  <- terbaik"
                else:
                    mandek += 1
            print(f"    fold {fold} ep {ep}/{args.epochs}  val-dalam "
                  f"acc {accv*100:5.1f}%  macroF1 {f1v:.3f}{tanda}", flush=True)
            if mandek >= PATIENCE:
                print(f"    early stop (macro-F1 mandek {PATIENCE} epoch)")
                break

        if bobot_terbaik is not None:
            model.load_state_dict(bobot_terbaik)
        oof[b] = prediksi(model, dl_luar)

        acc = accuracy_score(yi[b], oof[b])
        f1 = f1_score(yi[b], oof[b], average="macro", zero_division=0)
        jejak.append({"fold": fold, "epoch_terbaik": ep_terbaik,
                      "n_train_augmentasi": len(tx_tr), "acc_luar": acc,
                      "macro_f1_luar": f1})
        print(f"  fold {fold}/5  epoch terbaik {ep_terbaik}  "
              f"train {len(dalam)}->{len(tx_tr)} sampel  |  fold-luar "
              f"acc {acc*100:5.1f}%  macroF1 {f1:.3f}  "
              f"({(time.time()-t0)/60:.1f} menit)\n", flush=True)

        del model, opt, bobot_terbaik
        if dev == "cuda":
            torch.cuda.empty_cache()

    pred = le.inverse_transform(oof)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro", zero_division=0)
    accs = [j["acc_luar"] for j in jejak]

    print("=" * 66)
    print(f"IndoBERT v2 (augmentasi {args.n_augment}, early stop)   "
          f"acc {acc*100:.1f}%   macroF1 {f1:.3f}")
    print(f"  ragam antar-fold: {min(accs)*100:.1f}% - {max(accs)*100:.1f}%")
    print("-" * 66)
    print("  IndoBERT fine-tuned (v1)   acc  37,4%   macroF1 0,129")
    print("  TF-IDF + konsep (final)    acc  42,3%   macroF1 0,135   <- acuan")
    print("  baseline tebak Surprise    acc  41,2%   macroF1 0,058")
    print("=" * 66)
    print(classification_report(y, pred, zero_division=0))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": tr["id"], "y": y, "oof": pred}).to_csv(
        OUTPUTS_DIR / f"oof_indobert_{args.tag}.csv", index=False)
    pd.DataFrame(jejak).to_csv(
        OUTPUTS_DIR / f"jejak_indobert_{args.tag}.csv", index=False)
    print(f"\ndisimpan: outputs/oof_indobert_{args.tag}.csv, "
          f"outputs/jejak_indobert_{args.tag}.csv")
    return 0


if __name__ == "__main__":
    sys.exit(main())
