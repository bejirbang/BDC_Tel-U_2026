"""
Fine-tuning IndoBERT, diuji pada skema validasi silang yang sama persis.

Langkah ini dijalankan setelah IndoBERT beku kalah dari TF-IDF, karena embedding
beku memang dikenal lemah tanpa adaptasi tugas - jadi kekalahan itu belum
menjawab pertanyaan sebenarnya.

Risikonya nyata dan disadari: 124 juta parameter dilatih pada 803 baris dengan
10 kelas, dua di antaranya bersampel tunggal, dan label yang bernoise (plafon
kesepakatan anotator ~56%). Karena itu dipakai learning rate kecil, epoch
sedikit, dan bobot kelas pada fungsi loss.

Perbandingannya adil: fold yang sama, seed yang sama, metrik yang sama seperti
seluruh tahap lain di `train.py`.

Jalankan:  python src/finetune_bert.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES_DIR, OUTPUTS_DIR, SEED  # noqa: E402

MODEL = "indobenchmark/indobert-base-p1"
MAX_LEN = 512
BATCH = 8
EPOCHS = 4
LR = 2e-5
ALPHA_BOBOT = 0.5


def main() -> int:
    import torch
    import torch.nn as nn
    from sklearn.metrics import accuracy_score, classification_report, f1_score
    from sklearn.model_selection import StratifiedGroupKFold
    from sklearn.preprocessing import LabelEncoder
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    import train as T

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    ds = pd.read_parquet(FEATURES_DIR / "dataset.parquet")
    tr = ds[ds.split == "train"].reset_index(drop=True)
    teks = tr["text_combined"].fillna("").tolist()
    y = tr["emotion"].to_numpy()
    grup = tr["key"].to_numpy()

    le = LabelEncoder().fit(y)
    # int64 wajib: di Windows LabelEncoder mengembalikan int32, sedangkan
    # cross_entropy hanya menerima Long/Byte.
    yi = le.transform(y).astype(np.int64)
    n_kelas = len(le.classes_)

    tok = AutoTokenizer.from_pretrained(MODEL)
    enc = tok(teks, truncation=True, max_length=MAX_LEN, padding="max_length",
              return_tensors="pt")
    ids_all, mask_all = enc["input_ids"], enc["attention_mask"]

    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=SEED)
    oof = np.zeros(len(tr), dtype=int)
    t0 = time.time()

    for fold, (a, b) in enumerate(cv.split(tr, y, groups=grup), 1):
        torch.manual_seed(SEED)
        model = AutoModelForSequenceClassification.from_pretrained(
            MODEL, num_labels=n_kelas).to(dev)

        # Bobot kelas dihitung dari fold ini saja, konsisten dengan train.py
        w = T.bobot_kelas(y[a], ALPHA_BOBOT) or {}
        bobot = torch.tensor(
            [w.get(c, 1.0) for c in le.classes_], dtype=torch.float, device=dev)
        lossf = nn.CrossEntropyLoss(weight=bobot)

        opt = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=0.01)
        n_step = (len(a) // BATCH + 1) * EPOCHS
        sched = torch.optim.lr_scheduler.OneCycleLR(
            opt, max_lr=LR, total_steps=n_step, pct_start=0.1)
        scaler = torch.amp.GradScaler("cuda", enabled=(dev == "cuda"))

        model.train()
        for ep in range(EPOCHS):
            urut = np.random.RandomState(SEED + ep).permutation(a)
            for i in range(0, len(urut), BATCH):
                idx = urut[i:i + BATCH]
                opt.zero_grad(set_to_none=True)
                with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                    out = model(input_ids=ids_all[idx].to(dev),
                                attention_mask=mask_all[idx].to(dev)).logits
                    loss = lossf(out, torch.as_tensor(yi[idx], dtype=torch.long,
                                                      device=dev))
                scaler.scale(loss).backward()
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                scaler.step(opt); scaler.update(); sched.step()

        model.eval()
        pred = []
        with torch.no_grad():
            for i in range(0, len(b), 32):
                idx = b[i:i + 32]
                with torch.amp.autocast("cuda", enabled=(dev == "cuda")):
                    out = model(input_ids=ids_all[idx].to(dev),
                                attention_mask=mask_all[idx].to(dev)).logits
                pred.append(out.float().argmax(1).cpu().numpy())
        oof[b] = np.concatenate(pred)

        acc = accuracy_score(yi[b], oof[b])
        print(f"  fold {fold}/5  acc {acc*100:5.1f}%  "
              f"({(time.time()-t0)/60:.1f} menit berlalu)")
        del model
        torch.cuda.empty_cache()

    pred = le.inverse_transform(oof)
    acc = accuracy_score(y, pred)
    f1 = f1_score(y, pred, average="macro", zero_division=0)
    print("\n" + "=" * 60)
    print(f"IndoBERT fine-tuned   acc {acc*100:.1f}%   macroF1 {f1:.3f}")
    print(f"TF-IDF + LogReg       acc  42.0%   macroF1 0.099   <- acuan")
    print(f"baseline              acc  41.2%   macroF1 0.058")
    print("=" * 60)
    print(classification_report(y, pred, zero_division=0))

    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": tr["id"], "y": y, "oof": pred}).to_csv(
        OUTPUTS_DIR / "oof_indobert.csv", index=False)
    return 0


if __name__ == "__main__":
    sys.exit(main())
