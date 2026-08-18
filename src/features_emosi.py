"""
Lapisan emosi - menyerang emosinya langsung, bukan topiknya.

Seluruh proyek ini menemukan hal yang sama berulang kali: label melacak TOPIK
(§5A, §18). Tapi kesimpulan itu punya lubang yang belum pernah diperiksa, dan
lubangnya bernama **dilusi**.

Transkrip median 160 kata. Kalimat "aku sedih deh harus tinggal di sini" adalah
7 kata dari 160 - dan TF-IDF dengan `sublinear_tf` nyaris tidak mencatatnya,
sementara 153 kata topik di sekitarnya menenggelamkannya. Leksikon emosi yang
gagal di §5A dihitung dengan cara yang sama: menjumlahkan kata emosi di seluruh
dokumen lalu membaginya dengan panjang. Jadi kegagalan itu **belum membuktikan
kata emosinya tidak ada** - baru membuktikan bahwa kalau ada, ia tenggelam.

Modul ini menyerang dilusi itu dengan tiga lapis:

**A. Skor leksikon InSet.** Leksikon sentimen Bahasa Indonesia terbitan Koto &
Rahmaningtyas (IALP 2017): 3.609 kata positif dan 6.609 kata negatif dengan
bobot -5..+5. Dipakai berbobot, bukan sekadar dihitung, dan **dengan penanganan
negasi** - "tidak sedih" tidak boleh dihitung sebagai sedih. Jurnal yang jadi
rujukan (Zakira dkk., JIEnGS 2025) memakai InSet tanpa negasi; penambahan ini
perbaikan sadar, bukan penyimpangan tak sengaja.

**B. Ekstraksi kalimat emosional.** Ini inti gagasannya. Alih-alih merata-rata
seluruh dokumen, tiap kalimat diberi skor emosional, lalu hanya kalimat
ber-skor tertinggi yang disimpan sebagai `text_emosi`. Kalimat "aku sedih deh
harus tinggal di sini" berhenti bersaing dengan 153 kata soal harga bensin.
Skornya memakai bobot InSet, bukan daftar kata buatan sendiri - §17 sudah
menunjukkan daftar buatan tangan yang dipanjangkan justru memperburuk.

**C. Probabilitas model emosi eksternal.** Ini yang paling menjanjikan.
Model-model ini dilatih pada teks Bahasa Indonesia BERLABEL EMOSI BERSIH, jadi
keluarannya menjawab "teks ini mengekspresikan emosi apa" - lepas sepenuhnya
dari label topik kita yang bernoise. Kebetulan yang menguntungkan: label kita
adalah 6 dari 8 emosi dasar Plutchik.

| Model | Kelas | Menutup label kita |
|---|---|---|
| NusaBERT Plutchik | Anger, Anticipation, Disgust, Fear, Joy, Sadness, Surprise, Trust | Surprise, Trust, Joy, Anger, Sad, Fear |
| IndoBERT thoriqfy | Sadness, Anger, Love, Fear, Happy, Neutral | + Love, Neutral |

Delapan dari sepuluh label tertutup; hanya Proud dan Loyalty tidak.

Probabilitasnya diambil dengan **max-pooling antar potongan, bukan mean**:
emosi adalah puncak, bukan rata-rata. Satu kalimat marah di tengah video netral
membuat videonya marah; merata-ratakannya justru menghapus sinyal itu - kesalahan
yang sama bentuknya dengan dilusi yang sedang kita perbaiki.

Jalankan:  python src/features_emosi.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES_DIR, ROOT, describe, save_features  # noqa: E402

LEKSIKON_DIR = ROOT / "data" / "leksikon"
KOLOM_TEKS = "text_combined"

MODEL_EMOSI = [
    ("plutchik", "Aardiiiiy/NusaBERT-base-Indonesian-Plutchik-emotion-analysis"),
    ("emo6", "thoriqfy/indobert-emotion-classification"),
]

# Negasi Bahasa Indonesia baku + gaul. Cakupannya 2 kata ke depan: "tidak
# terlalu sedih" harus ikut terbalik, tapi "tidak, saya kemarin sedih" jangan.
NEGASI = {"tidak", "tak", "bukan", "belum", "jangan", "nggak", "ngga", "gak",
          "ga", "engga", "enggak", "kagak", "tanpa", "kurang"}
JANGKAUAN_NEGASI = 2

# Penanda pernyataan orang pertama. Kalimat "AKU sedih" jauh lebih kuat sebagai
# bukti emosi daripada "dia sedih" atau "kalau sedih".
ORANG_PERTAMA = {"aku", "gue", "gua", "gw", "saya", "kita", "kami", "ku", "aq",
                 "akutu", "gweh"}
BONUS_ORANG_PERTAMA = 1.5

# Whisper tidak memberi tanda baca, sehingga 15,2% "kalimat" memuat 62,6%
# seluruh kata - ada yang panjangnya 745 kata. Menilai unit sebesar itu sebagai
# satu kesatuan sama saja dengan tidak memadatkan. Karena itu segmen panjang
# dipecah lagi jadi jendela selebar JENDELA_KATA.
MAKS_KATA_SEGMEN = 30
JENDELA_KATA = 25

# Porsi segmen yang disimpan. Median dokumen punya 9 kalimat; mengambil angka
# TETAP (dulu 5) berarti nyaris tidak memadatkan. Porsi relatif menyesuaikan
# diri pada dokumen pendek maupun panjang.
PORSI_EMOSI = 0.30
MIN_SEGMEN, MAKS_SEGMEN = 2, 8

PISAH_KALIMAT = re.compile(r"(?<=[.!?])\s+|\n+")
KATA = re.compile(r"[a-zA-Z']+")


def segmentasi(teks: str) -> list[str]:
    """Pecah jadi segmen yang cukup pendek untuk dinilai secara lokal.

    Batas kalimat dipakai kalau ada; segmen yang tetap terlalu panjang - khas
    keluaran Whisper tanpa tanda baca - dipotong jadi jendela kata.
    """
    keluar = []
    for s in PISAH_KALIMAT.split(teks or ""):
        s = s.strip()
        if not s:
            continue
        kata = s.split()
        if len(kata) <= MAKS_KATA_SEGMEN:
            keluar.append(s)
        else:
            for a in range(0, len(kata), JENDELA_KATA):
                keluar.append(" ".join(kata[a:a + JENDELA_KATA]))
    return keluar


def muat_inset() -> dict[str, float]:
    """Gabungan InSet positif + negatif jadi satu peta kata -> bobot."""
    peta: dict[str, float] = {}
    for nama in ("positive", "negative"):
        path = LEKSIKON_DIR / f"inset_{nama}.tsv"
        if not path.exists():
            raise FileNotFoundError(
                f"{path} tidak ada. Unduh dulu dari github.com/fajri91/InSet")
        df = pd.read_csv(path, sep="\t")
        for w, b in zip(df["word"].astype(str), df["weight"].astype(float)):
            peta[w.lower()] = b
    return peta


def skor_kalimat(kalimat: str, inset: dict[str, float]) -> tuple[float, float, int]:
    """Skor sentimen satu kalimat, dengan negasi.

    Mengembalikan (skor_bertanda, intensitas_puncak, ada_orang_pertama).
    `intensitas_puncak` dipakai untuk peringkat: kalimat dengan satu kata sangat
    emosional lebih berharga daripada kalimat panjang berisi banyak kata hambar.
    """
    kata = [k.lower() for k in KATA.findall(kalimat)]
    if not kata:
        return 0.0, 0.0, 0

    total, puncak = 0.0, 0.0
    for i, k in enumerate(kata):
        b = inset.get(k)
        if b is None:
            continue
        # negasi membalik tanda kalau muncul dalam JANGKAUAN_NEGASI kata sebelumnya
        awal = max(0, i - JANGKAUAN_NEGASI)
        if any(x in NEGASI for x in kata[awal:i]):
            b = -b
        total += b
        puncak = max(puncak, abs(b))

    org1 = int(any(k in ORANG_PERTAMA for k in kata))
    return total, puncak, org1


def kalimat_emosional(teks: str, inset: dict[str, float]) -> str:
    """Ambil segmen paling emosional, urutan asli dipertahankan.

    Inilah lawan langsung dari dilusi: dokumen ratusan kata disaring jadi
    beberapa segmen yang benar-benar membawa muatan emosi.
    """
    kal = segmentasi(teks)
    if not kal:
        return ""
    n = int(np.clip(round(len(kal) * PORSI_EMOSI), MIN_SEGMEN, MAKS_SEGMEN))
    if len(kal) <= n:
        return " ".join(kal)

    nilai = []
    for i, s in enumerate(kal):
        total, puncak, org1 = skor_kalimat(s, inset)
        # peringkat memakai intensitas puncak + besaran total, dinaikkan kalau
        # segmennya pernyataan orang pertama
        v = (puncak + abs(total) * 0.3) * (BONUS_ORANG_PERTAMA if org1 else 1.0)
        nilai.append((v, i))
    pilih = sorted(i for _, i in sorted(nilai, reverse=True)[:n])
    return " ".join(kal[i] for i in pilih)


def fitur_leksikon(teks: list[str], inset: dict[str, float]) -> pd.DataFrame:
    """Fitur agregat InSet per dokumen."""
    baris = []
    for t in teks:
        kal = segmentasi(t)
        skor = [skor_kalimat(s, inset) for s in kal] or [(0.0, 0.0, 0)]
        total = np.array([s[0] for s in skor])
        puncak = np.array([s[1] for s in skor])
        org1 = np.array([s[2] for s in skor])
        n_kata = max(len(KATA.findall(t or "")), 1)

        baris.append({
            "ins_skor_total": total.sum(),
            "ins_skor_per_kata": total.sum() / n_kata,
            "ins_pos_rasio": float((total > 0).mean()),
            "ins_neg_rasio": float((total < 0).mean()),
            # puncak: kalimat paling emosional di seluruh dokumen. Ini yang
            # tahan dilusi - tidak peduli seberapa panjang dokumennya.
            "ins_puncak_pos": float(total.max()),
            "ins_puncak_neg": float(total.min()),
            "ins_intensitas_maks": float(puncak.max()),
            "ins_polaritas_ragam": float(total.std()),
            "ins_org1_rasio": float(org1.mean()),
            # kalimat orang pertama yang juga emosional - pola "aku sedih deh"
            "ins_org1_emosional": float((org1 * np.abs(total)).max()),
        })
    return pd.DataFrame(baris)


def prob_emosi(teks: list[str], model_id: str, batch: int = 16) -> pd.DataFrame:
    """Probabilitas emosi dari model eksternal, max-pooled antar potongan.

    Max, bukan mean: emosi itu puncak. Satu kalimat marah di tengah video netral
    membuat videonya marah - merata-ratakan justru menghapus sinyalnya.
    """
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    dev = "cuda" if torch.cuda.is_available() else "cpu"
    tok = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id).to(dev).eval()
    nama = [model.config.id2label[i] for i in range(model.config.num_labels)]
    maks = min(getattr(tok, "model_max_length", 512) or 512, 512)

    # potong tiap dokumen jadi bagian yang muat, lalu satukan supaya batch padat
    potongan, pemilik = [], []
    for i, t in enumerate(teks):
        ids = tok.encode(t or "", add_special_tokens=False)
        if not ids:
            potongan.append([tok.cls_token_id, tok.sep_token_id]); pemilik.append(i)
            continue
        for a in range(0, len(ids), maks - 2):
            potongan.append([tok.cls_token_id] + ids[a:a + maks - 2] + [tok.sep_token_id])
            pemilik.append(i)

    hasil = np.zeros((len(potongan), len(nama)), dtype=np.float32)
    with torch.inference_mode():
        for a in range(0, len(potongan), batch):
            b = potongan[a:a + batch]
            n = max(len(x) for x in b)
            ids = torch.full((len(b), n), tok.pad_token_id, dtype=torch.long)
            msk = torch.zeros((len(b), n), dtype=torch.long)
            for j, x in enumerate(b):
                ids[j, :len(x)] = torch.tensor(x); msk[j, :len(x)] = 1
            out = model(input_ids=ids.to(dev), attention_mask=msk.to(dev)).logits
            hasil[a:a + len(b)] = torch.softmax(out.float(), -1).cpu().numpy()

    pemilik = np.array(pemilik)
    emb = np.zeros((len(teks), len(nama)), dtype=np.float32)
    for i in range(len(teks)):
        emb[i] = hasil[pemilik == i].max(0)      # max-pooling
    del model
    if dev == "cuda":
        torch.cuda.empty_cache()
    return pd.DataFrame(emb, columns=nama)


def build() -> pd.DataFrame:
    ds = pd.read_parquet(FEATURES_DIR / "dataset.parquet")
    teks = ds[KOLOM_TEKS].fillna("").tolist()
    inset = muat_inset()
    print(f"InSet dimuat: {len(inset)} kata")

    # --- B. ekstraksi kalimat emosional ---
    teks_emosi = [kalimat_emosional(t, inset) for t in teks]
    panjang_asli = np.array([len(KATA.findall(t)) for t in teks])
    panjang_baru = np.array([len(KATA.findall(t)) for t in teks_emosi])
    isi = panjang_asli > 0
    print(f"ekstraksi kalimat: {panjang_asli[isi].mean():.0f} -> "
          f"{panjang_baru[isi].mean():.0f} kata rata-rata "
          f"(pemadatan {panjang_asli[isi].mean()/max(panjang_baru[isi].mean(),1):.1f}x)")

    out = ds[["split", "id", "key", "emotion"]].copy()
    out["text_emosi"] = teks_emosi

    # --- A. fitur leksikon InSet ---
    lek = fitur_leksikon(teks, inset)
    out = pd.concat([out.reset_index(drop=True), lek], axis=1)
    print(f"fitur leksikon: {lek.shape[1]} kolom")

    # --- C. probabilitas model emosi eksternal ---
    for tag, mid in MODEL_EMOSI:
        print(f"\nmenjalankan {mid} ...")
        # dihitung DUA kali: pada teks penuh, dan pada kalimat emosional saja.
        # Perbandingan keduanya sekaligus menguji hipotesis dilusi.
        for suf, sumber in (("penuh", teks), ("padat", teks_emosi)):
            p = prob_emosi(sumber, mid)
            p.columns = [f"emo_{tag}_{suf}_{c}".lower() for c in p.columns]
            out = pd.concat([out, p], axis=1)
            print(f"  {suf}: {p.shape[1]} kolom, "
                  f"rerata prob maks {p.max(axis=1).mean():.3f}")

    return out


if __name__ == "__main__":
    # Tahap ini butuh dua model HuggingFace (~1 GB unduhan). Kalau hasilnya
    # sudah ada - dan hasilnya IKUT DIKIRIM dalam paket - tahap ini dilewati,
    # sehingga panitia bisa menjalankan run_all.py tanpa internet. Pakai
    # --force untuk membangun ulang.
    keluaran = FEATURES_DIR / "emosi.parquet"
    if keluaran.exists() and "--force" not in sys.argv:
        n = len(pd.read_parquet(keluaran))
        print(f"memakai cache: {keluaran.name} ({n} baris). "
              f"Pakai --force untuk membangun ulang.")
        sys.exit(0)

    df = build()
    save_features(df.drop(columns=["text_emosi"]), "emosi")
    # teks padat disimpan terpisah supaya bisa divektorkan TF-IDF di train
    df[["split", "id", "key", "text_emosi"]].to_parquet(
        FEATURES_DIR / "teks_emosi.parquet", index=False)
    describe(df.drop(columns=["text_emosi"]), "emosi")
    print(f"\nteks padat: {FEATURES_DIR/'teks_emosi.parquet'}")
