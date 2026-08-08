"""
Menghasilkan seluruh grafik untuk laporan ke `outputs/figures/`.

Jalankan:  python src/viz.py

Aturan visual yang dipegang konsisten di semua grafik:
  - Satu seri = satu warna. Warna TIDAK dipakai untuk mengulang panjang batang.
  - Skala magnitudo memakai satu rona terang->gelap, bukan pelangi.
  - Tidak ada sumbu-Y ganda. Dua besaran berbeda skala = dua panel terpisah.
  - Garis bantu tipis dan redup; angka ditulis selektif, bukan di setiap titik.
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import FEATURES_DIR, OUTPUTS_DIR  # noqa: E402

FIGDIR = OUTPUTS_DIR / "figures"

# Palet kategorikal - urutan slot tetap, tidak pernah diputar ulang.
C = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100",
     "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
TEKS = "#0b0b0b"
TEKS2 = "#52514e"
REDUP = "#b8b7b2"
NETRAL = "#d6d5d0"
# Ramp magnitudo: SATU rona (biru slot 1) terang -> gelap. Bukan pelangi.
RAMP = LinearSegmentedColormap.from_list(
    "biru", ["#f4f8fd", "#9cc3ee", "#2a78d6", "#14416f"], N=256)


def gaya() -> None:
    plt.rcParams.update({
        "figure.facecolor": "white", "axes.facecolor": "white",
        "font.size": 10, "font.family": "DejaVu Sans",
        "axes.edgecolor": REDUP, "axes.linewidth": 0.8,
        "axes.labelcolor": TEKS2, "axes.titlecolor": TEKS,
        "axes.titlesize": 12, "axes.titleweight": "bold", "axes.titlepad": 12,
        "xtick.color": TEKS2, "ytick.color": TEKS2,
        "xtick.labelsize": 9, "ytick.labelsize": 9,
        "grid.color": "#ececea", "grid.linewidth": 0.8,
        "legend.frameon": False, "figure.dpi": 130,
        "savefig.bbox": "tight", "savefig.facecolor": "white",
    })


def rapikan(ax, grid_x: bool = False) -> None:
    for sisi in ("top", "right"):
        ax.spines[sisi].set_visible(False)
    ax.grid(axis="x" if grid_x else "y", alpha=0.9)
    ax.set_axisbelow(True)


def simpan(fig, nama: str) -> Path:
    FIGDIR.mkdir(parents=True, exist_ok=True)
    p = FIGDIR / f"{nama}.png"
    fig.savefig(p)
    plt.close(fig)
    print(f"  + {p.name}")
    return p


# --------------------------------------------------------------------------

def fig_distribusi_label(df: pd.DataFrame) -> None:
    """Satu seri -> satu warna. Kelas mustahil ditandai warna redup + anotasi."""
    tr = df[df.split == "train"]
    n = tr.emotion.value_counts()
    mustahil = {"Love", "Loyalty"}
    warna = [NETRAL if e in mustahil else C[0] for e in n.index]

    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.bar(range(len(n)), n.values, color=warna, width=0.68, zorder=3)
    for i, (e, v) in enumerate(n.items()):
        ax.text(i, v + 6, str(v), ha="center", va="bottom", fontsize=9,
                color=TEKS2 if e not in mustahil else REDUP)
    ax.set_xticks(range(len(n)))
    ax.set_xticklabels(n.index, rotation=35, ha="right")
    ax.set_ylabel("jumlah baris latih")
    ax.set_ylim(0, n.max() * 1.16)
    ax.set_title("Distribusi label sangat timpang — dua kelas mustahil dipelajari")
    ax.annotate("1 sampel:\ntidak akan pernah diprediksi",
                xy=(8.5, 8), xytext=(6.6, 118), fontsize=8.5, color=TEKS2,
                ha="center", arrowprops=dict(arrowstyle="-", color=REDUP, lw=0.9))
    ax.text(0, n.max() * 1.07, f"tebak «Surprise» selalu = {n.iloc[0]/len(tr)*100:.1f}% akurasi",
            fontsize=8.5, color=TEKS2)
    rapikan(ax)
    simpan(fig, "01_distribusi_label")


def fig_cakupan(df: pd.DataFrame) -> None:
    """Dua panel terpisah, bukan sumbu ganda: satuan keduanya sama (%) tapi
    ceritanya beda (sumber vs jenis sinyal)."""
    tr = df[df.split == "train"]
    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11, 4), gridspec_kw={"wspace": 0.32})

    # kiri: keberhasilan akuisisi per sumber
    sumber = pd.DataFrame({
        "sumber": ["Instagram", "Google Drive", "CDN mentah"],
        "berhasil": [621, 221, 0], "gagal": [16, 75, 24]})
    y = np.arange(len(sumber))
    a1.barh(y, sumber.berhasil, color=C[0], height=0.6, zorder=3, label="berhasil")
    a1.barh(y, sumber.gagal, left=sumber.berhasil + 2, color=NETRAL, height=0.6,
            zorder=3, label="gagal")
    for i, r in sumber.iterrows():
        tot = r.berhasil + r.gagal
        a1.text(tot + 14, i, f"{r.berhasil}/{tot}", va="center", fontsize=8.5, color=TEKS2)
    a1.set_yticks(y); a1.set_yticklabels(sumber.sumber)
    a1.set_xlabel("tautan unik"); a1.set_xlim(0, 760)
    a1.invert_yaxis()
    a1.set_title("Akuisisi per sumber", fontsize=11)
    a1.legend(fontsize=8.5, loc="lower right")
    rapikan(a1, grid_x=True)

    # kanan: cakupan sinyal
    cak = {
        "caption": (tr.cap_len > 0).mean() * 100,
        "transkrip\nWhisper": (tr.text_transcript.str.len() > 0).mean() * 100,
        "gabungan": (tr.text_combined.str.len() > 0).mean() * 100,
    }
    warna = [C[0], C[2], C[2]]
    b = a2.bar(range(3), list(cak.values()), color=warna, width=0.6, zorder=3)
    for i, v in enumerate(cak.values()):
        a2.text(i, v + 1.6, f"{v:.1f}%", ha="center", fontsize=9, color=TEKS2)
    a2.set_xticks(range(3)); a2.set_xticklabels(list(cak.keys()))
    a2.set_ylabel("% baris latih"); a2.set_ylim(0, 118)
    a2.set_title("Transkrip menutup celah caption", fontsize=11)
    # Panah ditaruh DI ATAS batang, bukan menyilanginya, supaya tidak menutupi data.
    a2.annotate("", xy=(1, 97), xytext=(0, 97),
                arrowprops=dict(arrowstyle="->", color=C[2], lw=1.3))
    a2.text(0.5, 100, "+25 poin cakupan", fontsize=8.5, color=C[2], ha="center")
    rapikan(a2)

    fig.suptitle("Sekitar 12% data hilang permanen — tautan sudah mati sejak dataset disusun",
                 fontsize=12, fontweight="bold", y=1.03)
    simpan(fig, "02_cakupan_data")


def fig_kata_pembeda(df: pd.DataFrame) -> None:
    """Bukti bahwa label melacak TOPIK, bukan kata emosi."""
    from sklearn.feature_extraction.text import CountVectorizer
    tr = df[(df.split == "train") & (df.text_caption.str.len() > 0)]
    cv = CountVectorizer(min_df=5, ngram_range=(1, 2))
    X = cv.fit_transform(tr.text_caption)
    vocab = np.array(cv.get_feature_names_out())
    tot = np.asarray(X.sum(0)).ravel(); N = tot.sum()

    kelas = ["Surprise", "Trust", "Proud", "Joy"]
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.6), gridspec_kw={"wspace": 0.55})
    for ax, e, warna in zip(axes, kelas, C):
        m = (tr.emotion == e).values
        c = np.asarray(X[m].sum(0)).ravel(); n = c.sum()
        p_in = (c + 1) / (n + X.shape[1]); p_out = (tot - c + 1) / (N - n + X.shape[1])
        skor = np.log(p_in / p_out) * np.sqrt(c)
        top = np.argsort(-skor)[:7][::-1]
        ax.barh(range(7), skor[top], color=warna, height=0.62, zorder=3)
        ax.set_yticks(range(7)); ax.set_yticklabels(vocab[top], fontsize=8.5)
        ax.set_title(f"{e}  (n={m.sum()})", fontsize=10.5)
        ax.set_xlabel("kekhasan", fontsize=8.5)
        ax.tick_params(axis="x", labelsize=8)
        rapikan(ax, grid_x=True)
    fig.suptitle("Kata pembeda adalah kata TOPIK, bukan kata emosi — "
                 "anotator melabeli jenis konten", fontsize=12, fontweight="bold", y=1.06)
    fig.text(0.5, -0.06, "Trust = konten kebugaran  ·  Proud = prestasi olahraga  ·  "
             "Surprise = otomotif  ·  Joy = liputan acara",
             ha="center", fontsize=9, color=TEKS2)
    simpan(fig, "03_kata_pembeda")


def fig_ablation() -> None:
    """Dua panel, BUKAN sumbu ganda: akurasi (%) dan macro-F1 (0-1) beda skala."""
    ab = pd.read_csv(OUTPUTS_DIR / "ablation.csv")
    nama = [t.split(". ", 1)[1] if ". " in t else t for t in ab.tahap]
    nama = [n.replace("teks saja (TF-IDF+LogReg), ", "teks saja ") for n in nama]
    y = np.arange(len(ab))
    final = len(ab) - 2                      # bobot^0.5 = model final

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 4.8), sharey=True,
                                 gridspec_kw={"wspace": 0.06})
    for ax, kol, judul, skala, fmt in (
            (a1, "accuracy", "Akurasi (%)", 100, "{:.1f}%"),
            (a2, "macro-F1", "macro-F1", 1, "{:.3f}")):
        kunci = "accuracy" if skala == 100 else "macro_f1"
        v = ab[kunci].values * skala
        garis = v[0]                       # baris pertama = baseline
        # Satu warna untuk model final, sisanya redup. Warna TIDAK dipakai
        # untuk mengulang informasi panjang batang.
        warna = [C[0] if i == final else NETRAL for i in range(len(v))]
        ax.barh(y, v, color=warna, height=0.62, zorder=3)
        ax.axvline(garis, color=C[1], lw=1.3, zorder=4)
        for i, x in enumerate(v):
            ax.text(x + v.max() * 0.02, i, fmt.format(x), va="center", fontsize=8.5,
                    color=TEKS if i == final else TEKS2,
                    fontweight="bold" if i == final else "normal")
        ax.set_title(judul, fontsize=11, pad=22)
        ax.set_xlim(0, v.max() * 1.24)
        # Label baseline di dalam area plot bagian atas: tidak menabrak judul
        # panel sebelahnya (masalah versi pertama) maupun label sumbu-X di bawah.
        ax.text(garis, -0.72, "baseline", fontsize=8.5, color=C[1],
                ha="center", va="bottom")
        rapikan(ax, grid_x=True)
    a1.set_yticks(y); a1.set_yticklabels(nama, fontsize=8.5)
    a1.invert_yaxis()
    fig.suptitle("Model final menang di kedua metrik — dan fitur numerik justru merugikan",
                 fontsize=12.5, fontweight="bold", y=1.04)
    simpan(fig, "04_ablation")


def fig_confusion() -> None:
    """Magnitudo -> satu rona terang->gelap. Dinormalisasi per baris supaya
    kelas kecil tidak tenggelam oleh Surprise."""
    oof = pd.read_csv(OUTPUTS_DIR / "oof_terbaik.csv")
    from sklearn.metrics import confusion_matrix
    urut = oof.y.value_counts().index.tolist()
    M = confusion_matrix(oof.y, oof.oof, labels=urut)
    Mn = M / np.maximum(M.sum(1, keepdims=True), 1)

    fig, ax = plt.subplots(figsize=(6.8, 5.6))
    im = ax.imshow(Mn, cmap=RAMP, vmin=0, vmax=1)
    for i in range(len(urut)):
        for j in range(len(urut)):
            if M[i, j]:
                ax.text(j, i, M[i, j], ha="center", va="center", fontsize=8,
                        color="white" if Mn[i, j] > 0.55 else TEKS2)
    ax.set_xticks(range(len(urut))); ax.set_xticklabels(urut, rotation=45, ha="right")
    ax.set_yticks(range(len(urut))); ax.set_yticklabels(urut)
    ax.set_xlabel("prediksi"); ax.set_ylabel("label sebenarnya")
    ax.set_title("Model condong ke Surprise; kelas kecil belum tertangkap")
    for s in ax.spines.values():
        s.set_visible(False)
    cb = fig.colorbar(im, ax=ax, fraction=0.045, pad=0.03)
    cb.set_label("proporsi per baris", fontsize=8.5, color=TEKS2)
    cb.outline.set_visible(False)
    simpan(fig, "05_confusion_matrix")


def fig_plafon() -> None:
    """Menempatkan hasil model terhadap plafon derau label."""
    fig, ax = plt.subplots(figsize=(8.4, 3.4))
    item = [("Tebak «Surprise» selalu", 41.2, NETRAL),
            ("Model final (teks + transkrip)", 42.0, C[0]),
            ("Plafon: kesepakatan anotator\npada video yang SAMA PERSIS", 56.1, C[1])]
    y = np.arange(len(item))
    ax.barh(y, [v for _, v, _ in item], color=[c for _, _, c in item],
            height=0.55, zorder=3)
    for i, (_, v, _) in enumerate(item):
        ax.text(v + 0.9, i, f"{v:.1f}%", va="center", fontsize=9.5, color=TEKS2)
    ax.set_yticks(y); ax.set_yticklabels([n for n, _, _ in item], fontsize=9)
    ax.invert_yaxis()
    ax.set_xlim(0, 68); ax.set_xlabel("akurasi")
    ax.set_title("Ruang perbaikan hanya ~15 poin — dibatasi ketidaksepakatan anotator")
    ax.annotate("", xy=(56.1, 2.48), xytext=(41.2, 2.48),
                arrowprops=dict(arrowstyle="<->", color=TEKS2, lw=1.1))
    ax.text(48.6, 2.72, "seluruh ruang yang tersedia", fontsize=8.5,
            color=TEKS2, ha="center")
    ax.set_ylim(2.95, -0.6)
    rapikan(ax, grid_x=True)
    fig.text(0.5, -0.10, "Satu video dilabeli Trust 7x, Proud 4x, Surprise 3x, Fear 1x, Neutral 1x "
             "— model tidak mungkin melampaui ketidaksepakatan ini.",
             ha="center", fontsize=8.5, color=TEKS2)
    simpan(fig, "06_plafon_derau_label")


def main() -> int:
    gaya()
    df = pd.read_parquet(FEATURES_DIR / "dataset.parquet")
    print(f"membuat grafik ke {FIGDIR} ...")
    fig_distribusi_label(df)
    fig_cakupan(df)
    fig_kata_pembeda(df)
    fig_ablation()
    fig_confusion()
    fig_plafon()
    print("selesai.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
