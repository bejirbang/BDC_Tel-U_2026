"""
Fitur teks dari caption, hashtag, dan komentar.

Menghasilkan DUA hal berbeda, dan pembedaannya disengaja:

1. **Fitur numerik yang bisa dijelaskan** - panjang caption, jumlah emoji,
   jumlah tanda seru, skor leksikon. Ini yang nanti muncul di grafik SHAP dan
   bisa diceritakan ke juri ("caption Surprise rata-rata punya 2x lebih banyak
   tanda seru").

2. **Kolom teks mentah yang sudah dibersihkan** (`text_caption`, `text_all`).
   TF-IDF sengaja TIDAK dihitung di sini, melainkan di dalam pipeline CV pada
   `train.py`. Alasannya: kalau vectorizer di-fit pada seluruh data train lalu
   dipakai di semua fold, kosakata dari fold validasi ikut membentuk fitur dan
   skor CV jadi optimistis. Memasangnya di dalam Pipeline membuat tiap fold
   punya vectorizer sendiri - bocornya nol.

Catatan soal "kata yang sering muncul": frekuensi mentah bukan sinyal. Kata
terbanyak di caption Bahasa Indonesia pasti "yang", "di", "ini" - muncul merata
di semua emosi. Yang berguna adalah kata yang MEMBEDAKAN, dan itu justru yang
ditonjolkan TF-IDF karena kata yang ada di mana-mana ditekan bobotnya.

Jalankan:  python src/features_text.py
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import describe, load_meta, load_rows, save_features  # noqa: E402

# --------------------------------------------------------------------------
# Emoji - pakai rentang unicode, tidak perlu paket tambahan
# --------------------------------------------------------------------------

RE_EMOJI = re.compile(
    "[\U0001F300-\U0001F5FF\U0001F600-\U0001F64F\U0001F680-\U0001F6FF"
    "\U0001F900-\U0001F9FF\U0001FA70-\U0001FAFF\U00002600-\U000026FF"
    "\U00002700-\U000027BF\U0001F1E0-\U0001F1FF]"
)
EMOJI_POS = set("😀😃😄😁😆😊🙂😍🥰😘🤩🥳😎👍👏🙌💪❤️❤🧡💛💚💙💜🔥✨⭐🎉🎊🏆🥇💯🤝🙏")
EMOJI_NEG = set("😢😭😞😔😟😩😫😤😠😡🤬😱😨😰😥💔😬🥺😖😣⚠️⚠❌")
EMOJI_WOW = set("😮😲😯🤯😳🤔👀‼️❗❓⁉️")

# --------------------------------------------------------------------------
# Leksikon emosi Bahasa Indonesia
#
# SUDAH DIUJI, DAN HASILNYA LEMAH. Dari 5 kelas besar, hanya `lex_anger` yang
# menempatkan kelasnya sendiri di peringkat 1; sisanya peringkat 2-4.
#
# Sebabnya ketahuan dari analisis log-odds: anotator memberi label berdasarkan
# JENIS KONTEN, bukan kata emosi di caption. Kata paling khas untuk Trust adalah
# "dumbbell/gym" (konten kebugaran), untuk Proud "emas/meraih" (prestasi
# olahraga), untuk Surprise "roda/bensin/harga" (otomotif). Tidak satupun kata
# emosi.
#
# Tetap dipertahankan karena hasil negatif ini bahan laporan yang bagus, dan
# biayanya nol. Jangan berharap banyak darinya - sinyal sesungguhnya ada di
# TF-IDF/embedding yang menangkap topik.
# --------------------------------------------------------------------------

LEKSIKON = {
    "lex_surprise": ["kaget", "nggak nyangka", "gak nyangka", "ternyata", "wow", "gila",
                     "parah", "mendadak", "tiba-tiba", "wah", "astaga", "ajaib", "spek dewa"],
    "lex_trust": ["resmi", "garansi", "terpercaya", "aman", "terjamin", "sertifikat",
                  "original", "asli", "legal", "kualitas", "standar", "rekomendasi"],
    "lex_proud": ["bangga", "prestasi", "juara", "terbaik", "nomor satu", "pertama di",
                  "penghargaan", "sukses", "karya anak bangsa", "membanggakan"],
    "lex_joy": ["senang", "seru", "asik", "asyik", "happy", "gembira", "bahagia",
                "mantap", "keren", "enak", "puas"],
    "lex_anger": ["kecewa", "marah", "kesal", "parah banget", "buruk", "jelek",
                  "protes", "komplain", "zonk", "nipu", "tipu"],
    "lex_sad": ["sedih", "duka", "kehilangan", "turut berduka", "prihatin", "malang"],
    "lex_fear": ["bahaya", "hati-hati", "waspada", "awas", "risiko", "takut",
                 "celaka", "kecelakaan", "rusak"],
    "lex_loyalty": ["setia", "langganan", "sejak", "bertahun", "keluarga besar",
                    "komunitas", "sahabat", "loyal"],
}


def hitung_leksikon(teks: str) -> dict[str, int]:
    t = teks.lower()
    return {k: sum(t.count(w) for w in kata) for k, kata in LEKSIKON.items()}


# --------------------------------------------------------------------------
# Fitur KONSEP - berbeda dari LEKSIKON di atas, dan jauh lebih berguna.
#
# LEKSIKON memetakan kata langsung ke EMOSI (`bangga` -> Proud) dan terbukti
# gagal. KONSEP memetakan kata ke IDE (`ditinggal` -> kehilangan), lalu biar
# classifier yang belajar ide mana menandakan emosi mana. Lapisan perantara itu
# yang membuatnya bekerja: satu konsep bisa muncul lewat kata yang berbeda-beda
# di topik yang berbeda-beda.
#
# Contoh pemicunya: video berlabel Sad tentang mobil yang ditinggal di Laos,
# dengan tangki bahan bakar "gak kepake". Tidak ada satu pun kata emosi di sana,
# tapi idenya jelas - kehilangan.
#
# CATATAN PENTING soal panjang daftar: versi dengan daftar kata JAUH LEBIH
# PANJANG (hasil membaca puluhan transkrip) justru berskor LEBIH BURUK -
# macro-F1 sama tapi akurasi turun 1 poin. Sebabnya kata umum seperti
# `indonesia`, `yuk`, `cara`, `penting` muncul di semua kelas dan hanya
# menambah derau. Presisi mengalahkan cakupan; jangan panjangkan daftar ini
# tanpa mengukur ulang.
# --------------------------------------------------------------------------

KONSEP = {
    "kon_kehilangan":  ["ditinggal", "kita tinggal", "hilang", "gak kepake", "sia-sia"],
    "kon_kerusakan":   ["rusak", "pecah", "luka", "mati", "parah"],
    "kon_kegagalan":   ["kalah", "gagal", "salah", "masalah"],
    "kon_kabar_buruk": ["hati-hati", "bahaya", "risiko", "awas"],
    "kon_kesedihan":   ["sedih", "duka", "prihatin"],
    "kon_pencapaian":  ["pemenang", "menang", "juara", "prestasi", "meraih", "sukses", "emas"],
    "kon_selamat":     ["selamat kepada", "bangga", "apresiasi"],
    "kon_kepercayaan": ["resmi", "garansi", "kualitas", "aman"],
    "kon_kegembiraan": ["seru", "happy", "senang", "keren", "mantap"],

}

# Sembilan konsep di atas yang dipakai MODEL FINAL. Yang di bawah dihitung dan
# disimpan untuk analisis laporan, tapi TIDAK masuk model - lihat §23.
KONSEP_FINAL = list(KONSEP)

KONSEP_UJI = {
    # --------------------------------------------------------------------
    # Konsep gelombang kedua (§23), dirumuskan dari aturan pembeda manusia
    # setelah menonton video-video yang paling sering tertukar. Sasarannya
    # spesifik: segitiga Trust/Proud/Surprise yang memuat 83% data dan yang
    # ANOTATORNYA SENDIRI tertukar di situ (§22).
    #
    # Aturannya:
    #   Trust    = menonjolkan informasi/pengetahuan yang bisa dipercaya,
    #              baik dengan menyampaikan fakta maupun menunjukkan caranya
    #   Proud    = menggambarkan pencapaian atau spesifikasi
    #   Surprise = produk baru yang lebih bagus dari yang lama; kemegahan
    #
    # Dipecah jadi sub-konsep, bukan tiga fitur gemuk, supaya kalau salah satu
    # sisi ternyata derau ia bisa dibuang tanpa membuang sisi yang bekerja.
    # Daftarnya sengaja pendek - lihat CATATAN PENTING di atas.
    #
    # HASIL: DITOLAK dari model final. Arah pembedanya benar untuk 4 dari 6
    # (kon_mengajari terkuat: Trust 0,918 lawan Surprise 0,420), tapi pada CV
    # akurasi kalah di 5 dari 5 seed (-1,1 poin) dan macro-F1 cuma +0,004.
    # Sebabnya: 17 dari 20 katanya SUDAH ADA di kosakata TF-IDF (`tips` 52 dok,
    # `fitur` 45, `harga` 149), jadi fitur ini cuma mengulang informasi yang
    # sudah dipunyai - sebagai jumlah kasar yang malah membuang pembobotan
    # per-kata. Bandingkan §17 yang berhasil karena menangkap FRASA langka
    # ("kita tinggal", 1 video) yang mustahil dilihat TF-IDF unigram.
    # --------------------------------------------------------------------

    # Trust-a: mengajari / menunjukkan cara. Contoh: id 513 (tips servis tenis
    # untuk pemula), id 487 (cara pakai dumbbell yang benar).
    # `cara` sendirian TIDAK dipakai - §17 mencatatnya muncul di semua kelas.
    "kon_mengajari": ["tips", "caranya", "cara pake", "cara buat", "tutorial",
                      "pemula", "newbie", "beginner", "langkah"],

    # Trust-b: menjelaskan sebab / mendiagnosis. Contoh: id 652 (kenapa ikan
    # punya tonjolan), id 174 (data jarak tempuh yang sudah dicapai tim).
    # `ternyata` sengaja TIDAK dipakai: muncul kuat juga di Proud (id 357).
    "kon_menjelaskan": ["terlihat adanya", "penyebabnya", "sebenarnya",
                        "untuk info", "faktanya", "artinya", "disebabkan"],

    # Proud-a: pencapaian. Melengkapi `kon_pencapaian` dengan bentuk yang lebih
    # spesifik dan tidak tumpang tindih dengannya.
    "kon_prestasi": ["juara dunia", "rekor", "medali", "podium", "peringkat"],

    # Proud-b: spesifikasi / varian tertinggi. Contoh: id 357 (174 cc, replica
    # motor juara dunia), id 165 (varian paling tinggi, lebih canggih).
    "kon_spesifikasi": ["varian", "spesifikasi", "fitur", "tipe tertinggi",
                        "paling tinggi", "canggih"],

    # Surprise-a: produk baru menggantikan yang lama.
    "kon_produk_baru": ["terbaru", "meluncur", "diluncurkan", "rilis",
                        "launching", "all new", "generasi baru", "model baru"],

    # Surprise-b: kemegahan + pembukaan harga. §5A menemukan kata paling khas
    # Surprise adalah `roda, bensin, mobilbaru, harga` - dua terakhir persis
    # pola ini.
    "kon_kemegahan": ["mewah", "megah", "gahar", "sultan", "termahal",
                      "harganya", "miliar"],
}


# Dikompilasi sekali sebagai satu alternation per konsep.
#
# WAJIB regex, JANGAN menjumlahkan `str.count()` tiap kata secara terpisah:
# kata yang bersarang akan terhitung dua kali. `"masalah"` akan dihitung
# sebagai `salah` DAN `masalah`; `"pemenang"` sebagai `menang` DAN `pemenang`.
# Bug ini sempat menggelembungkan kon_kegagalan dari 437 jadi 566 dan
# menurunkan macro-F1 dari 0,135 ke 0,117. Regex alternation mencocokkan
# tanpa tumpang tindih, jadi tiap kemunculan dihitung tepat sekali.
_SEMUA_KONSEP = {**KONSEP, **KONSEP_UJI}
_POLA_KONSEP = {
    k: re.compile("|".join(re.escape(w) for w in kata), re.I)
    for k, kata in _SEMUA_KONSEP.items()
}


def hitung_konsep(teks: str) -> dict[str, int]:
    return {k: len(p.findall(teks or "")) for k, p in _POLA_KONSEP.items()}


def bersihkan(t: str) -> str:
    """Normalisasi ringan. Sengaja TIDAK membuang emoji atau tanda baca dari
    `text_all` - keduanya sinyal emosi. Yang dibuang hanya URL dan spasi ganda."""
    t = re.sub(r"https?://\S+", " ", t)
    t = re.sub(r"\s+", " ", t)
    return t.strip()


def build() -> pd.DataFrame:
    rows = load_rows()
    meta = load_meta()

    recs = []
    for r in rows.itertuples(index=False):
        m = meta.get(r.key, {})
        cap = (m.get("caption") or "").strip()
        tags = m.get("hashtags") or []
        mentions = m.get("mentions") or []
        comments = m.get("comments") or []

        teks_komen = " ".join((c.get("text") or "") for c in comments)
        komen_len = [len(c.get("text") or "") for c in comments]
        komen_like = [c.get("like_count") or 0 for c in comments]

        emoji = RE_EMOJI.findall(cap)
        kata = cap.split()
        huruf = [c for c in cap if c.isalpha()]

        rec = {
            "split": r.split, "id": r.id, "key": r.key, "emotion": r.emotion,

            # --- teks mentah untuk TF-IDF di train.py (bukan fitur numerik) ---
            "text_caption": bersihkan(cap),
            "text_all": bersihkan(" ".join([cap, " ".join(tags), teks_komen])),

            # --- bentuk & panjang ---
            "cap_len": len(cap),
            "cap_words": len(kata),
            "cap_avg_word_len": float(np.mean([len(w) for w in kata])) if kata else np.nan,
            "cap_lines": cap.count("\n"),
            "cap_has_url": int(bool(re.search(r"https?://", cap))),

            # --- gaya penulisan: penanda intensitas emosi ---
            "n_exclaim": cap.count("!"),
            "n_question": cap.count("?"),
            "n_ellipsis": cap.count("..."),
            "upper_ratio": (sum(c.isupper() for c in huruf) / len(huruf)) if huruf else np.nan,
            "digit_ratio": (sum(c.isdigit() for c in cap) / len(cap)) if cap else np.nan,

            # --- emoji ---
            "n_emoji": len(emoji),
            "n_emoji_pos": sum(e in EMOJI_POS for e in emoji),
            "n_emoji_neg": sum(e in EMOJI_NEG for e in emoji),
            "n_emoji_wow": sum(e in EMOJI_WOW for e in emoji),
            "emoji_per_word": (len(emoji) / len(kata)) if kata else np.nan,

            # --- hashtag & mention ---
            "n_hashtag": len(tags),
            "n_mention": len(mentions),
            "hashtag_avg_len": float(np.mean([len(t) for t in tags])) if tags else np.nan,

            # --- komentar: reaksi penonton, bukan niat pembuat konten ---
            "n_comments_captured": len(comments),
            "comment_avg_len": float(np.mean(komen_len)) if komen_len else np.nan,
            "comment_like_sum": int(sum(komen_like)),
            "comment_like_max": int(max(komen_like)) if komen_like else np.nan,
        }
        rec.update(hitung_leksikon(cap + " " + teks_komen))
        recs.append(rec)

    df = pd.DataFrame(recs)
    df["lex_total"] = df[[c for c in df.columns if c.startswith("lex_")]].sum(axis=1)
    return df


if __name__ == "__main__":
    df = build()
    save_features(df, "text")
    describe(df, "text")

    ada = df["cap_len"] > 0
    print(f"\nbaris dengan caption: {ada.sum()} / {len(df)} ({ada.mean() * 100:.1f}%)")

    tr = df[(df["split"] == "train") & ada]
    if len(tr):
        print("\nrata-rata per emosi (hanya baris yang punya caption):")
        kol = ["cap_len", "n_exclaim", "n_emoji", "n_hashtag", "lex_total"]
        ring = tr.groupby("emotion")[kol].mean().round(2)
        ring["n"] = tr.groupby("emotion").size()
        print(ring.sort_values("n", ascending=False).to_string())
