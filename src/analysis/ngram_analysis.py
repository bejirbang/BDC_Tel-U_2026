from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "2_processed" / "transcripts"
OUTPUT_DIR = PROJECT_ROOT / "analysis"


STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu",
    "ada", "untuk", "dengan", "atau", "pada", "dalam",
    "jadi", "akan", "sudah", "udah", "telah", "bisa",
    "dapat", "tidak", "gak", "ga", "nggak", "enggak",
    "ya", "iya", "saya", "aku", "kami", "kita",
    "kamu", "lu", "lo", "gue", "dia", "mereka",
    "apa", "siapa", "mana", "bagaimana", "kenapa",
    "kalau", "kalo", "karena", "tapi", "namun",
    "juga", "lagi", "lebih", "sangat", "banget",
    "sama", "seperti", "kayak", "gitu", "begitu",
    "tuh", "nih", "kan", "kok", "sih", "dong",
    "deh", "lah", "pun", "saja", "aja",
    "masih", "belum", "pernah",
    "harus", "mau", "ingin", "boleh",
    "buat", "pakai", "pake", "terus",
    "dulu", "tadi", "nanti", "sekarang",
    "ketika", "saat", "setiap", "sebuah",
    "seorang", "orang", "hal", "nya",
}


def load_transcripts() -> pd.DataFrame:
    records = []

    if not INPUT_DIR.exists():
        raise FileNotFoundError(
            f"Directory transcript tidak ditemukan: {INPUT_DIR}"
        )

    for emotion_dir in sorted(INPUT_DIR.iterdir()):
        if not emotion_dir.is_dir():
            continue

        emotion = emotion_dir.name

        for transcript_file in sorted(
            emotion_dir.glob("*_processed.txt")
        ):
            text = transcript_file.read_text(
                encoding="utf-8"
            ).strip()

            if not text:
                continue

            records.append(
                {
                    "emotion": emotion,
                    "file_name": transcript_file.name,
                    "text": text,
                }
            )

    return pd.DataFrame(records)


def remove_stopwords(text: str) -> str:
    return " ".join(
        word
        for word in text.split()
        if word not in STOPWORDS
    )


def calculate_ngram_tfidf(
    df: pd.DataFrame,
    ngram_range: tuple[int, int],
    top_n: int = 30,
) -> pd.DataFrame:

    texts = df["text"].apply(remove_stopwords)

    vectorizer = TfidfVectorizer(
        lowercase=False,
        ngram_range=ngram_range,
        token_pattern=r"(?u)\b\w+\b",
        min_df=2,
    )

    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()

    records = []

    for emotion in sorted(df["emotion"].unique()):
        indices = df.index[
            df["emotion"] == emotion
        ].tolist()

        emotion_matrix = tfidf_matrix[indices]

        mean_scores = emotion_matrix.mean(
            axis=0
        ).A1

        top_indices = mean_scores.argsort()[
            -top_n:
        ][::-1]

        for rank, index in enumerate(
            top_indices,
            start=1,
        ):
            score = mean_scores[index]

            if score <= 0:
                continue

            records.append(
                {
                    "emotion": emotion,
                    "rank": rank,
                    "phrase": feature_names[index],
                    "mean_tfidf": round(
                        float(score),
                        6,
                    ),
                }
            )

    return pd.DataFrame(records)


def save_results(
    df: pd.DataFrame,
    filename: str,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    output_path = OUTPUT_DIR / filename

    df.to_csv(
        output_path,
        index=False,
        encoding="utf-8",
    )

    for emotion in sorted(df["emotion"].unique()):
        emotion_dir = OUTPUT_DIR / emotion
        emotion_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        emotion_df = df[
            df["emotion"] == emotion
        ]

        emotion_df.to_csv(
            emotion_dir / filename,
            index=False,
            encoding="utf-8",
        )


def print_results(
    df: pd.DataFrame,
    title: str,
) -> None:

    print("\n" + "=" * 60)
    print(title)
    print("=" * 60)

    for emotion in sorted(df["emotion"].unique()):
        print(f"\n[{emotion}]")

        emotion_df = df[
            df["emotion"] == emotion
        ]

        for _, row in emotion_df.iterrows():
            print(
                f"{row['rank']:>2}. "
                f"{row['phrase']:<30} "
                f"{row['mean_tfidf']:.6f}"
            )


def main() -> None:

    print("=" * 60)
    print("N-GRAM TF-IDF ANALYSIS")
    print("=" * 60)

    df = load_transcripts()

    if df.empty:
        print("Tidak ada transcript yang ditemukan.")
        return

    print(
        f"\nTotal transcript: {len(df)}"
    )

    bigram_df = calculate_ngram_tfidf(
        df,
        ngram_range=(2, 2),
        top_n=30,
    )

    trigram_df = calculate_ngram_tfidf(
        df,
        ngram_range=(3, 3),
        top_n=30,
    )

    print_results(
        bigram_df,
        "BIGRAM TF-IDF",
    )

    print_results(
        trigram_df,
        "TRIGRAM TF-IDF",
    )

    save_results(
        bigram_df,
        "bigram_tfidf.csv",
    )

    save_results(
        trigram_df,
        "trigram_tfidf.csv",
    )

    print("\nAnalysis selesai.")


if __name__ == "__main__":
    main()