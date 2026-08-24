from __future__ import annotations

from pathlib import Path

import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer


PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_DIR = PROJECT_ROOT / "data" / "2_processed" / "transcripts"
OUTPUT_DIR = PROJECT_ROOT / "analysis"


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
            try:
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

            except Exception as exc:
                print(
                    f"[ERROR] {emotion}/{transcript_file.name}: {exc}"
                )

    return pd.DataFrame(records)


def calculate_tfidf(
    df: pd.DataFrame,
    top_n: int = 30,
) -> pd.DataFrame:

    vectorizer = TfidfVectorizer(
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b",
    )

    tfidf_matrix = vectorizer.fit_transform(df["text"])
    feature_names = vectorizer.get_feature_names_out()

    records = []

    for emotion in sorted(df["emotion"].unique()):

        indices = df.index[
            df["emotion"] == emotion
        ].tolist()

        emotion_matrix = tfidf_matrix[indices]
        mean_scores = emotion_matrix.mean(axis=0).A1

        top_indices = mean_scores.argsort()[-top_n:][::-1]

        for rank, index in enumerate(top_indices, start=1):
            score = mean_scores[index]

            if score <= 0:
                continue

            records.append(
                {
                    "emotion": emotion,
                    "rank": rank,
                    "word": feature_names[index],
                    "mean_tfidf": round(float(score), 6),
                }
            )

    return pd.DataFrame(records)


def save_results(df: pd.DataFrame) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_DIR / "tfidf_by_emotion.csv",
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
            emotion_dir / "tfidf_top_words.csv",
            index=False,
            encoding="utf-8",
        )


def print_results(df: pd.DataFrame) -> None:

    print("\n" + "=" * 60)
    print("TOP TF-IDF WORDS BY EMOTION")
    print("=" * 60)

    for emotion in sorted(df["emotion"].unique()):

        print(f"\n[{emotion}]")

        emotion_df = df[
            df["emotion"] == emotion
        ]

        for _, row in emotion_df.iterrows():
            print(
                f"{row['rank']:>2}. "
                f"{row['word']:<20} "
                f"{row['mean_tfidf']:.6f}"
            )


def main() -> None:

    print("=" * 60)
    print("TF-IDF ANALYSIS")
    print("=" * 60)

    df = load_transcripts()

    if df.empty:
        print("Tidak ada transcript yang ditemukan.")
        return

    print(f"\nTotal transcript: {len(df)}")

    tfidf_df = calculate_tfidf(df)

    print_results(tfidf_df)
    save_results(tfidf_df)

    print("\nTF-IDF analysis selesai.")


if __name__ == "__main__":
    main()