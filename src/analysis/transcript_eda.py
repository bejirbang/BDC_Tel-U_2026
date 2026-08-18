from __future__ import annotations

from collections import Counter
from pathlib import Path

import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

PROCESSED_DIR = (
    PROJECT_ROOT
    / "data"
    / "2_processed"
    / "transcripts"
)

OUTPUT_DIR = (
    PROJECT_ROOT
    / "data"
    / "2_processed"
    / "eda"
)


def load_transcripts() -> pd.DataFrame:

    records = []

    if not PROCESSED_DIR.exists():
        raise FileNotFoundError(
            f"Processed transcript directory tidak ditemukan: "
            f"{PROCESSED_DIR}"
        )

    for emotion_dir in sorted(PROCESSED_DIR.iterdir()):

        if not emotion_dir.is_dir():
            continue

        emotion = emotion_dir.name

        transcript_files = sorted(
            emotion_dir.glob("*_processed.txt")
        )

        for transcript_file in transcript_files:

            try:
                text = transcript_file.read_text(
                    encoding="utf-8"
                ).strip()

                words = text.split()

                records.append(
                    {
                        "emotion": emotion,
                        "file_name": transcript_file.name,
                        "text": text,
                        "word_count": len(words),
                    }
                )

            except Exception as exc:

                print(
                    f"[ERROR] "
                    f"{emotion}/{transcript_file.name}: "
                    f"{exc}"
                )

    return pd.DataFrame(records)



def print_dataset_summary(df: pd.DataFrame) -> None:
    """Menampilkan ringkasan dataset."""

    print("\n" + "=" * 70)
    print("DATASET SUMMARY")
    print("=" * 70)

    print(f"Total transcript : {len(df)}")

    if df.empty:
        print("Dataset kosong.")
        return

    empty_count = (df["text"].str.strip() == "").sum()

    print(f"Transcript kosong: {empty_count}")

    print("\nJumlah transcript per emotion:")
    print(
        df["emotion"]
        .value_counts()
        .sort_index()
        .to_string()
    )


def analyze_text_length(df: pd.DataFrame) -> pd.DataFrame:


    if df.empty:
        return pd.DataFrame()

    summary = (
        df.groupby("emotion")["word_count"]
        .agg(
            transcript_count="count",
            total_words="sum",
            average_words="mean",
            median_words="median",
            min_words="min",
            max_words="max",
        )
        .round(2)
        .sort_index()
    )

    return summary


def print_text_length_analysis(
    summary: pd.DataFrame,
) -> None:

    print("\n" + "=" * 70)
    print("TEXT LENGTH BY EMOTION")
    print("=" * 70)

    if summary.empty:
        print("Tidak ada data.")
        return

    print(summary.to_string())


def get_top_words(
    df: pd.DataFrame,
    top_n: int = 20,
) -> dict[str, list[tuple[str, int]]]:

    results = {}

    for emotion in sorted(df["emotion"].unique()):

        emotion_df = df[
            df["emotion"] == emotion
        ]

        counter = Counter()

        for text in emotion_df["text"]:

            words = text.split()

            counter.update(words)

        results[emotion] = counter.most_common(top_n)

    return results


def print_top_words(
    top_words: dict[str, list[tuple[str, int]]],
) -> None:

    print("\n" + "=" * 70)
    print("TOP WORDS BY EMOTION")
    print("=" * 70)

    for emotion, words in top_words.items():

        print(f"\n[{emotion}]")

        if not words:
            print("Tidak ada kata.")
            continue

        for word, count in words:

            print(
                f"{word:<25} {count}"
            )


def calculate_vocabulary(
    df: pd.DataFrame,
) -> pd.DataFrame:

    records = []

    for emotion in sorted(df["emotion"].unique()):

        emotion_df = df[
            df["emotion"] == emotion
        ]

        vocabulary = set()

        for text in emotion_df["text"]:

            vocabulary.update(
                text.split()
            )

        records.append(
            {
                "emotion": emotion,
                "vocabulary_size": len(vocabulary),
            }
        )

    return pd.DataFrame(records)


def print_vocabulary(
    vocabulary_df: pd.DataFrame,
) -> None:

    print("\n" + "=" * 70)
    print("VOCABULARY SIZE")
    print("=" * 70)

    if vocabulary_df.empty:
        print("Tidak ada data.")
        return

    print(
        vocabulary_df
        .sort_values(
            "vocabulary_size",
            ascending=False,
        )
        .to_string(index=False)
    )


def save_results(
    df: pd.DataFrame,
    length_summary: pd.DataFrame,
    vocabulary_df: pd.DataFrame,
) -> None:

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    df.to_csv(
        OUTPUT_DIR / "transcript_dataset.csv",
        index=False,
        encoding="utf-8",
    )

    length_summary.to_csv(
        OUTPUT_DIR / "text_length_by_emotion.csv",
        encoding="utf-8",
    )

    vocabulary_df.to_csv(
        OUTPUT_DIR / "vocabulary_by_emotion.csv",
        index=False,
        encoding="utf-8",
    )

    print("\n" + "=" * 70)
    print("HASIL DISIMPAN")
    print("=" * 70)

    print(f"Output directory: {OUTPUT_DIR}")


def main() -> None:

    print("=" * 70)
    print("TRANSCRIPT EDA")
    print("=" * 70)

    df = load_transcripts()

    print_dataset_summary(df)

    if df.empty:
        print("\nTidak ada transcript untuk dianalisis.")
        return

    length_summary = analyze_text_length(df)

    print_text_length_analysis(
        length_summary
    )

    top_words = get_top_words(
        df,
        top_n=20,
    )

    print_top_words(
        top_words
    )

    vocabulary_df = calculate_vocabulary(
        df
    )

    print_vocabulary(
        vocabulary_df
    )

    save_results(
        df,
        length_summary,
        vocabulary_df,
    )

    print("\nEDA selesai.")


if __name__ == "__main__":
    main()