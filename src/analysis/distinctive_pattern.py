import pandas as pd
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parents[2]

WORD_FILE = BASE_DIR / "analysis" / "tfidf_by_emotion_stopwords_removed.csv"
BIGRAM_FILE = BASE_DIR / "analysis" / "bigram_tfidf.csv"
TRIGRAM_FILE = BASE_DIR / "analysis" / "trigram_tfidf.csv"

OUTPUT_DIR = BASE_DIR / "analysis" / "distinctive_patterns"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def calculate_distinctiveness(df, text_col):
    pivot = df.pivot_table(
        index=text_col,
        columns="emotion",
        values="mean_tfidf",
        aggfunc="mean",
        fill_value=0
    )

    results = []

    for emotion in pivot.columns:
        other_emotions = [e for e in pivot.columns if e != emotion]

        pivot["other_mean"] = pivot[other_emotions].mean(axis=1)

        pivot["distinctiveness"] = (
            pivot[emotion] - pivot["other_mean"]
        )

        temp = pivot[[emotion, "other_mean", "distinctiveness"]].copy()
        temp["emotion"] = emotion
        temp[text_col] = temp.index

        temp = temp.sort_values(
            "distinctiveness",
            ascending=False
        )

        results.append(temp.reset_index(drop=True))

    return pd.concat(results, ignore_index=True)


def process_file(input_file, output_file, text_col):
    df = pd.read_csv(input_file)

    result = calculate_distinctiveness(df, text_col)

    result = result[
        [
            "emotion",
            text_col,
            result.columns[1],
            "distinctiveness"
        ]
    ]

    result.to_csv(output_file, index=False)


process_file(
    WORD_FILE,
    OUTPUT_DIR / "distinctive_words.csv",
    "word"
)

process_file(
    BIGRAM_FILE,
    OUTPUT_DIR / "distinctive_bigrams.csv",
    "phrase"
)

process_file(
    TRIGRAM_FILE,
    OUTPUT_DIR / "distinctive_trigrams.csv",
    "phrase"
)

print("Distinctive pattern analysis selesai.")