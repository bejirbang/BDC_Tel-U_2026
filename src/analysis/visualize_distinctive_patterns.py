import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE_DIR / "analysis" / "distinctive_patterns"
OUTPUT_DIR = BASE_DIR / "analysis" / "distinctive_patterns" / "plots"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def visualize_patterns(file_name, text_col, pattern_type, top_n=10):

    input_file = INPUT_DIR / file_name

    df = pd.read_csv(input_file)

    emotions = df["emotion"].unique()

    for emotion in emotions:

        emotion_df = df[df["emotion"] == emotion].copy()

        emotion_df = emotion_df[
            emotion_df["distinctiveness"] > 0
        ]

        emotion_df = emotion_df.sort_values(
            "distinctiveness",
            ascending=False
        ).head(top_n)

        if emotion_df.empty:
            continue

        emotion_df = emotion_df.sort_values(
            "distinctiveness",
            ascending=True
        )

        plt.figure(figsize=(10, 6))

        plt.barh(
            emotion_df[text_col],
            emotion_df["distinctiveness"]
        )

        plt.xlabel("Distinctiveness")
        plt.ylabel(pattern_type)
        plt.title(
            f"Top {top_n} Distinctive {pattern_type} - {emotion}"
        )

        plt.tight_layout()

        output_file = (
            OUTPUT_DIR
            / f"{pattern_type.lower()}_{emotion.lower()}.png"
        )

        plt.savefig(output_file, dpi=300)
        plt.close()

        print(f"Saved: {output_file}")



visualize_patterns(
    file_name="distinctive_words.csv",
    text_col="word",
    pattern_type="Words",
    top_n=10
)


visualize_patterns(
    file_name="distinctive_bigrams.csv",
    text_col="phrase",
    pattern_type="Bigrams",
    top_n=10
)


visualize_patterns(
    file_name="distinctive_trigrams.csv",
    text_col="phrase",
    pattern_type="Trigrams",
    top_n=10
)


print("\nVisualisasi distinctive patterns selesai.")