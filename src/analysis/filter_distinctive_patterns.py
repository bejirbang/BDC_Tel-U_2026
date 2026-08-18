import pandas as pd
from pathlib import Path
import re


BASE_DIR = Path(__file__).resolve().parents[2]

INPUT_DIR = BASE_DIR / "analysis" / "distinctive_patterns"
OUTPUT_DIR = INPUT_DIR / "filtered"

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def clean_pattern(text):
    text = str(text).lower().strip()

    # Buang angka saja
    if re.fullmatch(r"\d+", text):
        return False

    # Minimal mengandung huruf
    if not re.search(r"[a-zA-Z]", text):
        return False

    # Jangan terlalu pendek
    if len(text) <= 1:
        return False

    return True


def filter_file(filename, text_col):

    input_file = INPUT_DIR / filename
    output_file = OUTPUT_DIR / filename

    df = pd.read_csv(input_file)

    # Hanya distinctive positif
    df = df[df["distinctiveness"] > 0].copy()

    # Filter pattern
    df = df[df[text_col].apply(clean_pattern)]

    # Urutkan berdasarkan emotion + distinctive
    df = df.sort_values(
        ["emotion", "distinctiveness"],
        ascending=[True, False]
    )

    df.to_csv(output_file, index=False)

    print(f"Saved: {output_file}")


filter_file(
    "distinctive_words.csv",
    "word"
)

filter_file(
    "distinctive_bigrams.csv",
    "phrase"
)

filter_file(
    "distinctive_trigrams.csv",
    "phrase"
)


print("\nFiltering selesai.")