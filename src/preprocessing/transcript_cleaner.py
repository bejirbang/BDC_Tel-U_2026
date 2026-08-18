from __future__ import annotations

import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]

RAW_DIR = PROJECT_ROOT / "data" / "1_raw" / "transcripts"
PROCESSED_DIR = PROJECT_ROOT / "data" / "2_processed" / "transcripts"

def clean_transcript(text: str) -> str:
    text = text.lower()

    text = re.sub(r"https?://\S+|www\.\S+", " ", text)

    text = re.sub(r"[-–—]", " ", text)

    text = re.sub(r"[^\w\s]", " ", text)

    text = re.sub(r"\s+", " ", text)

    text = text.strip()

    return text


def process_file(input_path: Path, output_path: Path) -> None:

    text = input_path.read_text(encoding="utf-8")

    cleaned_text = clean_transcript(text)

    output_path.parent.mkdir(parents=True, exist_ok=True)

    output_path.write_text(
        cleaned_text + "\n",
        encoding="utf-8",
    )

def process_all_transcripts() -> None:
    """Process seluruh transcript berdasarkan struktur folder emotion."""

    if not RAW_DIR.exists():
        raise FileNotFoundError(
            f"Raw transcript directory tidak ditemukan: {RAW_DIR}"
        )

    processed_count = 0
    skipped_count = 0

    for emotion_dir in sorted(RAW_DIR.iterdir()):

        if not emotion_dir.is_dir():
            continue

        emotion = emotion_dir.name

        raw_files = sorted(emotion_dir.glob("*_raw.txt"))

        if not raw_files:
            continue

        for raw_file in raw_files:

            output_filename = raw_file.name.replace(
                "_raw.txt",
                "_processed.txt",
            )

            output_path = (
                PROCESSED_DIR
                / emotion
                / output_filename
            )

            try:
                process_file(
                    input_path=raw_file,
                    output_path=output_path,
                )

                processed_count += 1

                print(
                    f"[OK] {emotion}/{raw_file.name}"
                )

            except Exception as exc:
                skipped_count += 1

                print(
                    f"[ERROR] {emotion}/{raw_file.name}: {exc}"
                )

    print("\n" + "=" * 60)
    print("PREPROCESSING SELESAI")
    print("=" * 60)
    print(f"Berhasil : {processed_count}")
    print(f"Gagal    : {skipped_count}")
    print(f"Output   : {PROCESSED_DIR}")

if __name__ == "__main__":
    process_all_transcripts()