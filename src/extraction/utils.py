from __future__ import annotations

import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable
from urllib.parse import urlparse


EMOTION_CODES = {
    "Surprise": "SPR",
    "Proud": "PRD",
    "Trust": "TRT",
    "Sad": "SAD",
    "Joy": "JOY",
    "Anger": "ANG",
    "Fear": "FEA",
    "Neutral": "NEU",
    "Love": "LOV",
    "Loyalty": "LOY",
}

INSTAGRAM_HOST_MARKERS = (
    "instagram.com",
    "cdninstagram.com",
    "instagram.",
)
GOOGLE_DRIVE_HOST = "drive.google.com"


@dataclass(frozen=True)
class VideoRow:
    dataset_id: str
    url: str
    emotion: str
    source_type: str
    video_id: str


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def clean_text(text: str) -> str:
    text = text.replace("\ufeff", " ")
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def detect_source_type(url: str) -> str:
    host = urlparse(url).hostname or ""
    host = host.lower()
    if host == GOOGLE_DRIVE_HOST:
        return "GD"
    if any(marker in host for marker in INSTAGRAM_HOST_MARKERS):
        return "IG"
    return "UNSUPPORTED"


def make_video_id(emotion: str, sequence: int, source_type: str) -> str:
    emotion_code = EMOTION_CODES.get(emotion, re.sub(r"[^A-Za-z]", "", emotion).upper()[:3])
    return f"{emotion_code}_{sequence:03d}_{source_type}"


def iter_dataset_rows(csv_path: Path) -> Iterable[VideoRow]:
    counters: dict[str, int] = {}
    with csv_path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"id", "video", "emotion"}
        missing = required.difference(reader.fieldnames or [])
        if missing:
            raise ValueError(f"Missing required column(s): {', '.join(sorted(missing))}")

        for row in reader:
            emotion = clean_text(row["emotion"])
            url = clean_text(row["video"])
            source_type = detect_source_type(url)
            counters[emotion] = counters.get(emotion, 0) + 1
            yield VideoRow(
                dataset_id=clean_text(row["id"]),
                url=url,
                emotion=emotion,
                source_type=source_type,
                video_id=make_video_id(emotion, counters[emotion], source_type),
            )


def ensure_transcript_dirs(root: Path, emotions: Iterable[str]) -> None:
    for emotion in set(emotions):
        (root / "data" / "1_raw" / "transcripts" / emotion).mkdir(parents=True, exist_ok=True)
        (root / "data" / "2_processed" / "transcripts" / emotion).mkdir(parents=True, exist_ok=True)


def transcript_paths(root: Path, row: VideoRow) -> tuple[Path, Path]:
    raw_path = root / "data" / "1_raw" / "transcripts" / row.emotion / f"{row.video_id}_raw.txt"
    clean_path = (
        root
        / "data"
        / "2_processed"
        / "transcripts"
        / row.emotion
        / f"{row.video_id}_clean.txt"
    )
    return raw_path, clean_path


def write_status_csv(path: Path, records: list[dict[str, str]]) -> None:
    fieldnames = [
        "dataset_id",
        "video_id",
        "emotion",
        "url",
        "source_type",
        "status",
        "has_raw_transcript",
        "has_cleaned_transcript",
        "transcription_method",
        "raw_transcript_path",
        "cleaned_transcript_path",
        "notes",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
