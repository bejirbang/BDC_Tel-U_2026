from __future__ import annotations

import argparse
import tempfile
from functools import lru_cache
from pathlib import Path

from tqdm import tqdm

from src.extraction.google_drive import DriveFileUnavailableError, download_drive_video
from src.extraction.instagram import VideoUnavailableError, download_instagram_video
from src.extraction.utils import (
    clean_text,
    ensure_transcript_dirs,
    iter_dataset_rows,
    project_root,
    transcript_paths,
    write_status_csv,
)


@lru_cache(maxsize=4)
def _load_whisper_model(model_name: str):
    try:
        import whisper
    except ImportError as exc:
        raise RuntimeError("openai-whisper is not installed. Run: pip install -r requirements.txt") from exc

    return whisper.load_model(model_name)


def transcribe_video(video_path: Path, model_name: str = "small", language: str | None = "id") -> str:
    model = _load_whisper_model(model_name)
    kwargs = {"fp16": False}
    if language:
        kwargs["language"] = language
    result = model.transcribe(str(video_path), **kwargs)
    return clean_text(result.get("text", ""))


def _download_video(row, video_path: Path, cookies_file: Path | None) -> Path:
    if row.source_type == "IG":
        return download_instagram_video(row.url, video_path, cookies_file=cookies_file)
    if row.source_type == "GD":
        return download_drive_video(row.url, video_path)
    raise ValueError("unsupported source domain")


def build_transcripts(
    dataset_path: Path,
    output_status_path: Path,
    model_name: str,
    language: str | None,
    limit: int | None,
    skip_existing: bool,
    keep_video_dir: Path | None,
    cookies_file: Path | None,
    dry_run: bool,
) -> list[dict[str, str]]:
    root = project_root()
    if cookies_file and not cookies_file.exists():
        raise FileNotFoundError(
            f"Instagram cookies file not found: {cookies_file}. "
            "Remove --instagram-cookies or pass a real cookies.txt path."
        )

    rows = list(iter_dataset_rows(dataset_path))
    if limit is not None:
        rows = rows[:limit]
    ensure_transcript_dirs(root, [row.emotion for row in rows])

    records: list[dict[str, str]] = []
    temp_context = tempfile.TemporaryDirectory(prefix="bdc_video_") if keep_video_dir is None else None
    video_dir = keep_video_dir or Path(temp_context.name)
    video_dir.mkdir(parents=True, exist_ok=True)

    try:
        for row in tqdm(rows, desc="Transcribing videos"):
            raw_path, clean_path = transcript_paths(root, row)
            record = {
                "dataset_id": row.dataset_id,
                "video_id": row.video_id,
                "emotion": row.emotion,
                "url": row.url,
                "source_type": row.source_type,
                "status": "",
                "has_raw_transcript": str(raw_path.exists()),
                "has_cleaned_transcript": str(clean_path.exists()),
                "transcription_method": f"whisper:{model_name}",
                "raw_transcript_path": str(raw_path.relative_to(root)),
                "cleaned_transcript_path": str(clean_path.relative_to(root)),
                "notes": "",
            }

            if row.source_type == "UNSUPPORTED":
                record.update(
                    {
                        "status": "unsupported_source",
                        "transcription_method": "",
                        "notes": "Only Instagram and Google Drive sources are allowed",
                    }
                )
                records.append(record)
                continue

            if skip_existing and raw_path.exists() and clean_path.exists():
                record.update(
                    {
                        "status": "skipped_existing",
                        "has_raw_transcript": "True",
                        "has_cleaned_transcript": "True",
                    }
                )
                records.append(record)
                continue

            if dry_run:
                record["status"] = "dry_run"
                records.append(record)
                continue

            video_path = video_dir / f"{row.video_id}.mp4"
            try:
                _download_video(row, video_path, cookies_file)
                transcript = transcribe_video(video_path, model_name=model_name, language=language)
                raw_path.write_text(transcript + "\n", encoding="utf-8")
                clean_path.write_text(clean_text(transcript).lower() + "\n", encoding="utf-8")
                record.update(
                    {
                        "status": "transcribed",
                        "has_raw_transcript": "True",
                        "has_cleaned_transcript": "True",
                    }
                )
            except (VideoUnavailableError, DriveFileUnavailableError) as exc:
                record.update({"status": "deleted_or_unavailable", "notes": str(exc)})
            except Exception as exc:
                record.update({"status": "failed", "notes": str(exc)})
            finally:
                if keep_video_dir is None and video_path.exists():
                    video_path.unlink()

            records.append(record)
            write_status_csv(output_status_path, records)
    finally:
        if temp_context is not None:
            temp_context.cleanup()

    write_status_csv(output_status_path, records)
    return records


def parse_args() -> argparse.Namespace:
    root = project_root()
    parser = argparse.ArgumentParser(description="Download and transcribe all videos in datatrain.csv")
    parser.add_argument("--dataset", type=Path, default=root / "data" / "metadata" / "datatrain.csv")
    parser.add_argument(
        "--status-out",
        type=Path,
        default=root / "data" / "metadata" / "transcript_status.csv",
    )
    parser.add_argument("--model", default="small", help="Whisper model name: tiny, base, small, medium, large")
    parser.add_argument("--language", default="id", help="Whisper language code. Use empty string for auto-detect.")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--keep-video-dir", type=Path, default=None)
    parser.add_argument("--instagram-cookies", type=Path, default=None)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    language = args.language or None
    build_transcripts(
        dataset_path=args.dataset,
        output_status_path=args.status_out,
        model_name=args.model,
        language=language,
        limit=args.limit,
        skip_existing=not args.overwrite,
        keep_video_dir=args.keep_video_dir,
        cookies_file=args.instagram_cookies,
        dry_run=args.dry_run,
    )


if __name__ == "__main__":
    main()
