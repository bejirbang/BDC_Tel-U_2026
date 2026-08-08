from __future__ import annotations

import re
import shutil
import subprocess
from pathlib import Path


class DriveFileUnavailableError(RuntimeError):
    pass


def extract_drive_file_id(url: str) -> str | None:
    patterns = (
        r"/file/d/([^/]+)",
        r"[?&]id=([^&]+)",
        r"/open\?id=([^&]+)",
    )
    for pattern in patterns:
        match = re.search(pattern, url)
        if match:
            return match.group(1)
    return None


def _classify_gdown_error(message: str) -> str:
    lowered = message.lower()
    deleted_markers = (
        "cannot retrieve the public link",
        "file not found",
        "permission",
        "access denied",
        "quota",
        "404",
        "403",
    )
    if any(marker in lowered for marker in deleted_markers):
        return "google drive file deleted/unavailable/private"
    return message.strip() or "gdown failed"


def download_drive_video(url: str, output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    file_id = extract_drive_file_id(url)
    if not file_id:
        raise DriveFileUnavailableError("google drive file id could not be parsed")

    gdown = shutil.which("gdown")
    if not gdown:
        raise RuntimeError("gdown is not installed or not available in PATH")

    result = subprocess.run(
        [gdown, "--output", str(output_path), "--quiet", url],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0 or not output_path.exists() or output_path.stat().st_size == 0:
        message = "\n".join(part for part in [result.stderr, result.stdout] if part)
        raise DriveFileUnavailableError(_classify_gdown_error(message))
    return output_path
