from __future__ import annotations

import shutil
import subprocess
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen
from urllib.parse import urlparse


class VideoUnavailableError(RuntimeError):
    pass


def _classify_ytdlp_error(message: str) -> str:
    lowered = message.lower()
    deleted_markers = (
        "unavailable",
        "not available",
        "private",
        "login required",
        "requested content is not available",
        "this video has been deleted",
        "unable to extract",
        "404",
        "403",
    )
    if any(marker in lowered for marker in deleted_markers):
        return "instagram video deleted/unavailable/private"
    return message.strip() or "yt-dlp failed"


def download_instagram_video(url: str, output_path: Path, cookies_file: Path | None = None) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    host = (urlparse(url).hostname or "").lower()

    if "cdninstagram.com" in host or host.startswith("instagram."):
        request = Request(url, headers={"User-Agent": "Mozilla/5.0"})
        try:
            with urlopen(request, timeout=60) as response:
                content_type = response.headers.get("Content-Type", "").lower()
                payload = response.read()
                if "video" not in content_type and not payload[:32].startswith(b"\x00\x00"):
                    raise VideoUnavailableError(
                        f"instagram cdn video deleted/unavailable: content-type={content_type or 'unknown'}"
                    )
                output_path.write_bytes(payload)
        except HTTPError as exc:
            raise VideoUnavailableError(f"instagram cdn video deleted/unavailable: HTTP {exc.code}") from exc
        except URLError as exc:
            raise VideoUnavailableError(f"instagram cdn video unavailable: {exc.reason}") from exc
        return output_path

    ytdlp = shutil.which("yt-dlp")
    if not ytdlp:
        raise RuntimeError("yt-dlp is not installed or not available in PATH")

    command = [
        ytdlp,
        "--no-playlist",
        "--format",
        "mp4/best",
        "--output",
        str(output_path),
        url,
    ]
    if cookies_file:
        if not cookies_file.exists():
            raise FileNotFoundError(f"Instagram cookies file not found: {cookies_file}")
        command.extend(["--cookies", str(cookies_file)])

    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        message = "\n".join(part for part in [result.stderr, result.stdout] if part)
        raise VideoUnavailableError(_classify_ytdlp_error(message))
    return output_path
