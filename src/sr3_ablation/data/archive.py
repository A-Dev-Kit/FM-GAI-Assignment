# ADDED: new file, not part of the upstream SR3 codebase.
"""Download, extract and list image files; independent of any particular dataset."""

from __future__ import annotations

import urllib.request
import zipfile
from pathlib import Path

from tqdm import tqdm

from sr3_ablation.utils.log import get_logger

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})
EXTRACTED_MARKER = ".extracted"
_CHUNK_BYTES = 1 << 20

logger = get_logger("data.archive")


def list_images(folder: Path) -> list[Path]:
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)


def download_file(url: str, destination: Path) -> Path:
    """Stream ``url`` (``https://`` or ``file://``) to ``destination``; skipped if it exists."""
    if destination.is_file():
        logger.info("Using existing download %s", destination)
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_suffix(destination.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": "sr3-ablation/1.0"})
    logger.info("Downloading %s", url)
    with urllib.request.urlopen(request) as response, partial.open("wb") as handle:
        total = int(response.headers.get("Content-Length", 0)) or None
        with tqdm(total=total, unit="B", unit_scale=True, desc=destination.name) as progress:
            while chunk := response.read(_CHUNK_BYTES):
                handle.write(chunk)
                progress.update(len(chunk))
    partial.replace(destination)
    return destination


def extract_archive(archive: Path, destination: Path) -> None:
    """Extract in place and mark completion (no folder renames; they fail under OneDrive)."""
    logger.info("Extracting %s to %s", archive, destination)
    destination.mkdir(parents=True, exist_ok=True)
    marker = destination / EXTRACTED_MARKER
    marker.unlink(missing_ok=True)
    with zipfile.ZipFile(archive) as bundle:
        bundle.extractall(destination)
    marker.write_text(archive.name, encoding="utf-8")
