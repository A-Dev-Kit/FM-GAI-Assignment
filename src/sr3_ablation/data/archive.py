# ADDED: new file, not part of the upstream SR3 codebase.
"""Download, extract and list image files; independent of any particular dataset."""  # ADDED

from __future__ import annotations  # ADDED

import urllib.request  # ADDED
import zipfile  # ADDED
from pathlib import Path  # ADDED

from tqdm import tqdm  # ADDED

from sr3_ablation.utils.log import get_logger  # ADDED

IMAGE_SUFFIXES = frozenset({".jpg", ".jpeg", ".png"})  # ADDED
EXTRACTED_MARKER = ".extracted"  # ADDED
_CHUNK_BYTES = 1 << 20  # ADDED

logger = get_logger("data.archive")  # ADDED


def list_images(folder: Path) -> list[Path]:  # ADDED
    """Image files directly inside ``folder``, sorted by name."""  # ADDED
    return sorted(p for p in folder.iterdir() if p.suffix.lower() in IMAGE_SUFFIXES)  # ADDED


def download_file(url: str, destination: Path) -> Path:  # ADDED
    """Stream ``url`` (``https://`` or ``file://``) to ``destination``; skipped if it exists."""  # ADDED
    if destination.is_file():  # ADDED
        logger.info("Using existing download %s", destination)  # ADDED
        return destination  # ADDED
    destination.parent.mkdir(parents=True, exist_ok=True)  # ADDED
    partial = destination.with_suffix(destination.suffix + ".part")  # ADDED
    request = urllib.request.Request(url, headers={"User-Agent": "sr3-ablation/1.0"})  # ADDED
    logger.info("Downloading %s", url)  # ADDED
    with urllib.request.urlopen(request) as response, partial.open("wb") as handle:  # ADDED
        total = int(response.headers.get("Content-Length", 0)) or None  # ADDED
        with tqdm(total=total, unit="B", unit_scale=True, desc=destination.name) as progress:  # ADDED
            while chunk := response.read(_CHUNK_BYTES):  # ADDED
                handle.write(chunk)  # ADDED
                progress.update(len(chunk))  # ADDED
    partial.replace(destination)  # ADDED
    return destination  # ADDED


def extract_archive(archive: Path, destination: Path) -> None:  # ADDED
    """Extract in place and mark completion (no folder renames; they fail under OneDrive)."""  # ADDED
    logger.info("Extracting %s to %s", archive, destination)  # ADDED
    destination.mkdir(parents=True, exist_ok=True)  # ADDED
    marker = destination / EXTRACTED_MARKER  # ADDED
    marker.unlink(missing_ok=True)  # ADDED
    with zipfile.ZipFile(archive) as bundle:  # ADDED
        bundle.extractall(destination)  # ADDED
    marker.write_text(archive.name, encoding="utf-8")  # ADDED
