# ADDED: new file, not part of the upstream SR3 codebase.
"""Download and unpack Kvasir-SEG (Jha et al., MMM 2020; research and education use only)."""

from __future__ import annotations

from pathlib import Path

from sr3_ablation.data.archive import EXTRACTED_MARKER, download_file, extract_archive, list_images
from sr3_ablation.utils.log import get_logger

DATASET_SLUG = "kvasir_seg"
IMAGES_FOLDER = "images"

logger = get_logger("data.kvasir")


class KvasirSegSource:
    """Locates the Kvasir-SEG frames under ``data_root/raw/kvasir-seg``, fetching them if missing.

    The archive unpacks to ``Kvasir-SEG/{images,masks}/*.jpg`` plus a bounding-box JSON. Only
    ``images/`` is used; the polyp masks play no part in super-resolution.
    """

    def __init__(self, url: str, data_root: Path) -> None:
        self._url = url
        self._data_root = data_root

    @property
    def archive_path(self) -> Path:
        return self._data_root / "kvasir-seg.zip"

    @property
    def raw_dir(self) -> Path:
        return self._data_root / "raw" / "kvasir-seg"

    def ensure(self) -> Path:
        """Return the folder holding the frames, downloading and extracting first if needed."""
        existing = self.locate()
        if existing is not None:
            logger.info("Kvasir-SEG already available at %s", existing)
            return existing
        download_file(self._url, self.archive_path)
        extract_archive(self.archive_path, self.raw_dir)
        located = self.locate()
        if located is None:
            raise FileNotFoundError(
                f"No {IMAGES_FOLDER}/ folder with images inside {self.raw_dir}."
            )
        return located

    def locate(self) -> Path | None:
        """The ``images`` folder if a previous extraction completed, else ``None``."""
        if not (self.raw_dir / EXTRACTED_MARKER).is_file():
            return None
        for candidate in sorted(self.raw_dir.rglob(IMAGES_FOLDER)):
            if candidate.is_dir() and list_images(candidate):
                return candidate
        return None
