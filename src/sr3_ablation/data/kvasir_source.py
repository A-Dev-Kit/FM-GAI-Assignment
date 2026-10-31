# ADDED: new file, not part of the upstream SR3 codebase.
"""Download and unpack Kvasir-SEG (Jha et al., MMM 2020; research and education use only)."""  # ADDED

from __future__ import annotations  # ADDED

from pathlib import Path  # ADDED

from sr3_ablation.data.archive import EXTRACTED_MARKER, download_file, extract_archive, list_images  # ADDED
from sr3_ablation.utils.log import get_logger  # ADDED

DATASET_SLUG = "kvasir_seg"  # ADDED
IMAGES_FOLDER = "images"  # ADDED

logger = get_logger("data.kvasir")  # ADDED


class KvasirSegSource:  # ADDED
    """Locates the Kvasir-SEG frames under ``data_root/raw/kvasir-seg``, fetching them if missing.  # ADDED

    The archive unpacks to ``Kvasir-SEG/{images,masks}/*.jpg`` plus a bounding-box JSON. Only  # ADDED
    ``images/`` is used; the polyp masks play no part in super-resolution.  # ADDED
    """  # ADDED

    def __init__(self, url: str, data_root: Path) -> None:  # ADDED
        self._url = url  # ADDED
        self._data_root = data_root  # ADDED

    @property  # ADDED
    def archive_path(self) -> Path:  # ADDED
        return self._data_root / "kvasir-seg.zip"  # ADDED

    @property  # ADDED
    def raw_dir(self) -> Path:  # ADDED
        return self._data_root / "raw" / "kvasir-seg"  # ADDED

    def ensure(self) -> Path:  # ADDED
        """Return the folder holding the frames, downloading and extracting first if needed."""  # ADDED
        existing = self.locate()  # ADDED
        if existing is not None:  # ADDED
            logger.info("Kvasir-SEG already available at %s", existing)  # ADDED
            return existing  # ADDED
        download_file(self._url, self.archive_path)  # ADDED
        extract_archive(self.archive_path, self.raw_dir)  # ADDED
        located = self.locate()  # ADDED
        if located is None:  # ADDED
            raise FileNotFoundError(  # ADDED
                f"No {IMAGES_FOLDER}/ folder with images inside {self.raw_dir}."  # ADDED
            )  # ADDED
        return located  # ADDED

    def locate(self) -> Path | None:  # ADDED
        """The ``images`` folder if a previous extraction completed, else ``None``."""  # ADDED
        if not (self.raw_dir / EXTRACTED_MARKER).is_file():  # ADDED
            return None  # ADDED
        for candidate in sorted(self.raw_dir.rglob(IMAGES_FOLDER)):  # ADDED
            if candidate.is_dir() and list_images(candidate):  # ADDED
                return candidate  # ADDED
        return None  # ADDED
