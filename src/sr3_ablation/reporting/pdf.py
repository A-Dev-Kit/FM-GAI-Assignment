# ADDED: new file, not part of the upstream SR3 codebase.
"""Print an HTML file to PDF with headless Edge / Chrome and stamp page numbers."""

from __future__ import annotations

import shutil
import subprocess
from pathlib import Path

from sr3_ablation.utils.log import get_logger

BROWSER_CANDIDATES = (
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",
    "google-chrome",
    "chromium",
    "chromium-browser",
    "microsoft-edge",
)

logger = get_logger("reporting.pdf")


def find_browser() -> str | None:
    for candidate in BROWSER_CANDIDATES:
        if Path(candidate).is_file():
            return candidate
        located = shutil.which(candidate)
        if located:
            return located
    return None


def html_to_pdf(html_path: Path, pdf_path: Path, browser: str) -> Path:
    command = [
        browser,
        "--headless=new",
        "--disable-gpu",
        "--no-sandbox",
        "--disable-extensions",
        "--no-pdf-header-footer",
        "--virtual-time-budget=45000",
        "--run-all-compositor-stages-before-draw",
        f"--print-to-pdf={pdf_path}",
        html_path.resolve().as_uri(),
    ]
    pdf_path.unlink(missing_ok=True)
    completed = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)
    if completed.returncode != 0 or not pdf_path.is_file():
        raise RuntimeError(
            f"PDF printing failed ({completed.returncode}): {completed.stderr[-600:]}"
        )
    return pdf_path


def stamp_page_numbers(pdf_path: Path, running_title: str) -> int:
    """Add 'Page x of y' and a running title to every page (needs PyMuPDF; else skipped)."""
    try:
        import pymupdf
    except ImportError:
        logger.warning("PyMuPDF not installed; page numbers skipped (pip install -e .[pdf]).")
        return 0
    document = pymupdf.open(pdf_path)
    total = document.page_count
    grey = (0.42, 0.46, 0.51)
    for index, page in enumerate(document):
        width, y = page.rect.width, page.rect.height - 24
        page.draw_line(
            pymupdf.Point(38, y - 10),
            pymupdf.Point(width - 38, y - 10),
            color=(0.72, 0.75, 0.79),
            width=0.5,
        )
        page.insert_text(
            pymupdf.Point(38, y + 2),
            running_title,
            fontname="Times-Italic",
            fontsize=7.5,
            color=grey,
        )
        label = f"Page {index + 1} of {total}"
        label_width = pymupdf.get_text_length(label, fontname="Times-Roman", fontsize=7.5)
        page.insert_text(
            pymupdf.Point(width - 38 - label_width, y + 2),
            label,
            fontname="Times-Roman",
            fontsize=7.5,
            color=grey,
        )
    document.saveIncr()
    document.close()
    return total
