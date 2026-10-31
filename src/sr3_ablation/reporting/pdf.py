# ADDED: new file, not part of the upstream SR3 codebase.
"""Print an HTML file to PDF with headless Edge / Chrome and stamp page numbers."""  # ADDED

from __future__ import annotations  # ADDED

import shutil  # ADDED
import subprocess  # ADDED
from pathlib import Path  # ADDED

from sr3_ablation.utils.log import get_logger  # ADDED

BROWSER_CANDIDATES = (  # ADDED
    r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe",  # ADDED
    r"C:\Program Files\Microsoft\Edge\Application\msedge.exe",  # ADDED
    r"C:\Program Files\Google\Chrome\Application\chrome.exe",  # ADDED
    "google-chrome",  # ADDED
    "chromium",  # ADDED
    "chromium-browser",  # ADDED
    "microsoft-edge",  # ADDED
)  # ADDED

logger = get_logger("reporting.pdf")  # ADDED


def find_browser() -> str | None:  # ADDED
    """Path of a Chromium-based browser for printing to PDF, or ``None``."""  # ADDED
    for candidate in BROWSER_CANDIDATES:  # ADDED
        if Path(candidate).is_file():  # ADDED
            return candidate  # ADDED
        located = shutil.which(candidate)  # ADDED
        if located:  # ADDED
            return located  # ADDED
    return None  # ADDED


def html_to_pdf(html_path: Path, pdf_path: Path, browser: str) -> Path:  # ADDED
    """Print ``html_path`` to ``pdf_path`` with a headless browser."""  # ADDED
    command = [  # ADDED
        browser,  # ADDED
        "--headless=new",  # ADDED
        "--disable-gpu",  # ADDED
        "--no-sandbox",  # ADDED
        "--disable-extensions",  # ADDED
        "--no-pdf-header-footer",  # ADDED
        "--virtual-time-budget=45000",  # ADDED
        "--run-all-compositor-stages-before-draw",  # ADDED
        f"--print-to-pdf={pdf_path}",  # ADDED
        html_path.resolve().as_uri(),  # ADDED
    ]  # ADDED
    pdf_path.unlink(missing_ok=True)  # ADDED
    completed = subprocess.run(command, capture_output=True, text=True, timeout=300, check=False)  # ADDED
    if completed.returncode != 0 or not pdf_path.is_file():  # ADDED
        raise RuntimeError(  # ADDED
            f"PDF printing failed ({completed.returncode}): {completed.stderr[-600:]}"  # ADDED
        )  # ADDED
    return pdf_path  # ADDED


def stamp_page_numbers(pdf_path: Path, running_title: str) -> int:  # ADDED
    """Add 'Page x of y' and a running title to every page (needs PyMuPDF; else skipped)."""  # ADDED
    try:  # ADDED
        import pymupdf  # ADDED
    except ImportError:  # ADDED
        logger.warning("PyMuPDF not installed; page numbers skipped (pip install -e .[pdf]).")  # ADDED
        return 0  # ADDED
    document = pymupdf.open(pdf_path)  # ADDED
    total = document.page_count  # ADDED
    grey = (0.42, 0.46, 0.51)  # ADDED
    for index, page in enumerate(document):  # ADDED
        width, y = page.rect.width, page.rect.height - 24  # ADDED
        page.draw_line(  # ADDED
            pymupdf.Point(38, y - 10),  # ADDED
            pymupdf.Point(width - 38, y - 10),  # ADDED
            color=(0.72, 0.75, 0.79),  # ADDED
            width=0.5,  # ADDED
        )  # ADDED
        page.insert_text(  # ADDED
            pymupdf.Point(38, y + 2),  # ADDED
            running_title,  # ADDED
            fontname="Times-Italic",  # ADDED
            fontsize=7.5,  # ADDED
            color=grey,  # ADDED
        )  # ADDED
        label = f"Page {index + 1} of {total}"  # ADDED
        label_width = pymupdf.get_text_length(label, fontname="Times-Roman", fontsize=7.5)  # ADDED
        page.insert_text(  # ADDED
            pymupdf.Point(width - 38 - label_width, y + 2),  # ADDED
            label,  # ADDED
            fontname="Times-Roman",  # ADDED
            fontsize=7.5,  # ADDED
            color=grey,  # ADDED
        )  # ADDED
    document.saveIncr()  # ADDED
    document.close()  # ADDED
    return total  # ADDED
