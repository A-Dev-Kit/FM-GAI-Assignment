# ADDED: new file, not part of the upstream SR3 codebase.
"""Assemble the report: render every section, fill the template and write the documents."""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from pathlib import Path

from sr3_ablation.config.schema import ExperimentConfig
from sr3_ablation.paths import REPORT_DIR
from sr3_ablation.reporting.markdown_html import markdown_to_html
from sr3_ablation.reporting.pdf import find_browser, html_to_pdf, stamp_page_numbers
from sr3_ablation.reporting.sections import (
    SECTION_REGISTRY,
    ReportContext,
    ReportSection,
    pending,
)
from sr3_ablation.utils.log import get_logger

TITLE = "Noise Schedule Ablation in SR3: theta_t vs theta_t^x on Kvasir-SEG"
RUNNING_TITLE = "CSL7860 Group Assignment - Noise Schedule Ablation (SR3, Kvasir-SEG)"
REPORT_STEM = "report"
_PLACEHOLDER_NAME = r"[A-Z_]+"
_PLACEHOLDER = re.compile(rf"\{{\{{({_PLACEHOLDER_NAME})\}}\}}")

logger = get_logger("reporting")

__all__ = ["ReportBuilder", "ReportWriter", "fill_template", "pending"]


def fill_template(template: str, values: Mapping[str, str]) -> str:
    """Replace ``{{NAME}}`` placeholders; unknown names raise ``KeyError``."""
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)


class ReportWriter:
    """Writes the filled Markdown and its HTML rendering, and prints the HTML to PDF."""

    def __init__(
        self, build_dir: Path, title: str = TITLE, running_title: str = RUNNING_TITLE
    ) -> None:
        self._build_dir = build_dir
        self._title = title
        self._running_title = running_title

    def write(self, markdown: str, pdf: bool) -> Path:
        """Write ``report.md``, ``report.html`` and optionally the PDF; return the ``.md`` path."""
        self._build_dir.mkdir(parents=True, exist_ok=True)
        markdown_path = self._build_dir / f"{REPORT_STEM}.md"
        markdown_path.write_text(markdown, encoding="utf-8")
        html_path = markdown_path.with_suffix(".html")
        html_path.write_text(markdown_to_html(markdown, self._title), encoding="utf-8")
        logger.info("Wrote %s and %s", markdown_path, html_path)
        if pdf:
            self._print_pdf(html_path)
        return markdown_path

    def _print_pdf(self, html_path: Path) -> None:
        browser = find_browser()
        if browser is None:
            logger.warning("No Edge/Chrome found; open report.html and print it to PDF instead.")
            return
        pdf_path = html_to_pdf(html_path, html_path.with_suffix(".pdf"), browser)
        pages = stamp_page_numbers(pdf_path, self._running_title)
        logger.info("Wrote %s (%d pages)", pdf_path, pages)


class ReportBuilder:
    """Fills the report template with the Markdown of every registered section."""

    def __init__(
        self,
        config: ExperimentConfig,
        exponent: float,
        template: Path = REPORT_DIR / "report_template.md",
        build_dir: Path = REPORT_DIR / "build",
        sections: Sequence[ReportSection] = SECTION_REGISTRY,
    ) -> None:
        placeholders = [section.placeholder for section in sections]
        invalid = sorted(name for name in placeholders if not re.fullmatch(_PLACEHOLDER_NAME, name))
        if invalid:
            raise ValueError(f"Placeholder names must match {_PLACEHOLDER_NAME}: {invalid}")
        duplicates = sorted({name for name in placeholders if placeholders.count(name) > 1})
        if duplicates:
            raise ValueError(f"Placeholders rendered by more than one section: {duplicates}")
        self._context = ReportContext.create(config, exponent, build_dir)
        self._template = template
        self._sections = tuple(sections)
        self._writer = ReportWriter(build_dir)

    def render(self) -> str:
        """Return the filled Markdown without writing anything except figure assets."""
        values = {section.placeholder: section.render(self._context) for section in self._sections}
        return fill_template(self._template.read_text(encoding="utf-8"), values)

    def build(self, pdf: bool = True) -> Path:
        """Render the report and write it; return the path of ``report.md``."""
        return self._writer.write(self.render(), pdf)
