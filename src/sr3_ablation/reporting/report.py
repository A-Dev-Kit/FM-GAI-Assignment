# ADDED: new file, not part of the upstream SR3 codebase.
"""Assemble the report: render every section, fill the template and write the documents."""  # ADDED

from __future__ import annotations  # ADDED

import re  # ADDED
from collections.abc import Mapping, Sequence  # ADDED
from pathlib import Path  # ADDED

from sr3_ablation.config.schema import ExperimentConfig  # ADDED
from sr3_ablation.paths import REPORT_DIR  # ADDED
from sr3_ablation.reporting.markdown_html import markdown_to_html  # ADDED
from sr3_ablation.reporting.pdf import find_browser, html_to_pdf, stamp_page_numbers  # ADDED
from sr3_ablation.reporting.sections import (  # ADDED
    SECTION_REGISTRY,  # ADDED
    ReportContext,  # ADDED
    ReportSection,  # ADDED
    pending,  # ADDED
)  # ADDED
from sr3_ablation.utils.log import get_logger  # ADDED

TITLE = "Noise Schedule Ablation in SR3: theta_t vs theta_t^x on Kvasir-SEG"  # ADDED
RUNNING_TITLE = "CSL7860 Group Assignment - Noise Schedule Ablation (SR3, Kvasir-SEG)"  # ADDED
REPORT_STEM = "report"  # ADDED
_PLACEHOLDER_NAME = r"[A-Z_]+"  # ADDED
_PLACEHOLDER = re.compile(rf"\{{\{{({_PLACEHOLDER_NAME})\}}\}}")  # ADDED

logger = get_logger("reporting")  # ADDED

__all__ = ["ReportBuilder", "ReportWriter", "fill_template", "pending"]  # ADDED


def fill_template(template: str, values: Mapping[str, str]) -> str:  # ADDED
    """Replace ``{{NAME}}`` placeholders; unknown names raise ``KeyError``."""  # ADDED
    return _PLACEHOLDER.sub(lambda match: values[match.group(1)], template)  # ADDED


class ReportWriter:  # ADDED
    """Writes the filled Markdown and its HTML rendering, and prints the HTML to PDF."""  # ADDED

    def __init__(  # ADDED
        self,  # ADDED
        build_dir: Path,  # ADDED
        title: str = TITLE,  # ADDED
        running_title: str = RUNNING_TITLE,  # ADDED
    ) -> None:  # ADDED
        self._build_dir = build_dir  # ADDED
        self._title = title  # ADDED
        self._running_title = running_title  # ADDED

    def write(self, markdown: str, pdf: bool) -> Path:  # ADDED
        """Write ``report.md``, ``report.html`` and optionally the PDF; return the ``.md`` path."""  # ADDED
        self._build_dir.mkdir(parents=True, exist_ok=True)  # ADDED
        markdown_path = self._build_dir / f"{REPORT_STEM}.md"  # ADDED
        markdown_path.write_text(markdown, encoding="utf-8")  # ADDED
        html_path = markdown_path.with_suffix(".html")  # ADDED
        html_path.write_text(markdown_to_html(markdown, self._title), encoding="utf-8")  # ADDED
        logger.info("Wrote %s and %s", markdown_path, html_path)  # ADDED
        if pdf:  # ADDED
            self._print_pdf(html_path)  # ADDED
        return markdown_path  # ADDED

    def _print_pdf(self, html_path: Path) -> None:  # ADDED
        browser = find_browser()  # ADDED
        if browser is None:  # ADDED
            logger.warning("No Edge/Chrome found; open report.html and print it to PDF instead.")  # ADDED
            return  # ADDED
        pdf_path = html_to_pdf(html_path, html_path.with_suffix(".pdf"), browser)  # ADDED
        pages = stamp_page_numbers(pdf_path, self._running_title)  # ADDED
        logger.info("Wrote %s (%d pages)", pdf_path, pages)  # ADDED


class ReportBuilder:  # ADDED
    """Fills the report template with the Markdown of every registered section."""  # ADDED

    def __init__(  # ADDED
        self,  # ADDED
        config: ExperimentConfig,  # ADDED
        exponent: float,  # ADDED
        template: Path = REPORT_DIR / "report_template.md",  # ADDED
        build_dir: Path = REPORT_DIR / "build",  # ADDED
        sections: Sequence[ReportSection] = SECTION_REGISTRY,  # ADDED
    ) -> None:  # ADDED
        placeholders = [section.placeholder for section in sections]  # ADDED
        invalid = sorted(name for name in placeholders if not re.fullmatch(_PLACEHOLDER_NAME, name))  # ADDED
        if invalid:  # ADDED
            raise ValueError(f"Placeholder names must match {_PLACEHOLDER_NAME}: {invalid}")  # ADDED
        duplicates = sorted({name for name in placeholders if placeholders.count(name) > 1})  # ADDED
        if duplicates:  # ADDED
            raise ValueError(f"Placeholders rendered by more than one section: {duplicates}")  # ADDED
        self._context = ReportContext.create(config, exponent, build_dir)  # ADDED
        self._template = template  # ADDED
        self._sections = tuple(sections)  # ADDED
        self._writer = ReportWriter(build_dir)  # ADDED

    def render(self) -> str:  # ADDED
        """Return the filled Markdown without writing anything except figure assets."""  # ADDED
        values = {section.placeholder: section.render(self._context) for section in self._sections}  # ADDED
        return fill_template(self._template.read_text(encoding="utf-8"), values)  # ADDED

    def build(self, pdf: bool = True) -> Path:  # ADDED
        """Render the report and write it; return the path of ``report.md``."""  # ADDED
        return self._writer.write(self.render(), pdf)  # ADDED
