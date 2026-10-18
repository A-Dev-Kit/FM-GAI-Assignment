# ADDED: new file, not part of the upstream SR3 codebase.
"""Report sections render on their own and degrade to "pending" text when inputs are missing."""

from __future__ import annotations

import re

import pytest

from sr3_ablation.config import load_config
from sr3_ablation.paths import REPORT_DIR
from sr3_ablation.reporting import ReportBuilder
from sr3_ablation.reporting.sections import SECTION_REGISTRY, ReportContext
from sr3_ablation.reporting.sections.schedule import (
    SCHEDULE_NUMBER_SECTIONS,
    ExponentSection,
    ScheduleNumberSection,
)
from tests.helpers import DECLARED_EXPONENT

TEMPLATE = REPORT_DIR / "report_template.md"
ALWAYS_AVAILABLE = {
    "EXPONENT",
    "SCHEDULE_TABLE",
    "TRAJECTORY_SNR_TABLE",
    "SCHEDULE_FIGURE",
    "SETUP_TABLE",
    "CODE_DIFF",
    *(section.placeholder for section in SCHEDULE_NUMBER_SECTIONS),
}
EMPTY_WITHOUT_INPUTS = {"LOSS_TABLE"}
PENDING_MARKER = "> *Pending: run `"


@pytest.fixture
def context(tiny_config, tmp_path) -> ReportContext:
    return ReportContext.create(tiny_config, DECLARED_EXPONENT, tmp_path / "build")


def test_registry_matches_template_placeholders():
    in_template = set(re.findall(r"\{\{([A-Z_]+)\}\}", TEMPLATE.read_text(encoding="utf-8")))
    registered = [section.placeholder for section in SECTION_REGISTRY]
    assert len(registered) == len(set(registered))
    assert set(registered) == in_template


@pytest.mark.parametrize("section", SECTION_REGISTRY, ids=lambda s: s.placeholder)
def test_section_without_run_outputs(section, context):
    text = section.render(context)
    if section.placeholder in ALWAYS_AVAILABLE:
        assert text and PENDING_MARKER not in text
    elif section.placeholder in EMPTY_WITHOUT_INPUTS:
        assert text == ""
    else:
        assert text.startswith(PENDING_MARKER)


def test_context_derives_both_schedules(context):
    assert context.original.exponent == 1.0
    assert context.modified.exponent == DECLARED_EXPONENT
    assert context.assets == context.outputs / "report_assets"
    assert (
        context.image("cap", context.assets / "a.png") == "![cap](../outputs/report_assets/a.png)"
    )


def test_exponent_has_two_decimals(context):
    assert ExponentSection().render(context) == f"{DECLARED_EXPONENT:.2f}"


def test_schedule_numbers_for_declared_exponent(tmp_path):
    full = ReportContext.create(load_config(local_config=None), DECLARED_EXPONENT, tmp_path)
    rendered = {section.placeholder: section.render(full) for section in SCHEDULE_NUMBER_SECTIONS}
    assert rendered == {
        "ORIGINAL_ALPHA_BAR_T": "$4.39\\times10^{-5}$",
        "ORIGINAL_NOISE_LEVEL_MIN": "0.007",
        "ORIGINAL_SNR_CROSSING": "527",
        "MODIFIED_BETA_FIRST": "$1.00\\times10^{-15}$",
        "MODIFIED_BETA_T": "$1.00\\times10^{-5}$",
        "MODIFIED_ALPHA_BAR_T": "0.9943",
        "MODIFIED_SNR_T": "174.4",
        "SNR_POWER_LAW_T": "175",
        "XT_SIGNAL_COEF": "0.997",
        "XT_NOISE_COEF": "0.076",
        "EMBEDDING_SHIFT": "0.003",
    }


def test_snr_crossing_reads_never_when_snr_stays_high(context):
    crossing = next(s for s in SCHEDULE_NUMBER_SECTIONS if s.placeholder == "ORIGINAL_SNR_CROSSING")
    assert crossing.render(context) == "never"


def test_builder_rejects_duplicate_placeholders(tiny_config, tmp_path):
    with pytest.raises(ValueError, match="EXPONENT"):
        ReportBuilder(
            tiny_config,
            DECLARED_EXPONENT,
            build_dir=tmp_path,
            sections=(ExponentSection(), ExponentSection()),
        )


def test_builder_rejects_placeholder_the_template_cannot_match(tiny_config, tmp_path):
    with pytest.raises(ValueError, match="BETA_1"):
        ReportBuilder(
            tiny_config,
            DECLARED_EXPONENT,
            build_dir=tmp_path,
            sections=(ScheduleNumberSection("BETA_1", lambda c: 1.0, str),),
        )


def test_builder_renders_template_with_custom_sections(tiny_config, tmp_path):
    template = tmp_path / "template.md"
    template.write_text("x = {{EXPONENT}}\n", encoding="utf-8")
    builder = ReportBuilder(
        tiny_config,
        DECLARED_EXPONENT,
        template=template,
        build_dir=tmp_path / "build",
        sections=(ExponentSection(),),
    )
    report = builder.build(pdf=False)
    assert report.read_text(encoding="utf-8") == f"x = {DECLARED_EXPONENT:.2f}\n"
    assert report.with_suffix(".html").is_file()
