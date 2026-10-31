# ADDED: new file, not part of the upstream SR3 codebase.
"""Report sections render on their own and degrade to "pending" text when inputs are missing."""  # ADDED

from __future__ import annotations  # ADDED

import re  # ADDED

import pytest  # ADDED

from sr3_ablation.config import load_config  # ADDED
from sr3_ablation.paths import REPORT_DIR  # ADDED
from sr3_ablation.reporting import ReportBuilder  # ADDED
from sr3_ablation.reporting.sections import SECTION_REGISTRY, ReportContext  # ADDED
from sr3_ablation.reporting.sections.schedule import (  # ADDED
    SCHEDULE_NUMBER_SECTIONS,  # ADDED
    ExponentSection,  # ADDED
    ScheduleNumberSection,  # ADDED
)  # ADDED
from tests.helpers import DECLARED_EXPONENT  # ADDED

TEMPLATE = REPORT_DIR / "report_template.md"  # ADDED
ALWAYS_AVAILABLE = {  # ADDED
    "EXPONENT",  # ADDED
    "SCHEDULE_TABLE",  # ADDED
    "TRAJECTORY_SNR_TABLE",  # ADDED
    "SCHEDULE_FIGURE",  # ADDED
    "SETUP_TABLE",  # ADDED
    "CODE_DIFF",  # ADDED
    "QUALITATIVE_ANALYSIS",  # ADDED
    "SUPPLEMENTARY_ANALYSIS",  # ADDED
    "TAKEAWAY",  # ADDED
    "CONTRIBUTION_TABLE",  # ADDED
    *(section.placeholder for section in SCHEDULE_NUMBER_SECTIONS),  # ADDED
}  # ADDED
EMPTY_WITHOUT_INPUTS = {"LOSS_TABLE"}  # ADDED
PENDING_MARKER = "> *Pending: run `"  # ADDED


@pytest.fixture  # ADDED
def context(tiny_config, tmp_path) -> ReportContext:  # ADDED
    return ReportContext.create(tiny_config, DECLARED_EXPONENT, tmp_path / "build")  # ADDED


def test_registry_matches_template_placeholders():  # ADDED
    in_template = set(re.findall(r"\{\{([A-Z_]+)\}\}", TEMPLATE.read_text(encoding="utf-8")))  # ADDED
    registered = [section.placeholder for section in SECTION_REGISTRY]  # ADDED
    assert len(registered) == len(set(registered))  # ADDED
    assert set(registered) == in_template  # ADDED


@pytest.mark.parametrize("section", SECTION_REGISTRY, ids=lambda s: s.placeholder)  # ADDED
def test_section_without_run_outputs(section, context):  # ADDED
    text = section.render(context)  # ADDED
    if section.placeholder in ALWAYS_AVAILABLE:  # ADDED
        assert text and PENDING_MARKER not in text  # ADDED
    elif section.placeholder in EMPTY_WITHOUT_INPUTS:  # ADDED
        assert text == ""  # ADDED
    else:  # ADDED
        assert text.startswith(PENDING_MARKER)  # ADDED


def test_context_derives_both_schedules(context):  # ADDED
    assert context.original.exponent == 1.0  # ADDED
    assert context.modified.exponent == DECLARED_EXPONENT  # ADDED
    assert context.assets == context.outputs / "report_assets"  # ADDED
    assert (  # ADDED
        context.image("cap", context.assets / "a.png") == "![cap](../outputs/report_assets/a.png)"  # ADDED
    )  # ADDED


def test_exponent_has_two_decimals(context):  # ADDED
    assert ExponentSection().render(context) == f"{DECLARED_EXPONENT:.2f}"  # ADDED


def test_schedule_numbers_for_declared_exponent(tmp_path):  # ADDED
    full = ReportContext.create(load_config(local_config=None), DECLARED_EXPONENT, tmp_path)  # ADDED
    rendered = {section.placeholder: section.render(full) for section in SCHEDULE_NUMBER_SECTIONS}  # ADDED
    assert rendered == {  # ADDED
        "ORIGINAL_ALPHA_BAR_T": "$4.39\\times10^{-5}$",  # ADDED
        "ORIGINAL_NOISE_LEVEL_MIN": "0.007",  # ADDED
        "ORIGINAL_SNR_CROSSING": "527",  # ADDED
        "MODIFIED_BETA_FIRST": "$1.00\\times10^{-15}$",  # ADDED
        "MODIFIED_BETA_T": "$1.00\\times10^{-5}$",  # ADDED
        "MODIFIED_ALPHA_BAR_T": "0.9943",  # ADDED
        "MODIFIED_SNR_T": "174.4",  # ADDED
        "SNR_POWER_LAW_T": "175",  # ADDED
        "XT_SIGNAL_COEF": "0.997",  # ADDED
        "XT_NOISE_COEF": "0.076",  # ADDED
        "EMBEDDING_SHIFT": "0.003",  # ADDED
    }  # ADDED


def test_snr_crossing_reads_never_when_snr_stays_high(context):  # ADDED
    crossing = next(s for s in SCHEDULE_NUMBER_SECTIONS if s.placeholder == "ORIGINAL_SNR_CROSSING")  # ADDED
    assert crossing.render(context) == "never"  # ADDED


def test_builder_rejects_duplicate_placeholders(tiny_config, tmp_path):  # ADDED
    with pytest.raises(ValueError, match="EXPONENT"):  # ADDED
        ReportBuilder(  # ADDED
            tiny_config,  # ADDED
            DECLARED_EXPONENT,  # ADDED
            build_dir=tmp_path,  # ADDED
            sections=(ExponentSection(), ExponentSection()),  # ADDED
        )  # ADDED


def test_builder_rejects_placeholder_the_template_cannot_match(tiny_config, tmp_path):  # ADDED
    with pytest.raises(ValueError, match="BETA_1"):  # ADDED
        ReportBuilder(  # ADDED
            tiny_config,  # ADDED
            DECLARED_EXPONENT,  # ADDED
            build_dir=tmp_path,  # ADDED
            sections=(ScheduleNumberSection("BETA_1", lambda c: 1.0, str),),  # ADDED
        )  # ADDED


def test_builder_renders_template_with_custom_sections(tiny_config, tmp_path):  # ADDED
    template = tmp_path / "template.md"  # ADDED
    template.write_text("x = {{EXPONENT}}\n", encoding="utf-8")  # ADDED
    builder = ReportBuilder(  # ADDED
        tiny_config,  # ADDED
        DECLARED_EXPONENT,  # ADDED
        template=template,  # ADDED
        build_dir=tmp_path / "build",  # ADDED
        sections=(ExponentSection(),),  # ADDED
    )  # ADDED
    report = builder.build(pdf=False)  # ADDED
    assert report.read_text(encoding="utf-8") == f"x = {DECLARED_EXPONENT:.2f}\n"  # ADDED
    assert report.with_suffix(".html").is_file()  # ADDED
