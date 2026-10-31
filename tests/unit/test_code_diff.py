# ADDED: new file, not part of the upstream SR3 codebase.
"""Side-by-side listing of the vendored changes."""  # ADDED

from __future__ import annotations  # ADDED

import pytest  # ADDED

from sr3_ablation.paths import UPSTREAM_SR3_DIR  # ADDED
from sr3_ablation.reporting.code_diff import (  # ADDED
    DiffRow,  # ADDED
    added_line_numbers,  # ADDED
    changed_files,  # ADDED
    side_by_side,  # ADDED
)  # ADDED
from sr3_ablation.reporting.sections import ReportContext  # ADDED
from sr3_ablation.reporting.sections.implementation import CodeDiffSection  # ADDED
from tests.helpers import DECLARED_EXPONENT  # ADDED

DIFFUSION = UPSTREAM_SR3_DIR / "model" / "sr3_modules" / "diffusion.py"  # ADDED
ADDED_LINES = [102, 103, 104, 105, 108, 109, 110, 111, 112]  # ADDED


def test_rows_number_both_files_around_an_added_line():  # ADDED
    lines = ["a = 1", "b = 2", "c = a + b  # ADDED: new", "d = c", "e = d"]
    assert side_by_side(lines, context=1) == [  # ADDED
        DiffRow(2, "b = 2", 2, "b = 2", added=False),  # ADDED
        DiffRow(None, None, 3, "c = a + b  # ADDED: new", added=True),
        DiffRow(3, "d = c", 4, "d = c", added=False),  # ADDED
    ]  # ADDED


def test_comment_lines_mentioning_the_marker_are_not_changes():  # ADDED
    assert side_by_side(["# ADDED lines are tagged like this", "x = 1"], context=1) == []


def test_modified_lines_are_rejected():  # ADDED
    with pytest.raises(ValueError, match="MODIFIED"):  # ADDED
        side_by_side(["x = 2  # MODIFIED: was 1"], context=0)  # ADDED


def test_changed_files_finds_only_marked_sources(tmp_path):  # ADDED
    (tmp_path / "pkg").mkdir()  # ADDED
    (tmp_path / "pkg" / "changed.py").write_text("x = 1  # ADDED\n", encoding="utf-8")
    (tmp_path / "pkg" / "same.py").write_text("x = 1\n", encoding="utf-8")  # ADDED
    (tmp_path / "notes.md").write_text("y = 2  # ADDED\n", encoding="utf-8")
    assert changed_files(tmp_path) == [tmp_path / "pkg" / "changed.py"]  # ADDED


def test_vendored_sr3_has_marked_schedule_changes():  # ADDED
    assert changed_files(UPSTREAM_SR3_DIR) == [DIFFUSION]  # ADDED
    rows = side_by_side(DIFFUSION.read_text(encoding="utf-8").splitlines(), context=3)  # ADDED
    assert added_line_numbers(rows) == ADDED_LINES  # ADDED
    run_b = next(row for row in rows if row.our_number == 105)  # ADDED
    assert "beta_exponent" in run_b.our_code or "betas ** beta_exponent" in run_b.our_code  # ADDED


def test_code_diff_section_labels_both_sides(tiny_config, tmp_path):  # ADDED
    context = ReportContext.create(tiny_config, DECLARED_EXPONENT, tmp_path)  # ADDED
    text = CodeDiffSection().render(context)  # ADDED
    assert "`third_party/sr3/model/sr3_modules/diffusion.py`, line(s) 102" in text  # ADDED
    assert "| Upstream `01d27a7` line | Original code | Our line | Modified code | Change |" in text  # ADDED
    assert "**ADDED**" in text  # ADDED


def test_code_diff_section_without_changes(tiny_config, tmp_path):  # ADDED
    context = ReportContext.create(tiny_config, DECLARED_EXPONENT, tmp_path)  # ADDED
    assert "no changes" in CodeDiffSection(upstream_root=tmp_path).render(context)  # ADDED
