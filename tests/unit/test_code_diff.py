# ADDED: new file, not part of the upstream SR3 codebase.
"""Side-by-side listing of the vendored changes."""

from __future__ import annotations

import pytest

from sr3_ablation.paths import UPSTREAM_SR3_DIR
from sr3_ablation.reporting.code_diff import (
    DiffRow,
    added_line_numbers,
    changed_files,
    side_by_side,
)
from sr3_ablation.reporting.sections import ReportContext
from sr3_ablation.reporting.sections.implementation import CodeDiffSection
from tests.helpers import DECLARED_EXPONENT

DIFFUSION = UPSTREAM_SR3_DIR / "model" / "sr3_modules" / "diffusion.py"
ADDED_LINE = 102


def test_rows_number_both_files_around_an_added_line():
    lines = ["a = 1", "b = 2", "c = a + b  # ADDED: new", "d = c", "e = d"]
    assert side_by_side(lines, context=1) == [
        DiffRow(2, "b = 2", 2, "b = 2", added=False),
        DiffRow(None, None, 3, "c = a + b  # ADDED: new", added=True),
        DiffRow(3, "d = c", 4, "d = c", added=False),
    ]


def test_comment_lines_mentioning_the_marker_are_not_changes():
    assert side_by_side(["# ADDED lines are tagged like this", "x = 1"], context=1) == []


def test_modified_lines_are_rejected():
    with pytest.raises(ValueError, match="MODIFIED"):
        side_by_side(["x = 2  # MODIFIED: was 1"], context=0)


def test_changed_files_finds_only_marked_sources(tmp_path):
    (tmp_path / "pkg").mkdir()
    (tmp_path / "pkg" / "changed.py").write_text("x = 1  # ADDED\n", encoding="utf-8")
    (tmp_path / "pkg" / "same.py").write_text("x = 1\n", encoding="utf-8")
    (tmp_path / "notes.md").write_text("y = 2  # ADDED\n", encoding="utf-8")
    assert changed_files(tmp_path) == [tmp_path / "pkg" / "changed.py"]


def test_vendored_sr3_has_exactly_one_added_line():
    assert changed_files(UPSTREAM_SR3_DIR) == [DIFFUSION]
    rows = side_by_side(DIFFUSION.read_text(encoding="utf-8").splitlines(), context=3)
    assert added_line_numbers(rows) == [ADDED_LINE]
    after = next(row for row in rows if row.our_number == ADDED_LINE + 1)
    assert after.upstream_number == ADDED_LINE
    assert "alphas = 1. - betas" in after.our_code


def test_code_diff_section_labels_both_sides(tiny_config, tmp_path):
    context = ReportContext.create(tiny_config, DECLARED_EXPONENT, tmp_path)
    text = CodeDiffSection().render(context)
    assert "`third_party/sr3/model/sr3_modules/diffusion.py`, line(s) 102 added" in text
    assert "| Upstream `01d27a7` line | Original code | Our line | Modified code | Change |" in text
    assert "| – | – | 102 |" in text and "**ADDED**" in text


def test_code_diff_section_without_changes(tiny_config, tmp_path):
    context = ReportContext.create(tiny_config, DECLARED_EXPONENT, tmp_path)
    assert "no changes" in CodeDiffSection(upstream_root=tmp_path).render(context)
