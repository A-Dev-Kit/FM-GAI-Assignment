# ADDED: new file, not part of the upstream SR3 codebase.
"""Sections of the Implementation: what we changed upstream, and proof of the schedule used."""

from __future__ import annotations

from pathlib import Path

from sr3_ablation.io.jsonio import read_json
from sr3_ablation.paths import PROJECT_ROOT, UPSTREAM_SR3_DIR
from sr3_ablation.reporting.code_diff import (
    DiffRow,
    added_line_numbers,
    changed_files,
    side_by_side,
)
from sr3_ablation.reporting.sections.base import (
    FINETUNE_B_COMMAND,
    ReportContext,
    pending,
)
from sr3_ablation.reporting.tables import markdown_table
from sr3_ablation.runs import FINE_TUNED_RUNS

UPSTREAM_COMMIT = "01d27a7"
DIFF_CONTEXT_LINES = 3
ABSENT = "–"


class CodeDiffSection:
    """Upstream and modified lines side by side for every changed vendored file."""

    placeholder = "CODE_DIFF"

    def __init__(
        self, upstream_root: Path = UPSTREAM_SR3_DIR, context: int = DIFF_CONTEXT_LINES
    ) -> None:
        self._root = upstream_root
        self._context = context

    def render(self, context: ReportContext) -> str:
        parts = [self._render_file(path) for path in changed_files(self._root)]
        return "\n\n".join(parts) or "We made no changes to the vendored upstream code."

    def _render_file(self, path: Path) -> str:
        rows = side_by_side(path.read_text(encoding="utf-8").splitlines(), self._context)
        added = ", ".join(str(n) for n in added_line_numbers(rows))
        name = path.relative_to(PROJECT_ROOT).as_posix()
        header = [
            f"Upstream `{UPSTREAM_COMMIT}` line",
            "Original code",
            "Our line",
            "Modified code",
            "Change",
        ]
        table = markdown_table(header, [_diff_cells(row) for row in rows])
        return f"**File:** `{name}`, line(s) {added} added; no line removed or modified.\n\n{table}"


def _diff_cells(row: DiffRow) -> list[str]:
    return [
        ABSENT if row.upstream_number is None else str(row.upstream_number),
        ABSENT if row.upstream_code is None else _code(row.upstream_code),
        str(row.our_number),
        _code(row.our_code),
        "**ADDED**" if row.added else "unchanged",
    ]


def _code(line: str) -> str:
    return f"`{line}`" if line.strip() else ""


class ScheduleEvidenceSection:
    """Schedule read back from each fine-tuned model against the independent recomputation."""

    placeholder = "SCHEDULE_EVIDENCE"

    def render(self, context: ReportContext) -> str:
        rows = []
        for name, label in FINE_TUNED_RUNS.items():
            path = context.run_paths(name).schedule_evidence
            if path.is_file():
                evidence = read_json(path)
                rows.append(
                    [
                        label,
                        evidence["description"],
                        f"{evidence['observed_alpha_bar_last']:.6g}",
                        f"{evidence['alpha_bar_last']:.6g}",
                        "yes" if evidence["verified"] else "NO",
                        evidence["stage"],
                    ]
                )
        if not rows:
            return pending(FINETUNE_B_COMMAND)
        header = [
            "Run",
            "Schedule",
            "model $\\bar\\alpha_T$",
            "expected $\\bar\\alpha_T$",
            "verified",
            "checked",
        ]
        return markdown_table(header, rows)
