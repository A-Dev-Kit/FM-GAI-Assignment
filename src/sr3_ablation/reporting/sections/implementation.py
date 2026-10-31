# ADDED: new file, not part of the upstream SR3 codebase.
"""Sections of the Implementation: what we changed upstream, and proof of the schedule used."""  # ADDED

from __future__ import annotations  # ADDED

from pathlib import Path  # ADDED

from sr3_ablation.io.jsonio import read_json  # ADDED
from sr3_ablation.paths import PROJECT_ROOT, UPSTREAM_SR3_DIR  # ADDED
from sr3_ablation.reporting.code_diff import (  # ADDED
    DiffRow,  # ADDED
    added_line_numbers,  # ADDED
    changed_files,  # ADDED
    side_by_side,  # ADDED
)  # ADDED
from sr3_ablation.reporting.sections.base import (  # ADDED
    FINETUNE_B_COMMAND,  # ADDED
    ReportContext,  # ADDED
    pending,  # ADDED
)  # ADDED
from sr3_ablation.reporting.tables import markdown_table  # ADDED
from sr3_ablation.runs import FINE_TUNED_RUNS  # ADDED

UPSTREAM_COMMIT = "01d27a7"  # ADDED
DIFF_CONTEXT_LINES = 3  # ADDED
ABSENT = "–"  # ADDED


class CodeDiffSection:  # ADDED
    """Upstream and modified lines side by side for every changed vendored file."""  # ADDED

    placeholder = "CODE_DIFF"  # ADDED

    def __init__(  # ADDED
        self,  # ADDED
        upstream_root: Path = UPSTREAM_SR3_DIR,  # ADDED
        context: int = DIFF_CONTEXT_LINES,  # ADDED
    ) -> None:  # ADDED
        self._root = upstream_root  # ADDED
        self._context = context  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        parts = [self._render_file(path) for path in changed_files(self._root)]  # ADDED
        return "\n\n".join(parts) or "We made no changes to the vendored upstream code."  # ADDED

    def _render_file(self, path: Path) -> str:  # ADDED
        rows = side_by_side(path.read_text(encoding="utf-8").splitlines(), self._context)  # ADDED
        added = ", ".join(str(n) for n in added_line_numbers(rows))  # ADDED
        name = path.relative_to(PROJECT_ROOT).as_posix()  # ADDED
        header = [  # ADDED
            f"Upstream `{UPSTREAM_COMMIT}` line",  # ADDED
            "Original code",  # ADDED
            "Our line",  # ADDED
            "Modified code",  # ADDED
            "Change",  # ADDED
        ]  # ADDED
        table = markdown_table(header, [_diff_cells(row) for row in rows])  # ADDED
        return f"**File:** `{name}`, line(s) {added} added; no line removed or modified.\n\n{table}"  # ADDED


def _diff_cells(row: DiffRow) -> list[str]:  # ADDED
    return [  # ADDED
        ABSENT if row.upstream_number is None else str(row.upstream_number),  # ADDED
        ABSENT if row.upstream_code is None else _code(row.upstream_code),  # ADDED
        str(row.our_number),  # ADDED
        _code(row.our_code),  # ADDED
        "**ADDED**" if row.added else "unchanged",  # ADDED
    ]  # ADDED


def _code(line: str) -> str:  # ADDED
    return f"`{line}`" if line.strip() else ""  # ADDED


class ScheduleEvidenceSection:  # ADDED
    """Schedule read back from each fine-tuned model against the independent recomputation."""  # ADDED

    placeholder = "SCHEDULE_EVIDENCE"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        rows = []  # ADDED
        for name, label in FINE_TUNED_RUNS.items():  # ADDED
            path = context.run_paths(name).schedule_evidence  # ADDED
            if path.is_file():  # ADDED
                evidence = read_json(path)  # ADDED
                rows.append(  # ADDED
                    [  # ADDED
                        label,  # ADDED
                        evidence["description"],  # ADDED
                        f"{evidence['observed_alpha_bar_last']:.6g}",  # ADDED
                        f"{evidence['alpha_bar_last']:.6g}",  # ADDED
                        "yes" if evidence["verified"] else "NO",  # ADDED
                        evidence["stage"],  # ADDED
                    ]  # ADDED
                )  # ADDED
        if not rows:  # ADDED
            return pending(FINETUNE_B_COMMAND)  # ADDED
        header = [  # ADDED
            "Run",  # ADDED
            "Schedule",  # ADDED
            "model $\\bar\\alpha_T$",  # ADDED
            "expected $\\bar\\alpha_T$",  # ADDED
            "verified",  # ADDED
            "checked",  # ADDED
        ]  # ADDED
        return markdown_table(header, rows)  # ADDED
