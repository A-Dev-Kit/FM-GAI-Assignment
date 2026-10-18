# ADDED: new file, not part of the upstream SR3 codebase.
"""Analysis-only section: Run B sampled from pure noise versus a forward-noised start."""

from __future__ import annotations

from typing import Any

from sr3_ablation.data.preprocess import layout_for
from sr3_ablation.io.jsonio import read_json
from sr3_ablation.io.naming import diagnostic_name
from sr3_ablation.reporting.grids import image_grid
from sr3_ablation.reporting.sections.base import (
    DIAGNOSE_START_COMMAND,
    ReportContext,
    pending,
)
from sr3_ablation.reporting.tables import markdown_table, metric_header, metric_value
from sr3_ablation.runs import BICUBIC_LABEL, BICUBIC_RUN, run_label, run_name

DIAGNOSED_KEY = "B"
DIAGNOSED_RUN = run_name(DIAGNOSED_KEY)
DIAGNOSED_LABEL = run_label(DIAGNOSED_KEY)
START_DIAGNOSTIC_FIGURE_FILE = "start_diagnostic.png"
ROW_LABELS = {
    BICUBIC_RUN: f"{BICUBIC_LABEL} input (no sampling)",
    "pure_noise": f"{DIAGNOSED_LABEL} from $x_T \\sim \\mathcal{{N}}(0, I)$ (as reported)",
    "forward_noised": f"{DIAGNOSED_LABEL} from $x_T \\sim q(x_T \\mid x_0 = \\text{{bicubic}})$",
}
SHORT_LABELS = {"pure_noise": "N(0, I)", "forward_noised": "q(x_T | x_0)"}


class StartDiagnosticSection:
    """Grid of x_T and x_0 for both starts plus the scores of x_0 on the same test images."""

    placeholder = "START_DIAGNOSTIC"

    def render(self, context: ReportContext) -> str:
        record_path = context.run_paths(DIAGNOSED_RUN).start_diagnostic_record
        if not record_path.is_file():
            return pending(DIAGNOSE_START_COMMAND)
        record = read_json(record_path)
        return f"{self._figure(context, record)}\n\n{self._table(context, record)}"

    @staticmethod
    def _figure(context: ReportContext, record: dict[str, Any]) -> str:
        cfg = context.config
        layout = layout_for(
            cfg.paths.data_root, "test", cfg.data.low_resolution, cfg.data.high_resolution
        )
        folder = context.run_paths(DIAGNOSED_RUN).diagnostics
        start_t = max(record["timesteps"])
        columns = [BICUBIC_LABEL]
        for start in record["starts"]:
            short = SHORT_LABELS.get(start, start)
            columns += [f"$x_{{{start_t}}}$, {short}", f"$x_0$, {short}"]
        columns.append("HR 128×128")
        rows = []
        for sample, image_id in enumerate(record["image_ids"], start=1):
            row = [layout.sr_path(image_id)]
            for start in record["starts"]:
                row += [
                    folder / diagnostic_name(DIAGNOSED_RUN, start, t, sample) for t in (start_t, 0)
                ]
            rows.append([*row, layout.hr_path(image_id)])
        path = image_grid(
            rows,
            record["image_ids"],
            columns,
            context.assets / START_DIAGNOSTIC_FIGURE_FILE,
            title="Run B: the same denoiser started from two different x_T",
        )
        return context.image("Run B sampled from pure noise and from a forward-noised start", path)

    @staticmethod
    def _table(context: ReportContext, record: dict[str, Any]) -> str:
        metrics = list(context.config.evaluation.metrics)
        header = [f"Start ({len(record['image_ids'])} test images)", *map(metric_header, metrics)]
        rows = [
            [ROW_LABELS.get(name, name), *(metric_value(m, entry["mean"][m]) for m in metrics)]
            for name, entry in record["scores"].items()
        ]
        return markdown_table(header, rows)
