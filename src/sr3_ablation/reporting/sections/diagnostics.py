# ADDED: new file, not part of the upstream SR3 codebase.
"""Analysis-only section: Run B sampled from pure noise versus a forward-noised start."""  # ADDED

from __future__ import annotations  # ADDED

from typing import Any  # ADDED

from sr3_ablation.data.preprocess import layout_for  # ADDED
from sr3_ablation.io.jsonio import read_json  # ADDED
from sr3_ablation.io.naming import diagnostic_name  # ADDED
from sr3_ablation.reporting.grids import image_grid  # ADDED
from sr3_ablation.reporting.sections.base import (  # ADDED
    DIAGNOSE_START_COMMAND,  # ADDED
    ReportContext,  # ADDED
    pending,  # ADDED
)  # ADDED
from sr3_ablation.reporting.tables import markdown_table, metric_header, metric_value  # ADDED
from sr3_ablation.runs import BICUBIC_LABEL, BICUBIC_RUN, run_label, run_name  # ADDED

DIAGNOSED_KEY = "B"  # ADDED
DIAGNOSED_RUN = run_name(DIAGNOSED_KEY)  # ADDED
DIAGNOSED_LABEL = run_label(DIAGNOSED_KEY)  # ADDED
START_DIAGNOSTIC_FIGURE_FILE = "start_diagnostic.png"  # ADDED
ROW_LABELS = {  # ADDED
    BICUBIC_RUN: f"{BICUBIC_LABEL} input (no sampling)",  # ADDED
    "pure_noise": f"{DIAGNOSED_LABEL} from $x_T \\sim \\mathcal{{N}}(0, I)$ (as reported)",  # ADDED
    "forward_noised": f"{DIAGNOSED_LABEL} from $x_T \\sim q(x_T \\mid x_0 = \\text{{bicubic}})$",  # ADDED
}  # ADDED
SHORT_LABELS = {"pure_noise": "N(0, I)", "forward_noised": "q(x_T | x_0)"}  # ADDED


class StartDiagnosticSection:  # ADDED
    """Grid of x_T and x_0 for both starts plus the scores of x_0 on the same test images."""  # ADDED

    placeholder = "START_DIAGNOSTIC"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        record_path = context.run_paths(DIAGNOSED_RUN).start_diagnostic_record  # ADDED
        if not record_path.is_file():  # ADDED
            return pending(DIAGNOSE_START_COMMAND)  # ADDED
        record = read_json(record_path)  # ADDED
        return f"{self._figure(context, record)}\n\n{self._table(context, record)}"  # ADDED

    @staticmethod  # ADDED
    def _figure(context: ReportContext, record: dict[str, Any]) -> str:  # ADDED
        cfg = context.config  # ADDED
        layout = layout_for(  # ADDED
            cfg.paths.data_root,  # ADDED
            "test",  # ADDED
            cfg.data.low_resolution,  # ADDED
            cfg.data.high_resolution,  # ADDED
        )  # ADDED
        folder = context.run_paths(DIAGNOSED_RUN).diagnostics  # ADDED
        start_t = max(record["timesteps"])  # ADDED
        columns = [BICUBIC_LABEL]  # ADDED
        for start in record["starts"]:  # ADDED
            short = SHORT_LABELS.get(start, start)  # ADDED
            columns += [f"$x_{{{start_t}}}$, {short}", f"$x_0$, {short}"]  # ADDED
        columns.append("HR 128×128")  # ADDED
        rows = []  # ADDED
        for sample, image_id in enumerate(record["image_ids"], start=1):  # ADDED
            row = [layout.sr_path(image_id)]  # ADDED
            for start in record["starts"]:  # ADDED
                row += [  # ADDED
                    folder / diagnostic_name(DIAGNOSED_RUN, start, t, sample)  # ADDED
                    for t in (start_t, 0)  # ADDED
                ]  # ADDED
            rows.append([*row, layout.hr_path(image_id)])  # ADDED
        path = image_grid(  # ADDED
            rows,  # ADDED
            record["image_ids"],  # ADDED
            columns,  # ADDED
            context.assets / START_DIAGNOSTIC_FIGURE_FILE,  # ADDED
            title="Run B: the same denoiser started from two different x_T",  # ADDED
        )  # ADDED
        return context.image("Run B sampled from pure noise and from a forward-noised start", path)  # ADDED

    @staticmethod  # ADDED
    def _table(context: ReportContext, record: dict[str, Any]) -> str:  # ADDED
        metrics = list(context.config.evaluation.metrics)  # ADDED
        header = [f"Start ({len(record['image_ids'])} test images)", *map(metric_header, metrics)]  # ADDED
        rows = [  # ADDED
            [ROW_LABELS.get(name, name), *(metric_value(m, entry["mean"][m]) for m in metrics)]  # ADDED
            for name, entry in record["scores"].items()  # ADDED
        ]  # ADDED
        return markdown_table(header, rows)  # ADDED
