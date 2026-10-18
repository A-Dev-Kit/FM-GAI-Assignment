# ADDED: new file, not part of the upstream SR3 codebase.
"""Section of Dataset and Verification: split sizes and duplicate removal."""

from __future__ import annotations

from sr3_ablation.data.preprocess import processed_root
from sr3_ablation.io.jsonio import read_json
from sr3_ablation.reporting.sections.base import PREPARE_DATA_COMMAND, ReportContext, pending
from sr3_ablation.reporting.tables import markdown_table

SPLIT_SUMMARY_FILE = "split_summary.json"


class DatasetTableSection:
    """Images per split, plus a sentence on exact duplicates."""

    placeholder = "DATASET_TABLE"

    def render(self, context: ReportContext) -> str:
        data = context.config.data
        root = processed_root(
            context.config.paths.data_root, data.low_resolution, data.high_resolution
        )
        path = root / SPLIT_SUMMARY_FILE
        if not path.is_file():
            return pending(PREPARE_DATA_COMMAND)
        summary = read_json(path)
        counts = summary["counts"]
        rows = [[split, count] for split, count in counts.items()]
        rows.append(["Total", sum(counts.values())])
        table = markdown_table(["Split", "Images"], rows)
        found = summary["images_found"]
        removed = len(summary["duplicates_removed"])
        if removed == 0:
            note = f"None of the {found:,} images was an exact duplicate, so all were kept."
        else:
            note = f"{removed} of the {found:,} images were exact duplicates and were removed."
        return f"{table}\n\n{note}"
