# ADDED: new file, not part of the upstream SR3 codebase.
"""Section of Dataset and Verification: split sizes and duplicate removal."""  # ADDED

from __future__ import annotations  # ADDED

from sr3_ablation.data.preprocess import processed_root  # ADDED
from sr3_ablation.io.jsonio import read_json  # ADDED
from sr3_ablation.reporting.sections.base import PREPARE_DATA_COMMAND, ReportContext, pending  # ADDED
from sr3_ablation.reporting.tables import markdown_table  # ADDED

SPLIT_SUMMARY_FILE = "split_summary.json"  # ADDED


class DatasetTableSection:  # ADDED
    """Images per split, plus a sentence on exact duplicates."""  # ADDED

    placeholder = "DATASET_TABLE"  # ADDED

    def render(self, context: ReportContext) -> str:  # ADDED
        data = context.config.data  # ADDED
        root = processed_root(  # ADDED
            context.config.paths.data_root,  # ADDED
            data.low_resolution,  # ADDED
            data.high_resolution,  # ADDED
        )  # ADDED
        path = root / SPLIT_SUMMARY_FILE  # ADDED
        if not path.is_file():  # ADDED
            return pending(PREPARE_DATA_COMMAND)  # ADDED
        summary = read_json(path)  # ADDED
        counts = summary["counts"]  # ADDED
        rows = [[split, count] for split, count in counts.items()]  # ADDED
        rows.append(["Total", sum(counts.values())])  # ADDED
        table = markdown_table(["Split", "Images"], rows)  # ADDED
        found = summary["images_found"]  # ADDED
        removed = len(summary["duplicates_removed"])  # ADDED
        if removed == 0:  # ADDED
            note = f"None of the {found:,} images was an exact duplicate, so all were kept."  # ADDED
        else:  # ADDED
            note = f"{removed} of the {found:,} images were exact duplicates and were removed."  # ADDED
        return f"{table}\n\n{note}"  # ADDED
