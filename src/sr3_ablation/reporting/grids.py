# ADDED: new file, not part of the upstream SR3 codebase.
"""Labelled image grids for the report (missing cells are drawn blank)."""  # ADDED

from __future__ import annotations  # ADDED

from collections.abc import Sequence  # ADDED
from pathlib import Path  # ADDED

import matplotlib  # ADDED

matplotlib.use("Agg")  # ADDED
import matplotlib.pyplot as plt  # ADDED

from sr3_ablation.io.images import load_image  # ADDED

DPI = 160  # ADDED


def image_grid(  # ADDED
    rows: Sequence[Sequence[Path | None]],  # ADDED
    row_labels: Sequence[str],  # ADDED
    column_labels: Sequence[str],  # ADDED
    output: Path,  # ADDED
    title: str | None = None,  # ADDED
    cell_inches: float = 1.55,  # ADDED
) -> Path:  # ADDED
    """Draw ``rows`` of image files; 16x16 inputs are shown with nearest-neighbour pixels."""  # ADDED
    n_rows, n_cols = len(rows), len(column_labels)  # ADDED
    figure, axes = plt.subplots(  # ADDED
        n_rows,  # ADDED
        n_cols,  # ADDED
        figsize=(cell_inches * n_cols + 0.9, cell_inches * n_rows + (0.5 if title else 0.2)),  # ADDED
        squeeze=False,  # ADDED
    )  # ADDED
    for r, row in enumerate(rows):  # ADDED
        for c in range(n_cols):  # ADDED
            axis = axes[r][c]  # ADDED
            axis.set_xticks([])  # ADDED
            axis.set_yticks([])  # ADDED
            path = row[c] if c < len(row) else None  # ADDED
            if path is not None and path.is_file():  # ADDED
                axis.imshow(load_image(path), interpolation="nearest")  # ADDED
            else:  # ADDED
                axis.text(0.5, 0.5, "n/a", ha="center", va="center", color="grey")  # ADDED
            if r == 0:  # ADDED
                axis.set_title(column_labels[c], fontsize=9)  # ADDED
            if c == 0:  # ADDED
                axis.set_ylabel(row_labels[r], fontsize=9)  # ADDED
    if title:  # ADDED
        figure.suptitle(title, fontsize=11)  # ADDED
    output.parent.mkdir(parents=True, exist_ok=True)  # ADDED
    figure.tight_layout()  # ADDED
    figure.savefig(output, dpi=DPI)  # ADDED
    plt.close(figure)  # ADDED
    return output  # ADDED
