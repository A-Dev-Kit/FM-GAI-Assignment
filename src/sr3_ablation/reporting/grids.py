# ADDED: new file, not part of the upstream SR3 codebase.
"""Labelled image grids for the report (missing cells are drawn blank)."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from sr3_ablation.io.images import load_image

DPI = 160


def image_grid(
    rows: Sequence[Sequence[Path | None]],
    row_labels: Sequence[str],
    column_labels: Sequence[str],
    output: Path,
    title: str | None = None,
    cell_inches: float = 1.55,
) -> Path:
    """Draw ``rows`` of image files; 16x16 inputs are shown with nearest-neighbour pixels."""
    n_rows, n_cols = len(rows), len(column_labels)
    figure, axes = plt.subplots(
        n_rows,
        n_cols,
        figsize=(cell_inches * n_cols + 0.9, cell_inches * n_rows + (0.5 if title else 0.2)),
        squeeze=False,
    )
    for r, row in enumerate(rows):
        for c in range(n_cols):
            axis = axes[r][c]
            axis.set_xticks([])
            axis.set_yticks([])
            path = row[c] if c < len(row) else None
            if path is not None and path.is_file():
                axis.imshow(load_image(path), interpolation="nearest")
            else:
                axis.text(0.5, 0.5, "n/a", ha="center", va="center", color="grey")
            if r == 0:
                axis.set_title(column_labels[c], fontsize=9)
            if c == 0:
                axis.set_ylabel(row_labels[r], fontsize=9)
    if title:
        figure.suptitle(title, fontsize=11)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.tight_layout()
    figure.savefig(output, dpi=DPI)
    plt.close(figure)
    return output
