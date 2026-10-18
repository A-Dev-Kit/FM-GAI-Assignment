# ADDED: new file, not part of the upstream SR3 codebase.
"""Side-by-side view of our changes to the vendored upstream code.

The vendored files are byte-identical to upstream except for lines that end in an
``# ADDED`` comment, so dropping those lines gives back the upstream file exactly. Lines
marked ``# MODIFIED`` cannot be reversed that way and are rejected.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

ADDED_MARKER = "# ADDED"
MODIFIED_MARKER = "# MODIFIED"
SOURCE_SUFFIX = ".py"


@dataclass(frozen=True)
class DiffRow:
    """One row of the side-by-side listing; a ``None`` side means the line is absent there."""

    upstream_number: int | None
    upstream_code: str | None
    our_number: int
    our_code: str
    added: bool


def _is_marked(line: str, marker: str) -> bool:
    return marker in line and not line.lstrip().startswith("#")


def changed_files(root: Path) -> list[Path]:
    """Source files under ``root`` that carry at least one change marker, sorted by path."""
    return sorted(
        path
        for path in root.rglob(f"*{SOURCE_SUFFIX}")
        if any(
            _is_marked(line, marker)
            for line in path.read_text(encoding="utf-8").splitlines()
            for marker in (ADDED_MARKER, MODIFIED_MARKER)
        )
    )


def side_by_side(lines: Sequence[str], context: int) -> list[DiffRow]:
    """Rows around every ``# ADDED`` line, with ``context`` unchanged lines on each side.

    Raises:
        ValueError: if a line is marked ``# MODIFIED`` (its upstream text is not recoverable).
    """
    modified = [n for n, line in enumerate(lines, 1) if _is_marked(line, MODIFIED_MARKER)]
    if modified:
        raise ValueError(f"Lines {modified} are marked {MODIFIED_MARKER}; supply the original.")
    added = {n for n, line in enumerate(lines, 1) if _is_marked(line, ADDED_MARKER)}
    shown = sorted(
        {
            n
            for change in added
            for n in range(change - context, change + context + 1)
            if 1 <= n <= len(lines)
        }
    )
    rows = []
    for number in shown:
        code = lines[number - 1].rstrip()
        is_added = number in added
        upstream_number = None if is_added else number - sum(1 for a in added if a < number)
        rows.append(
            DiffRow(
                upstream_number=upstream_number,
                upstream_code=None if is_added else code,
                our_number=number,
                our_code=code,
                added=is_added,
            )
        )
    return rows


def added_line_numbers(rows: Iterable[DiffRow]) -> list[int]:
    """Our line numbers of the added lines in ``rows``."""
    return [row.our_number for row in rows if row.added]
