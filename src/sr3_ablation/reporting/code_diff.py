# ADDED: new file, not part of the upstream SR3 codebase.
"""Side-by-side view of our changes to the vendored upstream code.  # ADDED

The vendored files are byte-identical to upstream except for lines that end in an  # ADDED
``# ADDED`` comment, so dropping those lines gives back the upstream file exactly. Lines
marked ``# MODIFIED`` cannot be reversed that way and are rejected.  # ADDED
"""  # ADDED

from __future__ import annotations  # ADDED

from collections.abc import Iterable, Sequence  # ADDED
from dataclasses import dataclass  # ADDED
from pathlib import Path  # ADDED

ADDED_MARKER = "# ADDED"
MODIFIED_MARKER = "# MODIFIED"  # ADDED
SOURCE_SUFFIX = ".py"  # ADDED


@dataclass(frozen=True)  # ADDED
class DiffRow:  # ADDED
    """One row of the side-by-side listing; a ``None`` side means the line is absent there."""  # ADDED

    upstream_number: int | None  # ADDED
    upstream_code: str | None  # ADDED
    our_number: int  # ADDED
    our_code: str  # ADDED
    added: bool  # ADDED


def _is_marked(line: str, marker: str) -> bool:  # ADDED
    return marker in line and not line.lstrip().startswith("#")  # ADDED


def changed_files(root: Path) -> list[Path]:  # ADDED
    """Source files under ``root`` that carry at least one change marker, sorted by path."""  # ADDED
    return sorted(  # ADDED
        path  # ADDED
        for path in root.rglob(f"*{SOURCE_SUFFIX}")  # ADDED
        if any(  # ADDED
            _is_marked(line, marker)  # ADDED
            for line in path.read_text(encoding="utf-8").splitlines()  # ADDED
            for marker in (ADDED_MARKER, MODIFIED_MARKER)  # ADDED
        )  # ADDED
    )  # ADDED


def side_by_side(lines: Sequence[str], context: int) -> list[DiffRow]:  # ADDED
    """Rows around every ``# ADDED`` line, with ``context`` unchanged lines on each side.

    Raises:  # ADDED
        ValueError: if a line is marked ``# MODIFIED`` (its upstream text is not recoverable).  # ADDED
    """  # ADDED
    modified = [n for n, line in enumerate(lines, 1) if _is_marked(line, MODIFIED_MARKER)]  # ADDED
    if modified:  # ADDED
        raise ValueError(f"Lines {modified} are marked {MODIFIED_MARKER}; supply the original.")  # ADDED
    added = {n for n, line in enumerate(lines, 1) if _is_marked(line, ADDED_MARKER)}  # ADDED
    shown = sorted(  # ADDED
        {  # ADDED
            n  # ADDED
            for change in added  # ADDED
            for n in range(change - context, change + context + 1)  # ADDED
            if 1 <= n <= len(lines)  # ADDED
        }  # ADDED
    )  # ADDED
    rows = []  # ADDED
    for number in shown:  # ADDED
        code = lines[number - 1].rstrip()  # ADDED
        is_added = number in added  # ADDED
        upstream_number = None if is_added else number - sum(1 for a in added if a < number)  # ADDED
        rows.append(  # ADDED
            DiffRow(  # ADDED
                upstream_number=upstream_number,  # ADDED
                upstream_code=None if is_added else code,  # ADDED
                our_number=number,  # ADDED
                our_code=code,  # ADDED
                added=is_added,  # ADDED
            )  # ADDED
        )  # ADDED
    return rows  # ADDED


def added_line_numbers(rows: Iterable[DiffRow]) -> list[int]:  # ADDED
    """Our line numbers of the added lines in ``rows``."""  # ADDED
    return [row.our_number for row in rows if row.added]  # ADDED
