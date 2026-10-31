# ADDED: new file, not part of the upstream SR3 codebase.
"""Check that # ADDED markers are comment-only (AST unchanged after stripping them)."""

from __future__ import annotations  # ADDED

import ast  # ADDED
import re  # ADDED
from pathlib import Path  # ADDED

ADDED_HEADER = "# ADDED"
INLINE_ADDED = re.compile(r"\s+#\s*ADDED\b")  # ADDED


def strip_added_comments(source: str) -> str:  # ADDED
    """Remove trailing ``# ADDED…`` comments and return equivalent Python source."""
    lines: list[str] = []  # ADDED
    for line in source.splitlines():  # ADDED
        if (  # ADDED
            ADDED_HEADER in line  # ADDED
            and line.lstrip().startswith("#")  # ADDED
            and (line.strip() == ADDED_HEADER or line.strip().startswith("# ADDED:"))
        ):  # ADDED
            continue  # ADDED
        cleaned = INLINE_ADDED.sub("", line)  # ADDED
        lines.append(cleaned)  # ADDED
    return "\n".join(lines) + ("\n" if source.endswith("\n") else "")  # ADDED


def ast_dump(source: str) -> str:  # ADDED
    """Return ``ast.dump`` for ``source``, or raise ``SyntaxError``."""  # ADDED
    tree = ast.parse(source)  # ADDED
    return ast.dump(tree, include_attributes=False)  # ADDED


def added_files(root: Path) -> list[Path]:  # ADDED
    """Python files under ``root`` whose first line is an ``# ADDED`` header."""
    files: list[Path] = []  # ADDED
    for path in sorted(root.rglob("*.py")):  # ADDED
        text = path.read_text(encoding="utf-8")  # ADDED
        first = next((line for line in text.splitlines() if line.strip()), "")  # ADDED
        if first.lstrip().startswith(ADDED_HEADER):  # ADDED
            files.append(path)  # ADDED
    return files  # ADDED


def lines_missing_marker(path: Path) -> list[int]:  # ADDED
    """1-based line numbers that are non-empty code/doc lines without an ``# ADDED`` tag."""
    missing: list[int] = []  # ADDED
    for index, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):  # ADDED
        stripped = line.strip()  # ADDED
        if not stripped:  # ADDED
            continue  # ADDED
        if ADDED_HEADER in line:  # ADDED
            continue  # ADDED
        missing.append(index)  # ADDED
    return missing  # ADDED


def ensure_inline_markers(path: Path) -> bool:  # ADDED
    """Append ``  # ADDED`` to lines that lack a marker; return True if the file changed."""
    original = path.read_text(encoding="utf-8")  # ADDED
    if not original.lstrip().startswith(ADDED_HEADER):  # ADDED
        return False  # ADDED
    before = ast_dump(strip_added_comments(original))  # ADDED
    lines = original.splitlines()  # ADDED
    changed = False  # ADDED
    for index, line in enumerate(lines):  # ADDED
        if not line.strip() or ADDED_HEADER in line:  # ADDED
            continue  # ADDED
        lines[index] = f"{line.rstrip()}  # ADDED"
        changed = True  # ADDED
    if not changed:  # ADDED
        return False  # ADDED
    updated = "\n".join(lines) + ("\n" if original.endswith("\n") else "")  # ADDED
    after = ast_dump(strip_added_comments(updated))  # ADDED
    if before != after:  # ADDED
        raise ValueError(f"Marker edit changed AST for {path}")  # ADDED
    path.write_text(updated, encoding="utf-8")  # ADDED
    return True  # ADDED


def apply_to_tree(root: Path) -> int:  # ADDED
    """Tag every ``# ADDED`` file under ``root``; return the number of files updated."""
    count = 0  # ADDED
    for path in added_files(root):  # ADDED
        if ensure_inline_markers(path):  # ADDED
            count += 1  # ADDED
    return count  # ADDED


def verify_tree(root: Path) -> list[str]:  # ADDED
    """Return human-readable violations (empty if compliant)."""  # ADDED
    errors: list[str] = []  # ADDED
    for path in added_files(root):  # ADDED
        source = path.read_text(encoding="utf-8")  # ADDED
        try:  # ADDED
            ast.parse(strip_added_comments(source))  # ADDED
        except SyntaxError as error:  # ADDED
            errors.append(f"{path}: stripped source does not parse ({error})")  # ADDED
            continue  # ADDED
        missing = lines_missing_marker(path)  # ADDED
        if missing:  # ADDED
            errors.append(f"{path}: lines without # ADDED marker: {missing[:5]}{'…' if len(missing) > 5 else ''}")
    return errors  # ADDED
