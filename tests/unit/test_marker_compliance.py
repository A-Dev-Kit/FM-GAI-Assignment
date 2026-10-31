# ADDED: new file, not part of the upstream SR3 codebase.
"""Per-line # ADDED markers must not change Python semantics."""

from __future__ import annotations  # ADDED

from sr3_ablation.paths import PROJECT_ROOT  # ADDED
from sr3_ablation.utils.marker_compliance import apply_to_tree, verify_tree  # ADDED


def test_added_files_have_inline_markers():  # ADDED
    errors = verify_tree(PROJECT_ROOT / "src" / "sr3_ablation")  # ADDED
    assert errors == [], "\n".join(errors)  # ADDED


def test_tests_tree_is_tagged():  # ADDED
    errors = verify_tree(PROJECT_ROOT / "tests")  # ADDED
    assert errors == [], "\n".join(errors)  # ADDED


def test_apply_is_idempotent():  # ADDED
    root = PROJECT_ROOT / "src" / "sr3_ablation"  # ADDED
    apply_to_tree(root)  # ADDED
    assert apply_to_tree(root) == 0  # ADDED
