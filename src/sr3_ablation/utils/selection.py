# ADDED: new file, not part of the upstream SR3 codebase.
"""Deterministic subset selection."""

from __future__ import annotations


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    """``count`` indices spread evenly over ``range(total)``, centred in equal-width bins.

    Image ids are ordered by class (cat, dog, wild), so this picks every class for small
    subsets instead of only the first one. Returns every index if ``count`` is 0 or too large.
    """
    if count <= 0 or count >= total:
        return list(range(total))
    return [int((index + 0.5) * total / count) for index in range(count)]
