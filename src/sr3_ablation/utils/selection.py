# ADDED: new file, not part of the upstream SR3 codebase.
"""Deterministic subset selection."""

from __future__ import annotations


def evenly_spaced_indices(total: int, count: int) -> list[int]:
    """``count`` indices spread evenly over ``range(total)``, centred in equal-width bins.

    Spreading the picks over the whole sorted id list keeps small subsets (validation,
    snapshots, trajectories) from all coming out of one corner of the split. Returns every
    index if ``count`` is 0 or too large.
    """
    if count <= 0 or count >= total:
        return list(range(total))
    return [int((index + 0.5) * total / count) for index in range(count)]
