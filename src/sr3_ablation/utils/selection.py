# ADDED: new file, not part of the upstream SR3 codebase.
"""Deterministic subset selection."""  # ADDED

from __future__ import annotations  # ADDED


def evenly_spaced_indices(total: int, count: int) -> list[int]:  # ADDED
    """``count`` indices spread evenly over ``range(total)``, centred in equal-width bins.  # ADDED

    Spreading the picks over the whole sorted id list keeps small subsets (validation,  # ADDED
    snapshots, trajectories) from all coming out of one corner of the split. Returns every  # ADDED
    index if ``count`` is 0 or too large.  # ADDED
    """  # ADDED
    if count <= 0 or count >= total:  # ADDED
        return list(range(total))  # ADDED
    return [int((index + 0.5) * total / count) for index in range(count)]  # ADDED
