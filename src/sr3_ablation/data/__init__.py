# ADDED: new file, not part of the upstream SR3 codebase.
"""Kvasir-SEG acquisition, cleaning, seeded splitting, SR3-layout preprocessing and loading."""

from sr3_ablation.data.archive import list_images
from sr3_ablation.data.cleaning import Deduplicated, drop_exact_duplicates
from sr3_ablation.data.kvasir_source import DATASET_SLUG, KvasirSegSource
from sr3_ablation.data.loaders import PairedSrDataset, make_loader
from sr3_ablation.data.manifest import (
    SPLITS,
    ManifestEntry,
    build_manifest,
    read_manifest,
    split_counts,
    write_manifest,
)
from sr3_ablation.data.preprocess import SrLayout, layout_for, prepare_split, processed_root

__all__ = [
    "DATASET_SLUG",
    "SPLITS",
    "Deduplicated",
    "KvasirSegSource",
    "ManifestEntry",
    "PairedSrDataset",
    "SrLayout",
    "build_manifest",
    "drop_exact_duplicates",
    "layout_for",
    "list_images",
    "make_loader",
    "prepare_split",
    "processed_root",
    "read_manifest",
    "split_counts",
    "write_manifest",
]
