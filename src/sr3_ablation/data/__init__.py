# ADDED: new file, not part of the upstream SR3 codebase.
"""Kvasir-SEG acquisition, cleaning, seeded splitting, SR3-layout preprocessing and loading."""  # ADDED

from sr3_ablation.data.archive import list_images  # ADDED
from sr3_ablation.data.cleaning import Deduplicated, drop_exact_duplicates  # ADDED
from sr3_ablation.data.kvasir_source import DATASET_SLUG, KvasirSegSource  # ADDED
from sr3_ablation.data.loaders import PairedSrDataset, make_loader  # ADDED
from sr3_ablation.data.manifest import (  # ADDED
    SPLITS,  # ADDED
    ManifestEntry,  # ADDED
    build_manifest,  # ADDED
    read_manifest,  # ADDED
    split_counts,  # ADDED
    write_manifest,  # ADDED
)  # ADDED
from sr3_ablation.data.preprocess import SrLayout, layout_for, prepare_split, processed_root  # ADDED

__all__ = [  # ADDED
    "DATASET_SLUG",  # ADDED
    "SPLITS",  # ADDED
    "Deduplicated",  # ADDED
    "KvasirSegSource",  # ADDED
    "ManifestEntry",  # ADDED
    "PairedSrDataset",  # ADDED
    "SrLayout",  # ADDED
    "build_manifest",  # ADDED
    "drop_exact_duplicates",  # ADDED
    "layout_for",  # ADDED
    "list_images",  # ADDED
    "make_loader",  # ADDED
    "prepare_split",  # ADDED
    "processed_root",  # ADDED
    "read_manifest",  # ADDED
    "split_counts",  # ADDED
    "write_manifest",  # ADDED
]  # ADDED
