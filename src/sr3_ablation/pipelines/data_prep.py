# ADDED: new file, not part of the upstream SR3 codebase.
"""Commands: download-data, download-checkpoint, prepare-data."""

from __future__ import annotations

import shutil
from pathlib import Path

from sr3_ablation.config.schema import ExperimentConfig
from sr3_ablation.data.archive import list_images
from sr3_ablation.data.cleaning import drop_exact_duplicates
from sr3_ablation.data.kvasir_source import KvasirSegSource
from sr3_ablation.data.manifest import SPLITS, build_manifest, split_counts, write_manifest
from sr3_ablation.data.preprocess import layout_for, prepare_split, processed_root
from sr3_ablation.io.jsonio import write_json
from sr3_ablation.model.checkpoints import download_pretrained
from sr3_ablation.utils.log import configure_logging, get_logger

logger = get_logger("pipelines.data")


def download_data(config: ExperimentConfig) -> Path:
    configure_logging()
    return KvasirSegSource(config.data.source_url, config.paths.data_root).ensure()


def download_checkpoint(config: ExperimentConfig) -> Path:
    configure_logging()
    return download_pretrained(
        config.model.pretrained_folder_url,
        config.model.pretrained_file,
        config.paths.checkpoint_dir,
    )


def prepare_data(config: ExperimentConfig, workers: int = 8) -> dict[str, int]:
    """Clean and split Kvasir-SEG with the configured seed, then write the SR3 16/128 layout.

    The manifest and split summary are also copied to ``outputs/dataset/`` as evidence.
    """
    configure_logging()
    data = config.data
    image_root = KvasirSegSource(data.source_url, config.paths.data_root).ensure()
    cleaned = drop_exact_duplicates(list_images(image_root))
    if cleaned.duplicates:
        logger.warning("Dropped %d exact duplicate images.", len(cleaned.duplicates))
    sources = [path.relative_to(image_root).as_posix() for path in cleaned.unique]
    entries = build_manifest(sources, data.val_size, data.test_size, data.split_seed)
    root = processed_root(config.paths.data_root, data.low_resolution, data.high_resolution)
    manifest_path = root / "manifest.csv"
    write_manifest(entries, manifest_path)

    for split in SPLITS:
        layout = layout_for(
            config.paths.data_root, split, data.low_resolution, data.high_resolution
        )
        selected = [entry for entry in entries if entry.split == split]
        prepare_split(selected, image_root, layout, workers)
        logger.info("%s: %d images -> %s", split, len(selected), layout.root)

    counts = split_counts(entries)
    summary = {
        "source_root": image_root,
        "images_found": len(cleaned.unique) + len(cleaned.duplicates),
        "duplicates_removed": [path.name for path in cleaned.duplicates],
        "split_seed": data.split_seed,
        "resolution": f"{data.low_resolution} -> {data.high_resolution}",
        "counts": counts,
    }
    evidence_dir = config.paths.output_root / "dataset"
    write_json(summary, root / "split_summary.json")
    write_json(summary, evidence_dir / "split_summary.json")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path, evidence_dir / "manifest.csv")
    logger.info("Split counts: %s", counts)
    return counts
