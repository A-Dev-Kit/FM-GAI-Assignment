# ADDED: new file, not part of the upstream SR3 codebase.
"""Commands: download-data, download-checkpoint, prepare-data."""

from __future__ import annotations

import shutil
from pathlib import Path

from sr3_ablation.config.schema import ExperimentConfig
from sr3_ablation.data.afhq_source import AfhqSource
from sr3_ablation.data.manifest import SPLITS, build_manifest, split_counts, write_manifest
from sr3_ablation.data.preprocess import layout_for, prepare_split, processed_root
from sr3_ablation.io.jsonio import write_json
from sr3_ablation.model.checkpoints import download_pretrained
from sr3_ablation.utils.log import configure_logging, get_logger

logger = get_logger("pipelines.data")


def download_data(config: ExperimentConfig) -> Path:
    configure_logging()
    return AfhqSource(config.data.afhq_url, config.paths.data_root).ensure()


def download_checkpoint(config: ExperimentConfig) -> Path:
    configure_logging()
    return download_pretrained(
        config.model.pretrained_folder_url,
        config.model.pretrained_file,
        config.paths.checkpoint_dir,
    )


def prepare_data(config: ExperimentConfig, workers: int = 8) -> dict[str, dict[str, int]]:
    """Split AFHQ with the configured seed and write the SR3 16/128 layout for each split.

    The manifest and split counts are also copied to ``outputs/dataset/`` as evidence.
    """
    configure_logging()
    data = config.data
    afhq_root = AfhqSource(data.afhq_url, config.paths.data_root).ensure()
    entries = build_manifest(afhq_root, data.val_size, data.test_per_class, data.split_seed)
    root = processed_root(config.paths.data_root, data.low_resolution, data.high_resolution)
    manifest_path = root / "manifest.csv"
    write_manifest(entries, manifest_path)

    for split in SPLITS:
        layout = layout_for(
            config.paths.data_root, split, data.low_resolution, data.high_resolution
        )
        selected = [entry for entry in entries if entry.split == split]
        prepare_split(selected, afhq_root, layout, workers)
        logger.info("%s: %d images -> %s", split, len(selected), layout.root)

    counts = split_counts(entries)
    summary = {
        "afhq_root": afhq_root,
        "split_seed": data.split_seed,
        "resolution": f"{data.low_resolution} -> {data.high_resolution}",
        "counts": counts,
        "totals": {split: sum(per_class.values()) for split, per_class in counts.items()},
    }
    evidence_dir = config.paths.output_root / "dataset"
    write_json(summary, root / "split_summary.json")
    write_json(summary, evidence_dir / "split_summary.json")
    evidence_dir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(manifest_path, evidence_dir / "manifest.csv")
    logger.info("Split totals: %s", summary["totals"])
    return counts
