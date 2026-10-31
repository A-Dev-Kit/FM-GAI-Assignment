# ADDED: new file, not part of the upstream SR3 codebase.
"""Commands: download-data, download-checkpoint, prepare-data."""  # ADDED

from __future__ import annotations  # ADDED

import shutil  # ADDED
from pathlib import Path  # ADDED

from sr3_ablation.config.schema import ExperimentConfig  # ADDED
from sr3_ablation.data.archive import list_images  # ADDED
from sr3_ablation.data.cleaning import drop_exact_duplicates  # ADDED
from sr3_ablation.data.kvasir_source import KvasirSegSource  # ADDED
from sr3_ablation.data.manifest import SPLITS, build_manifest, split_counts, write_manifest  # ADDED
from sr3_ablation.data.preprocess import layout_for, prepare_split, processed_root  # ADDED
from sr3_ablation.io.jsonio import write_json  # ADDED
from sr3_ablation.model.checkpoints import download_pretrained  # ADDED
from sr3_ablation.utils.log import configure_logging, get_logger  # ADDED

logger = get_logger("pipelines.data")  # ADDED


def download_data(config: ExperimentConfig) -> Path:  # ADDED
    """Command ``download-data``: fetch and extract Kvasir-SEG."""  # ADDED
    configure_logging()  # ADDED
    return KvasirSegSource(config.data.source_url, config.paths.data_root).ensure()  # ADDED


def download_checkpoint(config: ExperimentConfig) -> Path:  # ADDED
    """Command ``download-checkpoint``: fetch the pretrained SR3 16->128 generator."""  # ADDED
    configure_logging()  # ADDED
    return download_pretrained(  # ADDED
        config.model.pretrained_folder_url,  # ADDED
        config.model.pretrained_file,  # ADDED
        config.paths.checkpoint_dir,  # ADDED
    )  # ADDED


def prepare_data(config: ExperimentConfig, workers: int = 8) -> dict[str, int]:  # ADDED
    """Clean and split Kvasir-SEG with the configured seed, then write the SR3 16/128 layout.  # ADDED

    The manifest and split summary are also copied to ``outputs/dataset/`` as evidence.  # ADDED
    """  # ADDED
    configure_logging()  # ADDED
    data = config.data  # ADDED
    image_root = KvasirSegSource(data.source_url, config.paths.data_root).ensure()  # ADDED
    cleaned = drop_exact_duplicates(list_images(image_root))  # ADDED
    if cleaned.duplicates:  # ADDED
        logger.warning("Dropped %d exact duplicate images.", len(cleaned.duplicates))  # ADDED
    sources = [path.relative_to(image_root).as_posix() for path in cleaned.unique]  # ADDED
    entries = build_manifest(sources, data.val_size, data.test_size, data.split_seed)  # ADDED
    root = processed_root(config.paths.data_root, data.low_resolution, data.high_resolution)  # ADDED
    manifest_path = root / "manifest.csv"  # ADDED
    write_manifest(entries, manifest_path)  # ADDED

    for split in SPLITS:  # ADDED
        layout = layout_for(  # ADDED
            config.paths.data_root,  # ADDED
            split,  # ADDED
            data.low_resolution,  # ADDED
            data.high_resolution,  # ADDED
        )  # ADDED
        selected = [entry for entry in entries if entry.split == split]  # ADDED
        prepare_split(selected, image_root, layout, workers)  # ADDED
        logger.info("%s: %d images -> %s", split, len(selected), layout.root)  # ADDED

    counts = split_counts(entries)  # ADDED
    summary = {  # ADDED
        "source_root": image_root,  # ADDED
        "images_found": len(cleaned.unique) + len(cleaned.duplicates),  # ADDED
        "duplicates_removed": [path.name for path in cleaned.duplicates],  # ADDED
        "split_seed": data.split_seed,  # ADDED
        "resolution": f"{data.low_resolution} -> {data.high_resolution}",  # ADDED
        "counts": counts,  # ADDED
    }  # ADDED
    evidence_dir = config.paths.output_root / "dataset"  # ADDED
    write_json(summary, root / "split_summary.json")  # ADDED
    write_json(summary, evidence_dir / "split_summary.json")  # ADDED
    evidence_dir.mkdir(parents=True, exist_ok=True)  # ADDED
    shutil.copyfile(manifest_path, evidence_dir / "manifest.csv")  # ADDED
    logger.info("Split counts: %s", counts)  # ADDED
    return counts  # ADDED
