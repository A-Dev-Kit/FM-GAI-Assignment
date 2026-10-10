# ADDED: new file, not part of the upstream SR3 codebase.
import pytest
from PIL import Image

from sr3_ablation.data import (
    KvasirSegSource,
    PairedSrDataset,
    build_manifest,
    drop_exact_duplicates,
    layout_for,
    list_images,
    prepare_split,
    read_manifest,
    split_counts,
    write_manifest,
)
from sr3_ablation.data.preprocess import make_triplet, resize_and_crop
from sr3_ablation.utils.selection import evenly_spaced_indices
from tests.helpers import FAKE_DUPLICATE

SOURCES = [f"frame_{index:03d}.jpg" for index in range(15)]


def test_source_locates_images_folder(tiny_config, fake_kvasir):
    source = KvasirSegSource(tiny_config.data.source_url, tiny_config.paths.data_root)
    assert source.locate() == fake_kvasir
    assert fake_kvasir.name == "images"


def test_source_ignores_unfinished_extraction(tmp_path):
    (tmp_path / "raw" / "kvasir-seg" / "Kvasir-SEG" / "images").mkdir(parents=True)
    assert KvasirSegSource("unused", tmp_path).locate() is None


def test_exact_duplicates_are_dropped(fake_kvasir):
    cleaned = drop_exact_duplicates(list_images(fake_kvasir))
    assert [path.name for path in cleaned.duplicates] == [FAKE_DUPLICATE]
    assert [path.name for path in cleaned.unique] == SOURCES


def test_manifest_is_seeded_sized_and_disjoint():
    entries = build_manifest(SOURCES, val_size=3, test_size=3, seed=42)
    assert entries == build_manifest(SOURCES, val_size=3, test_size=3, seed=42)
    assert entries != build_manifest(SOURCES, val_size=3, test_size=3, seed=7)
    assert split_counts(entries) == {"train": 9, "val": 3, "test": 3}
    assert sorted(entry.source for entry in entries) == SOURCES


def test_manifest_rejects_split_without_training_images():
    with pytest.raises(ValueError, match="no training images"):
        build_manifest(SOURCES, val_size=8, test_size=7, seed=42)


def test_manifest_csv_round_trip(tmp_path):
    entries = build_manifest(SOURCES, val_size=3, test_size=3, seed=1)
    path = tmp_path / "manifest.csv"
    write_manifest(entries, path)
    assert read_manifest(path) == entries
    assert path.read_text(encoding="utf-8").splitlines()[0] == "image_id,split,source"


def test_resize_matches_sr3_protocol():
    image = Image.new("RGB", (64, 48), (120, 30, 200))
    assert resize_and_crop(image, 16).size == (16, 16)
    lr, hr, sr = make_triplet(image, 4, 16)
    assert (lr.size, hr.size, sr.size) == ((4, 4), (16, 16), (16, 16))


def test_prepare_split_writes_sr3_layout(tiny_config, fake_kvasir):
    entries = [e for e in build_manifest(SOURCES, 3, 3, 42) if e.split == "test"]
    layout = layout_for(tiny_config.paths.data_root, "test", 4, 16)
    assert prepare_split(entries, fake_kvasir, layout, workers=2) == 3
    assert [p.name for p in (layout.lr_dir, layout.hr_dir, layout.sr_dir)] == [
        "lr_4",
        "hr_16",
        "sr_4_16",
    ]
    assert layout.root.parent.name == "kvasir_seg_4_16"
    assert layout.image_ids() == ["test_00000", "test_00001", "test_00002"]

    dataset = PairedSrDataset(layout, limit=2)
    item = dataset[0]
    assert item["HR"].shape == (3, 16, 16) and item["SR"].shape == (3, 16, 16)
    assert float(item["HR"].min()) >= -1.0 and float(item["HR"].max()) <= 1.0
    assert len(dataset) == 2


def test_evenly_spaced_indices_spread_over_the_split():
    assert evenly_spaced_indices(300, 3) == [50, 150, 250]
    assert evenly_spaced_indices(5, 0) == [0, 1, 2, 3, 4]
    assert evenly_spaced_indices(2, 5) == [0, 1]
