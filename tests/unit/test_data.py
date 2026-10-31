# ADDED: new file, not part of the upstream SR3 codebase.
import pytest  # ADDED
from PIL import Image  # ADDED

from sr3_ablation.data import (  # ADDED
    KvasirSegSource,  # ADDED
    PairedSrDataset,  # ADDED
    build_manifest,  # ADDED
    drop_exact_duplicates,  # ADDED
    layout_for,  # ADDED
    list_images,  # ADDED
    prepare_split,  # ADDED
    read_manifest,  # ADDED
    split_counts,  # ADDED
    write_manifest,  # ADDED
)  # ADDED
from sr3_ablation.data.preprocess import make_triplet, resize_and_crop  # ADDED
from sr3_ablation.utils.selection import evenly_spaced_indices  # ADDED
from tests.helpers import FAKE_DUPLICATE  # ADDED

SOURCES = [f"frame_{index:03d}.jpg" for index in range(15)]  # ADDED


def test_source_locates_images_folder(tiny_config, fake_kvasir):  # ADDED
    source = KvasirSegSource(tiny_config.data.source_url, tiny_config.paths.data_root)  # ADDED
    assert source.locate() == fake_kvasir  # ADDED
    assert fake_kvasir.name == "images"  # ADDED


def test_source_ignores_unfinished_extraction(tmp_path):  # ADDED
    (tmp_path / "raw" / "kvasir-seg" / "Kvasir-SEG" / "images").mkdir(parents=True)  # ADDED
    assert KvasirSegSource("unused", tmp_path).locate() is None  # ADDED


def test_exact_duplicates_are_dropped(fake_kvasir):  # ADDED
    cleaned = drop_exact_duplicates(list_images(fake_kvasir))  # ADDED
    assert [path.name for path in cleaned.duplicates] == [FAKE_DUPLICATE]  # ADDED
    assert [path.name for path in cleaned.unique] == SOURCES  # ADDED


def test_manifest_is_seeded_sized_and_disjoint():  # ADDED
    entries = build_manifest(SOURCES, val_size=3, test_size=3, seed=42)  # ADDED
    assert entries == build_manifest(SOURCES, val_size=3, test_size=3, seed=42)  # ADDED
    assert entries != build_manifest(SOURCES, val_size=3, test_size=3, seed=7)  # ADDED
    assert split_counts(entries) == {"train": 9, "val": 3, "test": 3}  # ADDED
    assert sorted(entry.source for entry in entries) == SOURCES  # ADDED


def test_manifest_puts_test_and_val_after_training_block():  # ADDED
    order = sorted(SOURCES)  # ADDED
    import random  # ADDED

    random.Random(42).shuffle(order)  # ADDED
    entries = build_manifest(SOURCES, val_size=3, test_size=3, seed=42)  # ADDED
    test_sources = {entry.source for entry in entries if entry.split == "test"}  # ADDED
    assert test_sources == set(order[-3:])  # ADDED
    val_sources = {entry.source for entry in entries if entry.split == "val"}  # ADDED
    assert val_sources == set(order[-6:-3])  # ADDED


def test_manifest_rejects_split_without_training_images():  # ADDED
    with pytest.raises(ValueError, match="no training images"):  # ADDED
        build_manifest(SOURCES, val_size=8, test_size=7, seed=42)  # ADDED


def test_manifest_csv_round_trip(tmp_path):  # ADDED
    entries = build_manifest(SOURCES, val_size=3, test_size=3, seed=1)  # ADDED
    path = tmp_path / "manifest.csv"  # ADDED
    write_manifest(entries, path)  # ADDED
    assert read_manifest(path) == entries  # ADDED
    assert path.read_text(encoding="utf-8").splitlines()[0] == "image_id,split,source"  # ADDED


def test_resize_matches_sr3_protocol():  # ADDED
    image = Image.new("RGB", (64, 48), (120, 30, 200))  # ADDED
    assert resize_and_crop(image, 16).size == (16, 16)  # ADDED
    lr, hr, sr = make_triplet(image, 4, 16)  # ADDED
    assert (lr.size, hr.size, sr.size) == ((4, 4), (16, 16), (16, 16))  # ADDED


def test_prepare_split_writes_sr3_layout(tiny_config, fake_kvasir):  # ADDED
    entries = [e for e in build_manifest(SOURCES, 3, 3, 42) if e.split == "test"]  # ADDED
    layout = layout_for(tiny_config.paths.data_root, "test", 4, 16)  # ADDED
    assert prepare_split(entries, fake_kvasir, layout, workers=2) == 3  # ADDED
    assert [p.name for p in (layout.lr_dir, layout.hr_dir, layout.sr_dir)] == [  # ADDED
        "lr_4",  # ADDED
        "hr_16",  # ADDED
        "sr_4_16",  # ADDED
    ]  # ADDED
    assert layout.root.parent.name == "kvasir_seg_4_16"  # ADDED
    assert layout.image_ids() == ["test_00000", "test_00001", "test_00002"]  # ADDED

    dataset = PairedSrDataset(layout, limit=2)  # ADDED
    item = dataset[0]  # ADDED
    assert item["HR"].shape == (3, 16, 16) and item["SR"].shape == (3, 16, 16)  # ADDED
    assert float(item["HR"].min()) >= -1.0 and float(item["HR"].max()) <= 1.0  # ADDED
    assert len(dataset) == 2  # ADDED


def test_evenly_spaced_indices_spread_over_the_split():  # ADDED
    assert evenly_spaced_indices(300, 3) == [50, 150, 250]  # ADDED
    assert evenly_spaced_indices(5, 0) == [0, 1, 2, 3, 4]  # ADDED
    assert evenly_spaced_indices(2, 5) == [0, 1]  # ADDED
