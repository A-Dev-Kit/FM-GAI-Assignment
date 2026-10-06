# ADDED: new file, not part of the upstream SR3 codebase.
from PIL import Image

from sr3_ablation.data import (
    AfhqSource,
    PairedSrDataset,
    build_manifest,
    layout_for,
    prepare_split,
    read_manifest,
    split_counts,
    write_manifest,
)
from sr3_ablation.data.preprocess import make_triplet, resize_and_crop
from sr3_ablation.utils.selection import evenly_spaced_indices


def test_source_locates_extracted_tree(tiny_config, fake_afhq):
    source = AfhqSource(tiny_config.data.afhq_url, tiny_config.paths.data_root)
    assert source.locate() == fake_afhq


def test_source_ignores_unfinished_extraction(tmp_path):
    (tmp_path / "raw" / "afhq" / "train").mkdir(parents=True)
    assert AfhqSource("unused", tmp_path).locate() is None


def test_manifest_is_seeded_stratified_and_disjoint(fake_afhq):
    entries = build_manifest(fake_afhq, val_size=3, test_per_class=1, seed=42)
    assert entries == build_manifest(fake_afhq, val_size=3, test_per_class=1, seed=42)
    counts = split_counts(entries)
    assert counts["val"] == {"cat": 1, "dog": 1, "wild": 1}
    assert counts["test"] == {"cat": 1, "dog": 1, "wild": 1}
    assert counts["train"] == {"cat": 3, "dog": 3, "wild": 3}
    sources = [entry.source for entry in entries]
    assert len(sources) == len(set(sources))
    assert all(e.source.startswith("val/") for e in entries if e.split == "test")
    assert all(e.source.startswith("train/") for e in entries if e.split != "test")


def test_manifest_csv_round_trip(fake_afhq, tmp_path):
    entries = build_manifest(fake_afhq, val_size=3, test_per_class=1, seed=1)
    path = tmp_path / "manifest.csv"
    write_manifest(entries, path)
    assert read_manifest(path) == entries
    assert path.read_text(encoding="utf-8").splitlines()[0] == "image_id,split,animal,source"


def test_resize_matches_sr3_protocol():
    image = Image.new("RGB", (64, 48), (120, 30, 200))
    assert resize_and_crop(image, 16).size == (16, 16)
    lr, hr, sr = make_triplet(image, 4, 16)
    assert (lr.size, hr.size, sr.size) == ((4, 4), (16, 16), (16, 16))


def test_prepare_split_writes_sr3_layout(tiny_config, fake_afhq):
    entries = [e for e in build_manifest(fake_afhq, 3, 1, 42) if e.split == "test"]
    layout = layout_for(tiny_config.paths.data_root, "test", 4, 16)
    assert prepare_split(entries, fake_afhq, layout, workers=2) == 3
    assert [p.name for p in (layout.lr_dir, layout.hr_dir, layout.sr_dir)] == [
        "lr_4",
        "hr_16",
        "sr_4_16",
    ]
    assert layout.image_ids() == ["test_00000", "test_00001", "test_00002"]

    dataset = PairedSrDataset(layout, limit=2)
    item = dataset[0]
    assert item["HR"].shape == (3, 16, 16) and item["SR"].shape == (3, 16, 16)
    assert float(item["HR"].min()) >= -1.0 and float(item["HR"].max()) <= 1.0
    assert len(dataset) == 2


def test_evenly_spaced_indices_cover_all_classes():
    assert evenly_spaced_indices(300, 3) == [50, 150, 250]
    assert evenly_spaced_indices(5, 0) == [0, 1, 2, 3, 4]
    assert evenly_spaced_indices(2, 5) == [0, 1]
