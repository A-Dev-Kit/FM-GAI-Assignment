# ADDED: new file, not part of the upstream SR3 codebase.
"""Submission zip: exclusions, selectors, completeness checks and the archive layout."""

from __future__ import annotations

import zipfile
from pathlib import Path

import pytest

from sr3_ablation.config import load_config
from sr3_ablation.paths import PROJECT_ROOT
from sr3_ablation.submission import (
    DEFAULT_EXCLUSIONS,
    Artifact,
    IncompleteSubmissionError,
    Requirement,
    SubmissionPackager,
    brief_requirements,
)
from sr3_ablation.submission.selectors import (
    CodeSelector,
    DatasetManifestSelector,
    ReportSelector,
    RunArtifactSelector,
    git_tracked_files,
)

ROOT_FOLDER = "Group_X"


@pytest.mark.parametrize(
    ("name", "excluded"),
    [
        ("outputs/runA/checkpoints/final_gen.pth", True),
        ("outputs/runA/test_outputs/frame.jpg", True),
        ("data/raw/kvasir-seg/Kvasir-SEG/images/a.png", True),
        ("src/sr3_ablation/data/loaders.py", False),
        ("src/sr3_ablation/__pycache__/cli.cpython-314.pyc", True),
        ("configs/local.toml", True),
        ("configs/base.toml", False),
        ("outputs/runB/samples/runB_epoch12_sample1.png", False),
    ],
)
def test_default_exclusions(name, excluded):
    assert DEFAULT_EXCLUSIONS.excludes(name) is excluded


def _touch(path: Path, text: str = "x") -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text, encoding="utf-8")
    return path


@pytest.fixture
def tree(tmp_path) -> dict[str, Path]:
    project = tmp_path / "project"
    outputs = project / "outputs"
    build = project / "report" / "build"
    _touch(project / "README.md")
    _touch(project / "configs" / "local.toml")
    for run in ("runA", "runB"):
        _touch(outputs / run / "loss.csv", "epoch,train_loss\n1,0.5\n")
        _touch(outputs / run / "samples" / f"{run}_epoch01_sample1.png")
        _touch(outputs / run / "trajectories" / f"{run}_t0000_sample1.png")
        _touch(outputs / run / "trajectories" / "trajectory.json", "{}")
        _touch(outputs / run / "checkpoints" / "final_gen.pth")
        _touch(outputs / run / "test_outputs" / f"{run}_test_00001.png")
    _touch(outputs / "runB" / "diagnostics" / "runB_forward_noised_t2000_sample1.png")
    _touch(outputs / "runB" / "diagnostics" / "start_diagnostic.json", "{}")
    _touch(outputs / "dataset" / "manifest.csv")
    _touch(outputs / "report_assets" / "loss_curves.png")
    _touch(build / "report.pdf")
    return {"project": project, "outputs": outputs, "build": build}


def _packager(tree, requirements) -> SubmissionPackager:
    project, outputs = tree["project"], tree["outputs"]
    selectors = [
        CodeSelector(project, lister=lambda root: ["README.md", "configs/local.toml"]),
        RunArtifactSelector(outputs, "runA"),
        RunArtifactSelector(outputs, "runB"),
        DatasetManifestSelector(outputs),
        ReportSelector(project, tree["build"], outputs),
    ]
    return SubmissionPackager(selectors, requirements, ROOT_FOLDER)


def test_zip_layout_and_exclusions(tree, tmp_path):
    requirements = [
        Requirement("loss", "outputs/run*/loss.csv", 2),
        Requirement("report", "report/build/report.pdf"),
    ]
    result = _packager(tree, requirements).build(tmp_path / "out" / "sub.zip")
    assert result.complete
    with zipfile.ZipFile(result.path) as archive:
        names = set(archive.namelist())
    assert all(name.startswith(f"{ROOT_FOLDER}/") for name in names)
    relative = {name.removeprefix(f"{ROOT_FOLDER}/") for name in names}
    assert relative == {
        "MANIFEST.txt",
        "README.md",
        "outputs/dataset/manifest.csv",
        "outputs/report_assets/loss_curves.png",
        "report/build/report.pdf",
        "outputs/runB/diagnostics/runB_forward_noised_t2000_sample1.png",
        "outputs/runB/diagnostics/start_diagnostic.json",
        *(
            f"outputs/{run}/{item}"
            for run in ("runA", "runB")
            for item in (
                "loss.csv",
                f"samples/{run}_epoch01_sample1.png",
                f"trajectories/{run}_t0000_sample1.png",
                "trajectories/trajectory.json",
            )
        ),
    }


def test_manifest_lists_every_file(tree, tmp_path):
    result = _packager(tree, []).build(tmp_path / "sub.zip")
    with zipfile.ZipFile(result.path) as archive:
        manifest = archive.read(f"{ROOT_FOLDER}/MANIFEST.txt").decode("utf-8")
    assert all(name in manifest for name in result.names)


def test_missing_items_stop_the_build_before_writing(tree, tmp_path):
    zip_path = tmp_path / "sub.zip"
    packager = _packager(tree, [Requirement("Run 0 metrics", "outputs/run0/test_metrics.json")])
    with pytest.raises(IncompleteSubmissionError, match="Run 0 metrics"):
        packager.build(zip_path)
    assert not zip_path.exists()
    result = packager.build(zip_path, allow_incomplete=True)
    assert not result.complete and zip_path.is_file()
    assert result.to_dict()["missing"] == ["Run 0 metrics (outputs/run0/test_metrics.json, need 1)"]


def test_two_sources_for_one_name_are_rejected(tmp_path):
    first, second = _touch(tmp_path / "a.txt"), _touch(tmp_path / "b.txt")

    class Fixed:
        def __init__(self, source: Path) -> None:
            self._source = source

        def select(self):
            yield Artifact(self._source, "same.txt")

    packager = SubmissionPackager([Fixed(first), Fixed(second)], [], ROOT_FOLDER)
    with pytest.raises(ValueError, match=r"same\.txt"):
        packager.build(tmp_path / "sub.zip")


def test_brief_requirements_follow_the_configuration():
    requirements = {
        r.pattern: r.minimum for r in brief_requirements(load_config(local_config=None))
    }
    assert requirements["outputs/runB/samples/*.png"] == 15
    assert requirements["outputs/runA/trajectories/*.png"] == 15
    assert requirements["report/build/report.pdf"] == 1


def test_code_selector_outside_a_git_checkout_fails_clearly(tmp_path):
    with pytest.raises(RuntimeError, match="not a git checkout"):
        list(CodeSelector(tmp_path).select())


@pytest.mark.skipif(
    not (PROJECT_ROOT / ".git").exists(), reason="needs a git checkout, not an unpacked zip"
)
def test_code_selector_uses_committed_files_only():
    names = {artifact.archive_name for artifact in CodeSelector(PROJECT_ROOT).select()}
    assert {"scripts/finetune.py", "README.md", "pyproject.toml"} <= names
    assert "configs/local.toml" not in names
    assert not any(name.startswith(("outputs/", "data/")) for name in names)
    assert set(git_tracked_files(PROJECT_ROOT)) >= names
