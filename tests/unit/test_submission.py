# ADDED: new file, not part of the upstream SR3 codebase.
"""Submission zip: exclusions, selectors, completeness checks and the archive layout."""  # ADDED

from __future__ import annotations  # ADDED

import zipfile  # ADDED
from pathlib import Path  # ADDED

import pytest  # ADDED

from sr3_ablation.config import load_config  # ADDED
from sr3_ablation.paths import PROJECT_ROOT  # ADDED
from sr3_ablation.submission import (  # ADDED
    DEFAULT_EXCLUSIONS,  # ADDED
    Artifact,  # ADDED
    IncompleteSubmissionError,  # ADDED
    Requirement,  # ADDED
    SubmissionPackager,  # ADDED
    brief_requirements,  # ADDED
)  # ADDED
from sr3_ablation.submission.selectors import (  # ADDED
    CodeSelector,  # ADDED
    DatasetManifestSelector,  # ADDED
    ReportSelector,  # ADDED
    RunArtifactSelector,  # ADDED
    git_tracked_files,  # ADDED
)  # ADDED

ROOT_FOLDER = "Group_X"  # ADDED


@pytest.mark.parametrize(  # ADDED
    ("name", "excluded"),  # ADDED
    [  # ADDED
        ("outputs/runA/checkpoints/final_gen.pth", True),  # ADDED
        ("outputs/runA/test_outputs/frame.jpg", True),  # ADDED
        ("data/raw/kvasir-seg/Kvasir-SEG/images/a.png", True),  # ADDED
        ("src/sr3_ablation/data/loaders.py", False),  # ADDED
        ("src/sr3_ablation/__pycache__/cli.cpython-314.pyc", True),  # ADDED
        ("configs/local.toml", True),  # ADDED
        ("configs/base.toml", False),  # ADDED
        ("outputs/runB/samples/runB_epoch12_sample1.png", False),  # ADDED
    ],  # ADDED
)  # ADDED
def test_default_exclusions(name, excluded):  # ADDED
    assert DEFAULT_EXCLUSIONS.excludes(name) is excluded  # ADDED


def _touch(path: Path, text: str = "x") -> Path:  # ADDED
    path.parent.mkdir(parents=True, exist_ok=True)  # ADDED
    path.write_text(text, encoding="utf-8")  # ADDED
    return path  # ADDED


@pytest.fixture  # ADDED
def tree(tmp_path) -> dict[str, Path]:  # ADDED
    project = tmp_path / "project"  # ADDED
    outputs = project / "outputs"  # ADDED
    build = project / "report" / "build"  # ADDED
    _touch(project / "README.md")  # ADDED
    _touch(project / "configs" / "local.toml")  # ADDED
    for run in ("runA", "runB"):  # ADDED
        _touch(outputs / run / "loss.csv", "epoch,train_loss\n1,0.5\n")  # ADDED
        _touch(outputs / run / "samples" / f"{run}_epoch01_sample1.png")  # ADDED
        _touch(outputs / run / "trajectories" / f"{run}_t0000_sample1.png")  # ADDED
        _touch(outputs / run / "trajectories" / "trajectory.json", "{}")  # ADDED
        _touch(outputs / run / "checkpoints" / "final_gen.pth")  # ADDED
        _touch(outputs / run / "test_outputs" / f"{run}_test_00001.png")  # ADDED
    _touch(outputs / "runB" / "diagnostics" / "runB_forward_noised_t2000_sample1.png")  # ADDED
    _touch(outputs / "runB" / "diagnostics" / "start_diagnostic.json", "{}")  # ADDED
    _touch(outputs / "dataset" / "manifest.csv")  # ADDED
    _touch(outputs / "report_assets" / "loss_curves.png")  # ADDED
    _touch(build / "report.pdf")  # ADDED
    return {"project": project, "outputs": outputs, "build": build}  # ADDED


def _packager(tree, requirements) -> SubmissionPackager:  # ADDED
    project, outputs = tree["project"], tree["outputs"]  # ADDED
    selectors = [  # ADDED
        CodeSelector(project, lister=lambda root: ["README.md", "configs/local.toml"]),  # ADDED
        RunArtifactSelector(outputs, "runA"),  # ADDED
        RunArtifactSelector(outputs, "runB"),  # ADDED
        DatasetManifestSelector(outputs),  # ADDED
        ReportSelector(project, tree["build"], outputs),  # ADDED
    ]  # ADDED
    return SubmissionPackager(selectors, requirements, ROOT_FOLDER)  # ADDED


def test_zip_layout_and_exclusions(tree, tmp_path):  # ADDED
    requirements = [  # ADDED
        Requirement("loss", "outputs/run*/loss.csv", 2),  # ADDED
        Requirement("report", "report/build/report.pdf"),  # ADDED
    ]  # ADDED
    result = _packager(tree, requirements).build(tmp_path / "out" / "sub.zip")  # ADDED
    assert result.complete  # ADDED
    with zipfile.ZipFile(result.path) as archive:  # ADDED
        names = set(archive.namelist())  # ADDED
    assert all(name.startswith(f"{ROOT_FOLDER}/") for name in names)  # ADDED
    relative = {name.removeprefix(f"{ROOT_FOLDER}/") for name in names}  # ADDED
    assert relative == {  # ADDED
        "MANIFEST.txt",  # ADDED
        "README.md",  # ADDED
        "outputs/dataset/manifest.csv",  # ADDED
        "outputs/report_assets/loss_curves.png",  # ADDED
        "report/build/report.pdf",  # ADDED
        "outputs/runB/diagnostics/runB_forward_noised_t2000_sample1.png",  # ADDED
        "outputs/runB/diagnostics/start_diagnostic.json",  # ADDED
        *(  # ADDED
            f"outputs/{run}/{item}"  # ADDED
            for run in ("runA", "runB")  # ADDED
            for item in (  # ADDED
                "loss.csv",  # ADDED
                f"samples/{run}_epoch01_sample1.png",  # ADDED
                f"trajectories/{run}_t0000_sample1.png",  # ADDED
                "trajectories/trajectory.json",  # ADDED
            )  # ADDED
        ),  # ADDED
    }  # ADDED


def test_manifest_lists_every_file(tree, tmp_path):  # ADDED
    result = _packager(tree, []).build(tmp_path / "sub.zip")  # ADDED
    with zipfile.ZipFile(result.path) as archive:  # ADDED
        manifest = archive.read(f"{ROOT_FOLDER}/MANIFEST.txt").decode("utf-8")  # ADDED
    assert all(name in manifest for name in result.names)  # ADDED


def test_missing_items_stop_the_build_before_writing(tree, tmp_path):  # ADDED
    zip_path = tmp_path / "sub.zip"  # ADDED
    packager = _packager(tree, [Requirement("Run 0 metrics", "outputs/run0/test_metrics.json")])  # ADDED
    with pytest.raises(IncompleteSubmissionError, match="Run 0 metrics"):  # ADDED
        packager.build(zip_path)  # ADDED
    assert not zip_path.exists()  # ADDED
    result = packager.build(zip_path, allow_incomplete=True)  # ADDED
    assert not result.complete and zip_path.is_file()  # ADDED
    assert result.to_dict()["missing"] == ["Run 0 metrics (outputs/run0/test_metrics.json, need 1)"]  # ADDED


def test_two_sources_for_one_name_are_rejected(tmp_path):  # ADDED
    first, second = _touch(tmp_path / "a.txt"), _touch(tmp_path / "b.txt")  # ADDED

    class Fixed:  # ADDED
        def __init__(self, source: Path) -> None:  # ADDED
            self._source = source  # ADDED

        def select(self):  # ADDED
            yield Artifact(self._source, "same.txt")  # ADDED

    packager = SubmissionPackager([Fixed(first), Fixed(second)], [], ROOT_FOLDER)  # ADDED
    with pytest.raises(ValueError, match=r"same\.txt"):  # ADDED
        packager.build(tmp_path / "sub.zip")  # ADDED


def test_brief_requirements_follow_the_configuration():  # ADDED
    requirements = {  # ADDED
        r.pattern: r.minimum  # ADDED
        for r in brief_requirements(load_config(local_config=None))  # ADDED
    }  # ADDED
    assert requirements["outputs/runB/samples/*.png"] == 15  # ADDED
    assert requirements["outputs/runA/trajectories/*.png"] == 15  # ADDED
    assert requirements["report/build/report.pdf"] == 1  # ADDED


def test_code_selector_outside_a_git_checkout_fails_clearly(tmp_path):  # ADDED
    with pytest.raises(RuntimeError, match="not a git checkout"):  # ADDED
        list(CodeSelector(tmp_path).select())  # ADDED


@pytest.mark.skipif(  # ADDED
    not (PROJECT_ROOT / ".git").exists(),  # ADDED
    reason="needs a git checkout, not an unpacked zip",  # ADDED
)  # ADDED
def test_code_selector_uses_committed_files_only():  # ADDED
    names = {artifact.archive_name for artifact in CodeSelector(PROJECT_ROOT).select()}  # ADDED
    assert {"scripts/finetune.py", "README.md", "pyproject.toml"} <= names  # ADDED
    assert "configs/local.toml" not in names  # ADDED
    assert not any(name.startswith(("outputs/", "data/")) for name in names)  # ADDED
    assert set(git_tracked_files(PROJECT_ROOT)) >= names  # ADDED
