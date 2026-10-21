
from scripts.evaluate_crowdnav_generalization import _checkpoint_path


def test_generalization_checkpoint_path_accepts_run_and_project_relative_paths(
    tmp_path, monkeypatch
):
    run_dir = tmp_path / "results" / "run"
    project_relative = run_dir / "stage3_train" / "model.zip"
    project_relative.parent.mkdir(parents=True)
    project_relative.write_bytes(b"checkpoint")
    monkeypatch.chdir(tmp_path)

    assert _checkpoint_path(
        "results/run/stage3_train/model.zip", run_dir
    ) == project_relative

    run_relative = run_dir / "models" / "candidate.zip"
    run_relative.parent.mkdir(parents=True)
    run_relative.write_bytes(b"checkpoint")
    assert _checkpoint_path("models/candidate.zip", run_dir) == run_relative
