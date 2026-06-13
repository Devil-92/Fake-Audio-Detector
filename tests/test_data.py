from pathlib import Path

from audio_detection.data import build_manifest, resolve_dataset_root


def test_resolve_dataset_root_finds_la_norm_train(tmp_path):
    target = tmp_path / "the-fake-or-real-dataset" / "LA norm" / "train"
    target.mkdir(parents=True)

    assert resolve_dataset_root(tmp_path) == target.resolve()


def test_build_manifest_uses_folder_labels(tmp_path):
    root = tmp_path / "LA norm" / "train"
    real_dir = root / "real"
    fake_dir = root / "fake"
    real_dir.mkdir(parents=True)
    fake_dir.mkdir(parents=True)
    (real_dir / "a.wav").write_bytes(b"placeholder")
    (fake_dir / "b.wav").write_bytes(b"placeholder")

    manifest = build_manifest(root)

    assert sorted(item.label for item in manifest) == [0, 1]
    assert {Path(item.path).name for item in manifest} == {"a.wav", "b.wav"}
