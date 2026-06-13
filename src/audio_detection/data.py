from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
import csv
import os
import re

DATASET_HANDLE = "mohammedabdeldayem/the-fake-or-real-dataset"
TARGET_PARTS = ("LA norm", "train")
TARGET_LAYOUTS = (
    ("LA norm", "train"),
    ("LA norm", "training"),
    ("for-norm", "training"),
    ("for-original", "training"),
)
SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}

GENUINE_TOKENS = {"bonafide", "bona-fide", "genuine", "human", "real", "authentic", "0"}
DEEPFAKE_TOKENS = {"spoof", "fake", "deepfake", "ai", "generated", "synthetic", "1"}


@dataclass(frozen=True)
class AudioExample:
    path: Path
    label: int


def download_with_kagglehub() -> Path:
    import kagglehub

    path = kagglehub.dataset_download(DATASET_HANDLE)
    print("Path to dataset files:", path)
    return Path(path)


def resolve_dataset_root(data_root: str | Path | None = None, use_kagglehub: bool = False) -> Path:
    if data_root:
        root = Path(data_root).expanduser().resolve()
    elif use_kagglehub:
        root = download_with_kagglehub()
    else:
        env_root = os.getenv("DATASET_ROOT")
        if env_root:
            root = Path(env_root).expanduser().resolve()
        else:
            root = download_with_kagglehub()

    if any(root.name == child and root.parent.name == parent for parent, child in TARGET_LAYOUTS):
        return root

    matches = []
    for parent, child in TARGET_LAYOUTS:
        matches.extend(p for p in root.rglob(child) if p.is_dir() and p.parent.name == parent)

    if not matches:
        raise FileNotFoundError(
            f"Could not locate a supported training folder under {root}. Looked for: "
            + ", ".join(f"{parent}/{child}" for parent, child in TARGET_LAYOUTS)
            + ". Pass --data-root directly to the training folder if the dataset layout differs."
        )
    return matches[0].resolve()


def discover_audio_files(root: Path) -> list[Path]:
    files = sorted(p for p in root.rglob("*") if p.suffix.lower() in SUPPORTED_AUDIO_EXTENSIONS)
    if not files:
        raise FileNotFoundError(f"No supported audio files found under {root}.")
    return files


def label_from_token(value: str) -> int | None:
    token = re.sub(r"[^a-zA-Z0-9-]+", "", value).lower()
    if token in GENUINE_TOKENS:
        return 0
    if token in DEEPFAKE_TOKENS:
        return 1
    return None


def _metadata_files(root: Path) -> list[Path]:
    candidates: list[Path] = []
    for parent in [root, *root.parents[:3]]:
        if not parent.exists():
            continue
        candidates.extend(
            p
            for p in parent.iterdir()
            if p.is_file() and p.suffix.lower() in {".csv", ".tsv", ".txt", ".protocol"}
        )
    return sorted(set(candidates))


def _load_metadata_labels(root: Path, audio_files: list[Path]) -> dict[str, int]:
    stems = {p.stem: p for p in audio_files}
    names = {p.name: p for p in audio_files}
    labels: dict[str, int] = {}

    for meta in _metadata_files(root):
        try:
            rows = _read_metadata_rows(meta)
        except UnicodeDecodeError:
            continue
        for row in rows:
            row_labels = [label_from_token(cell) for cell in row]
            row_labels = [label for label in row_labels if label is not None]
            if not row_labels:
                continue

            matched: Path | None = None
            for cell in row:
                value = Path(cell).name
                if value in names:
                    matched = names[value]
                    break
                stem = Path(value).stem
                if stem in stems:
                    matched = stems[stem]
                    break

            if matched is not None:
                labels[str(matched.resolve())] = row_labels[-1]

    return labels


def _read_metadata_rows(path: Path) -> list[list[str]]:
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() == ".csv":
        return [row for row in csv.reader(text.splitlines())]
    if path.suffix.lower() == ".tsv":
        return [row for row in csv.reader(text.splitlines(), delimiter="\t")]
    return [line.split() for line in text.splitlines() if line.strip() and not line.startswith("#")]


def _folder_label(path: Path, root: Path) -> int | None:
    try:
        parts = path.relative_to(root).parts[:-1]
    except ValueError:
        parts = path.parts[:-1]
    for part in reversed(parts):
        label = label_from_token(part)
        if label is not None:
            return label
    return None


def _filename_label(path: Path) -> int | None:
    for token in re.split(r"[_\-. ]+", path.stem):
        label = label_from_token(token)
        if label is not None:
            return label
    return None


def build_manifest(root: str | Path, sample_per_class: int | None = None) -> list[AudioExample]:
    root = Path(root).expanduser().resolve()
    audio_files = discover_audio_files(root)
    metadata_labels = _load_metadata_labels(root, audio_files)

    examples: list[AudioExample] = []
    unlabeled: list[Path] = []

    for path in audio_files:
        label = metadata_labels.get(str(path.resolve()))
        if label is None:
            label = _folder_label(path, root)
        if label is None:
            label = _filename_label(path)
        if label is None:
            unlabeled.append(path)
        else:
            examples.append(AudioExample(path=path, label=label))

    if unlabeled:
        preview = ", ".join(str(p) for p in unlabeled[:5])
        raise ValueError(
            f"Could not infer labels for {len(unlabeled)} audio files. Examples: {preview}. "
            "Add a protocol/CSV metadata file or organize files into class-named folders."
        )

    counts = defaultdict(int)
    for item in examples:
        counts[item.label] += 1
    if counts[0] == 0 or counts[1] == 0:
        raise ValueError(f"Need both classes after label discovery; found counts={dict(counts)}.")

    if sample_per_class:
        limited: list[AudioExample] = []
        seen = defaultdict(int)
        for item in examples:
            if seen[item.label] < sample_per_class:
                limited.append(item)
                seen[item.label] += 1
        examples = limited

    return examples
