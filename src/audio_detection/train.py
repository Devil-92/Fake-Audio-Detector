from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset

from .config import ModelConfig
from .data import AudioExample, build_manifest, resolve_dataset_root
from .features import log_mel_spectrogram
from .metrics import compute_metrics, save_metrics_report
from .model import AudioCNN


class SpectrogramDataset(Dataset):
    def __init__(self, examples: list[AudioExample], config: ModelConfig) -> None:
        self.examples = examples
        self.config = config

    def __len__(self) -> int:
        return len(self.examples)

    def __getitem__(self, idx: int) -> tuple[torch.Tensor, torch.Tensor]:
        item = self.examples[idx]
        mel = log_mel_spectrogram(item.path, self.config.audio)
        x = torch.from_numpy(mel).unsqueeze(0)
        y = torch.tensor(float(item.label), dtype=torch.float32)
        return x, y


def _evaluate(model: AudioCNN, loader: DataLoader, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    scores: list[float] = []
    labels: list[float] = []
    with torch.no_grad():
        for x, y in loader:
            x = x.to(device)
            logits = model(x)
            probs = torch.sigmoid(logits).cpu().numpy()
            scores.extend(probs.tolist())
            labels.extend(y.numpy().tolist())
    return np.asarray(labels, dtype=int), np.asarray(scores, dtype=float)


def stratified_split(
    examples: list[AudioExample],
    val_size: float,
    seed: int,
) -> tuple[list[AudioExample], list[AudioExample]]:
    rng = np.random.default_rng(seed)
    train_items: list[AudioExample] = []
    val_items: list[AudioExample] = []

    for label in sorted({item.label for item in examples}):
        class_items = [item for item in examples if item.label == label]
        indices = np.arange(len(class_items))
        rng.shuffle(indices)
        n_val = max(1, int(round(len(class_items) * val_size)))
        val_indices = set(indices[:n_val].tolist())
        for idx, item in enumerate(class_items):
            if idx in val_indices:
                val_items.append(item)
            else:
                train_items.append(item)

    rng.shuffle(train_items)
    rng.shuffle(val_items)
    return train_items, val_items


def train(args: argparse.Namespace) -> dict:
    dataset_root = resolve_dataset_root(args.data_root, args.use_kagglehub)
    manifest = build_manifest(dataset_root, sample_per_class=args.sample_per_class)
    labels = [item.label for item in manifest]
    train_items, val_items = stratified_split(manifest, val_size=args.val_size, seed=args.seed)

    config = ModelConfig()
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    train_loader = DataLoader(
        SpectrogramDataset(train_items, config),
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
    )
    val_loader = DataLoader(
        SpectrogramDataset(val_items, config),
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
    )

    model = AudioCNN().to(device)
    positive = max(sum(labels), 1)
    negative = max(len(labels) - sum(labels), 1)
    pos_weight = torch.tensor([negative / positive], dtype=torch.float32, device=device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    best_eer = float("inf")
    best_state = None
    patience_left = args.patience

    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        for x, y in train_loader:
            x = x.to(device)
            y = y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.item()) * x.size(0)

        y_true, y_score = _evaluate(model, val_loader, device)
        metrics = compute_metrics(y_true, y_score, threshold=config.threshold)
        mean_loss = total_loss / max(len(train_items), 1)
        print(
            f"Epoch {epoch:03d} loss={mean_loss:.4f} "
            f"val_acc={metrics['accuracy']:.4f} val_eer={metrics['eer']:.4f} val_f1={metrics['f1']:.4f}"
        )

        if metrics["eer"] < best_eer:
            best_eer = metrics["eer"]
            best_state = model.state_dict()
            patience_left = args.patience
        else:
            patience_left -= 1
            if patience_left <= 0:
                print("Early stopping.")
                break

    if best_state is not None:
        model.load_state_dict(best_state)

    y_true, y_score = _evaluate(model, val_loader, device)
    metrics = compute_metrics(y_true, y_score, threshold=config.threshold)

    model_dir = Path(args.model_dir)
    model_dir.mkdir(parents=True, exist_ok=True)
    torch.save(model.state_dict(), model_dir / "deepfake_cnn.pth")
    config.save(model_dir / "model_config.json")
    save_metrics_report(metrics, args.reports_dir)
    print(f"Saved model to {model_dir / 'deepfake_cnn.pth'}")
    print(f"Saved metrics to {Path(args.reports_dir) / 'metrics.json'}")
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train a compact CNN deepfake audio detector.")
    parser.add_argument("--data-root", type=str, default=None, help="Path to the LA norm/train folder.")
    parser.add_argument("--use-kagglehub", action="store_true", help="Download/attach the Kaggle dataset with kagglehub.")
    parser.add_argument("--sample-per-class", type=int, default=None, help="Limit examples per class for low-disk smoke runs.")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--val-size", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--model-dir", type=str, default="models")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    train(args)


if __name__ == "__main__":
    main()
