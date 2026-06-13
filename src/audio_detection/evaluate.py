from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .config import ModelConfig
from .data import build_manifest, resolve_dataset_root
from .metrics import compute_metrics, save_metrics_report
from .model import AudioCNN
from .train import SpectrogramDataset


def evaluate(args: argparse.Namespace) -> dict:
    dataset_root = resolve_dataset_root(args.data_root, args.use_kagglehub)
    manifest = build_manifest(dataset_root, sample_per_class=args.sample_per_class)

    config = ModelConfig.load(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() and not args.cpu else "cpu")
    model = AudioCNN().to(device)
    model.load_state_dict(torch.load(args.model, map_location=device))

    loader = DataLoader(SpectrogramDataset(manifest, config), batch_size=args.batch_size, shuffle=False)
    model.eval()
    scores: list[float] = []
    labels: list[int] = []
    with torch.no_grad():
        for x, y in loader:
            probs = torch.sigmoid(model(x.to(device))).cpu().numpy()
            scores.extend(probs.tolist())
            labels.extend(y.numpy().astype(int).tolist())

    metrics = compute_metrics(np.asarray(labels), np.asarray(scores), threshold=config.threshold)
    save_metrics_report(metrics, args.reports_dir)
    print(metrics)
    return metrics


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Evaluate a trained deepfake audio detector.")
    parser.add_argument("--data-root", type=str, default=None)
    parser.add_argument("--use-kagglehub", action="store_true")
    parser.add_argument("--sample-per-class", type=int, default=None)
    parser.add_argument("--model", type=str, default="models/deepfake_cnn.pth")
    parser.add_argument("--config", type=str, default="models/model_config.json")
    parser.add_argument("--reports-dir", type=str, default="reports")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--cpu", action="store_true")
    return parser


def main() -> None:
    evaluate(build_parser().parse_args())


if __name__ == "__main__":
    main()
