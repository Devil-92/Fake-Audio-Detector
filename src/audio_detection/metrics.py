from __future__ import annotations

from pathlib import Path
import json

import numpy as np


def equal_error_rate(y_true: np.ndarray, y_score: np.ndarray) -> float:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    thresholds = np.r_[np.inf, np.sort(np.unique(y_score))[::-1], -np.inf]
    positives = max(int((y_true == 1).sum()), 1)
    negatives = max(int((y_true == 0).sum()), 1)

    fpr_values = []
    fnr_values = []
    for threshold in thresholds:
        y_pred = (y_score >= threshold).astype(int)
        false_positive = int(((y_pred == 1) & (y_true == 0)).sum())
        false_negative = int(((y_pred == 0) & (y_true == 1)).sum())
        fpr_values.append(false_positive / negatives)
        fnr_values.append(false_negative / positives)

    fpr = np.asarray(fpr_values)
    fnr = np.asarray(fnr_values)
    idx = int(np.nanargmin(np.abs(fnr - fpr)))
    return float((fpr[idx] + fnr[idx]) / 2.0)


def compute_metrics(y_true: np.ndarray, y_score: np.ndarray, threshold: float = 0.5) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_score = np.asarray(y_score, dtype=float)
    y_pred = (y_score >= threshold).astype(int)
    cm = np.array(
        [
            [int(((y_true == 0) & (y_pred == 0)).sum()), int(((y_true == 0) & (y_pred == 1)).sum())],
            [int(((y_true == 1) & (y_pred == 0)).sum()), int(((y_true == 1) & (y_pred == 1)).sum())],
        ]
    )

    accuracy = float((y_true == y_pred).mean()) if y_true.size else 0.0
    true_positive = cm[1, 1]
    false_positive = cm[0, 1]
    false_negative = cm[1, 0]
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 0.0 if precision + recall == 0 else 2 * precision * recall / (precision + recall)

    per_class_accuracy = {}
    for label, name in [(0, "Genuine"), (1, "Deepfake")]:
        mask = y_true == label
        per_class_accuracy[name] = float((y_true[mask] == y_pred[mask]).mean()) if mask.any() else 0.0

    return {
        "accuracy": accuracy,
        "eer": equal_error_rate(y_true, y_score),
        "f1": float(f1),
        "per_class_accuracy": per_class_accuracy,
        "confusion_matrix": cm.tolist(),
        "threshold": threshold,
    }


def save_metrics_report(metrics: dict, reports_dir: str | Path) -> None:
    import matplotlib.pyplot as plt
    import seaborn as sns

    reports = Path(reports_dir)
    reports.mkdir(parents=True, exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    cm = np.asarray(metrics["confusion_matrix"])
    plt.figure(figsize=(5, 4))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=["Genuine", "Deepfake"],
        yticklabels=["Genuine", "Deepfake"],
    )
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(reports / "confusion_matrix.png", dpi=160)
    plt.close()
