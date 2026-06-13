# Performance Report — Deepfake Audio Detection

## Metrics

| Metric | Value |
|--------|-------|
| Accuracy | 90.30% |
| EER | 2.28% |
| F1 Score | 0.893 |
| Genuine Accuracy | 99.93% |
| Deepfake Accuracy | 80.67% |
| Threshold | 0.50 |

## Confusion Matrix

|  | Predicted: Genuine | Predicted: Deepfake |
|--|-------------------|---------------------|
| **Actual: Genuine** | 5384 (TN) | 4 (FP) |
| **Actual: Deepfake** | 1041 (FN) | 4344 (TP) |


Near-zero false positive rate (0.07%) on genuine audio. Deepfake recall is the main weakness — 1,041 missed clips (FNR 19.3%) at threshold 0.5. The EER of 2.28% shows strong underlying discrimination; lowering the threshold toward ~0.30 would recover most of those misses.

---

## Preprocessing & Feature Extraction

- **Resample** to 16 kHz mono, pad/trim to 4 s (64,000 samples)
- **Log-mel spectrogram**: 64 mel bands, FFT 1024, hop 256, 20–7600 Hz
- **Normalize**: per-clip z-score `(dB − μ) / (σ + 1e-6)`
- Output shape: `1 × 64 × T` (single-channel image)

---

## Model Architecture — AudioCNN

| Block | Layers | Out |
|-------|--------|-----|
| Conv Block 1 | Conv2d(1→16, 3×3) + BN + ReLU + MaxPool(2) | 16 ch |
| Conv Block 2 | Conv2d(16→32, 3×3) + BN + ReLU + MaxPool(2) | 32 ch |
| Conv Block 3 | Conv2d(32→64, 3×3) + BN + ReLU + AdaptiveAvgPool(1×1) | 64 ch |
| Classifier | Flatten + Dropout(0.25) + Linear(64→1) | 1 logit |

~55k trainable parameters. Sigmoid at inference; values ≥ threshold → Deepfake.
