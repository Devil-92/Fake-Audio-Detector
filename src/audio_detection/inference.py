from __future__ import annotations

from pathlib import Path

import torch

from .config import ModelConfig
from .features import log_mel_spectrogram
from .model import AudioCNN


class AudioPredictor:
    def __init__(
        self,
        model_path: str | Path = "models/deepfake_cnn.pth",
        config_path: str | Path = "models/model_config.json",
        cpu: bool = False,
    ) -> None:
        self.config = ModelConfig.load(config_path)
        self.device = torch.device("cuda" if torch.cuda.is_available() and not cpu else "cpu")
        self.model = AudioCNN().to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()

    def predict(self, audio_path: str | Path) -> dict:
        mel = log_mel_spectrogram(audio_path, self.config.audio)
        x = torch.from_numpy(mel).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            probability = float(torch.sigmoid(self.model(x)).cpu().item())

        label_idx = int(probability >= self.config.threshold)
        confidence = probability if label_idx == 1 else 1.0 - probability
        return {
            "label": self.config.class_names[label_idx],
            "confidence": confidence,
            "deepfake_probability": probability,
            "threshold": self.config.threshold,
        }
