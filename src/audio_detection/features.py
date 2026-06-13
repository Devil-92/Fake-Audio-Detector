from __future__ import annotations

from pathlib import Path

import librosa
import numpy as np

from .config import AudioConfig


SUPPORTED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a", ".aac"}


def load_audio(path: str | Path, config: AudioConfig) -> np.ndarray:
    """Load audio as mono float32 and pad/trim to the configured duration."""
    y, _ = librosa.load(path, sr=config.sample_rate, mono=True)
    y = np.asarray(y, dtype=np.float32)

    if y.size < config.n_samples:
        y = np.pad(y, (0, config.n_samples - y.size))
    else:
        y = y[: config.n_samples]

    return y


def log_mel_spectrogram(path: str | Path, config: AudioConfig) -> np.ndarray:
    y = load_audio(path, config)
    mel = librosa.feature.melspectrogram(
        y=y,
        sr=config.sample_rate,
        n_fft=config.n_fft,
        hop_length=config.hop_length,
        n_mels=config.n_mels,
        fmin=config.fmin,
        fmax=config.fmax,
        power=2.0,
    )
    db = librosa.power_to_db(mel, ref=np.max)
    db = (db - db.mean()) / (db.std() + 1e-6)
    return db.astype(np.float32)
