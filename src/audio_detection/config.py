from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import json


@dataclass(frozen=True)
class AudioConfig:
    sample_rate: int = 16_000
    duration: float = 4.0
    n_mels: int = 64
    n_fft: int = 1024
    hop_length: int = 256
    fmin: int = 20
    fmax: int | None = 7_600

    @property
    def n_samples(self) -> int:
        return int(self.sample_rate * self.duration)


@dataclass(frozen=True)
class ModelConfig:
    audio: AudioConfig = AudioConfig()
    class_names: tuple[str, str] = ("Genuine", "Deepfake")
    threshold: float = 0.5

    def save(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        payload = asdict(self)
        payload["class_names"] = list(self.class_names)
        target.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> "ModelConfig":
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        audio = AudioConfig(**payload.get("audio", {}))
        class_names = tuple(payload.get("class_names", ("Genuine", "Deepfake")))
        return cls(
            audio=audio,
            class_names=(class_names[0], class_names[1]),
            threshold=float(payload.get("threshold", 0.5)),
        )
