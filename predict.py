from __future__ import annotations

import argparse
import json

from audio_detection.inference import AudioPredictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Predict whether an audio file is genuine or deepfake.")
    parser.add_argument("audio_path", help="Path to a new audio file.")
    parser.add_argument("--model", default="models/deepfake_cnn.pth")
    parser.add_argument("--config", default="models/model_config.json")
    parser.add_argument("--cpu", action="store_true")
    args = parser.parse_args()

    predictor = AudioPredictor(args.model, args.config, cpu=args.cpu)
    result = predictor.predict(args.audio_path)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
