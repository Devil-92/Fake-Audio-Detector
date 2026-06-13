from __future__ import annotations

from pathlib import Path
import tempfile

import streamlit as st

from audio_detection.inference import AudioPredictor


st.set_page_config(page_title="Deepfake Audio Detector", layout="centered")
st.title("Deepfake Audio Detector")

model_path = Path("models/deepfake_cnn.pth")
config_path = Path("models/model_config.json")

if not model_path.exists() or not config_path.exists():
    st.error("Model artifacts are missing. Train on Kaggle first, then place deepfake_cnn.pth and model_config.json in models/.")
    st.stop()


@st.cache_resource
def load_predictor() -> AudioPredictor:
    return AudioPredictor(model_path, config_path)


uploaded = st.file_uploader("Upload an audio file", type=["wav", "mp3", "flac", "ogg", "m4a"])

if uploaded is not None:
    suffix = Path(uploaded.name).suffix or ".wav"
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
        tmp.write(uploaded.getbuffer())
        tmp_path = Path(tmp.name)

    st.audio(uploaded)
    with st.spinner("Analyzing audio..."):
        result = load_predictor().predict(tmp_path)

    st.metric("Prediction", result["label"])
    st.metric("Confidence", f"{result['confidence'] * 100:.2f}%")
    st.caption(f"Deepfake probability: {result['deepfake_probability'] * 100:.2f}%")
