import os
from inference_custom import load_model, predict_file


def test_load_model_returns_model():
    model = load_model(device="cpu")
    assert model is not None


def test_predict_file_returns_probability(tmp_path):
    import numpy as np
    import soundfile as sf

    sr = 16000
    t = np.linspace(0, 4, 4 * sr, endpoint=False)
    sine = 0.01 * np.sin(2 * np.pi * 220 * t)
    p = tmp_path / "test.wav"
    sf.write(str(p), sine, sr)

    model = load_model(device="cpu")
    prob = predict_file(model, str(p), device="cpu")
    assert isinstance(prob, float)
    assert 0.0 <= prob <= 1.0
