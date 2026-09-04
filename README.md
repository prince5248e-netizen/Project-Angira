# Deepfake Voice Detector — local test harness

Quick scripts to run the `koyelog/deepfake-voice-detector-sota` model locally for testing.

Files added:
- `inference_custom.py` — native `MelodyMachine/Deepfake-audio-detection-V2` loader, `load_model()`, `predict_file()` and CLI (`--predict`, `--batch`).
- `eval_inference.py` — small batch evaluation harness that creates `test_data/real` and `test_data/fake` if missing and computes metrics.
- `test_pipeline.py` — minimal HF pipeline test.
- `requirements.txt` — runtime requirements.

Notes:
- The app uses the native Hugging Face Wav2Vec2 classifier `MelodyMachine/Deepfake-audio-detection-V2`. Its label mapping is `fake=0`, `real=1`.
- The inference boundary applies the calibrated inversion enabled by `INVERT_MODEL_LABELS` because the published mapping was reversed on the user's labeled real/fake samples.
- The first run downloads approximately 378 MB of model weights into the Hugging Face cache.
- The repo's model is optimized for 4s clips at 16kHz; ensure audio is preprocessed accordingly. The scripts produce demo 4s files if none exist.
- Results near the 0.5 threshold are reported as `Uncertain` instead of forcing a binary decision.
- For meaningful evaluation replace `test_data` with real labeled datasets (ASVspoof, WaveFake, etc.).
