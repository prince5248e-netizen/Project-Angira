
import os
import tempfile
from flask import Flask, jsonify, render_template, request

from inference_custom import load_model, predict_file

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024

MODEL = load_model(device="cpu")


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    if "audio" not in request.files:
        return jsonify({"error": "No audio file uploaded or recorded."}), 400

    audio_file = request.files["audio"]
    if audio_file.filename == "":
        return jsonify({"error": "No file selected."}), 400

    suffix = ".wav"
    original_name = audio_file.filename.lower()
    if original_name.endswith(".wav"):
        suffix = ".wav"
    elif original_name.endswith(".mp3"):
        suffix = ".mp3"
    elif original_name.endswith(".flac"):
        suffix = ".flac"
    elif original_name.endswith(".m4a"):
        suffix = ".m4a"
    elif original_name.endswith(".ogg"):
        suffix = ".ogg"
    elif original_name.endswith(".webm"):
        suffix = ".webm"

    fd, temp_path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    try:
        audio_file.save(temp_path)
        prob = predict_file(MODEL, temp_path, device="cpu")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Avoid forcing a binary decision when the model is close to its threshold.
    if 0.35 <= prob <= 0.65:
        label = "Uncertain"
    else:
        label = "Fake" if prob > 0.5 else "Real"
    confidence = prob if label == "Fake" else 1.0 - prob if label == "Real" else 0.5
    return jsonify({
        "label": label,
        "fake_probability": float(prob),
        "confidence": float(confidence),
        "threshold": 0.5,
        "model": "MelodyMachine/Deepfake-audio-detection-V2 (calibrated label mapping)",
        "message": f"Predicted {label} (fake probability: {prob:.4f})"
    })


if __name__ == "__main__":
    port = int(os.environ.get("PORT", "5000"))
    app.run(host="0.0.0.0", port=port, debug=False)
