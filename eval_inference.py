import os
from pathlib import Path
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score, confusion_matrix
import csv
from inference_custom import load_model, predict_file


def build_demo_dataset(base_dir: Path):
    # creates a tiny demo dataset if none exists
    real_dir = base_dir / "real"
    fake_dir = base_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)
    # if empty, generate simple tones/noise
    if not any(real_dir.iterdir()):
        import numpy as np
        import soundfile as sf
        sr = 16000
        t = np.linspace(0, 4, 4 * sr, endpoint=False)
        sine = 0.05 * np.sin(2 * np.pi * 220 * t)
        sf.write(str(real_dir / "real1.wav"), sine, sr)
    if not any(fake_dir.iterdir()):
        import numpy as np
        import soundfile as sf
        sr = 16000
        noise = 0.01 * np.random.randn(4 * sr)
        sf.write(str(fake_dir / "fake1.wav"), noise, sr)
    return real_dir, fake_dir


def collect_files(real_dir: Path, fake_dir: Path):
    files = []
    for p in sorted(real_dir.glob("*.wav")):
        files.append((str(p), 0))
    for p in sorted(fake_dir.glob("*.wav")):
        files.append((str(p), 1))
    return files


def evaluate(model, files):
    y_true = []
    y_prob = []
    for path, label in files:
        prob = predict_file(model, path)
        y_true.append(label)
        y_prob.append(prob)
    y_pred = [1 if p >= 0.5 else 0 for p in y_prob]
    acc = accuracy_score(y_true, y_pred)
    prec, rec, f1, _ = precision_recall_fscore_support(y_true, y_pred, average="binary")
    auc = None
    try:
        auc = roc_auc_score(y_true, y_prob)
    except Exception:
        auc = float("nan")
    cm = confusion_matrix(y_true, y_pred)
    return {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1, "auc": auc, "confusion_matrix": cm.tolist()}


def main():
    base = Path("test_data")
    real_dir, fake_dir = build_demo_dataset(base)
    files = collect_files(real_dir, fake_dir)
    print("Files for eval:")
    for p, l in files:
        print(p, l)
    model = load_model(device="cpu")
    results = evaluate(model, files)
    print("Evaluation results:")
    for k, v in results.items():
        print(f"{k}: {v}")


if __name__ == '__main__':
    main()
