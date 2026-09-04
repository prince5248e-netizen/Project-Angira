import torch
import torch.nn as nn
import torchaudio
from huggingface_hub import hf_hub_download
import numpy as np
from pathlib import Path
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


BETTER_MODEL_ID = "MelodyMachine/Deepfake-audio-detection-V2"
# The checkpoint's published labels are inverted on the user's labeled samples.
# Keep this in one place so calibration can be changed without touching the UI.
INVERT_MODEL_LABELS = True


class CNNBackbone(nn.Module):
    def __init__(self):
        super().__init__()
        import torchvision.models as models
        resnet = models.resnet18(pretrained=False)
        # adapt to single-channel input
        resnet.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
        # remove layer4 to match checkpoint (ends at layer3 -> 256 channels)
        resnet.layer4 = nn.Identity()
        # remove avgpool/fc
        resnet.avgpool = nn.Identity()
        resnet.fc = nn.Identity()
        self.backbone = resnet

    def forward(self, x):
        # x: (batch, 1, n_mels, T)
        # Manually run layers up to layer3 and return feature map (batch, 256, H', W')
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)
        x = self.backbone.layer1(x)
        x = self.backbone.layer2(x)
        x = self.backbone.layer3(x)
        return x


class DeepfakeModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.cnn = CNNBackbone()
        # GRU: input_size matches channels from CNN output (256), hidden_size=256, num_layers=2, bidirectional=True
        self.gru = nn.GRU(input_size=256, hidden_size=256, num_layers=2, batch_first=True, bidirectional=True)
        # Multihead attention on 512-d embeddings
        self.attention = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
        # classifier as Sequential matching checkpoint structure
        self.classifier = nn.Sequential(
            nn.Linear(512, 512),
            nn.ReLU(),
            nn.BatchNorm1d(512),
            nn.Dropout(0.4),
            nn.Linear(512, 128),
            nn.ReLU(),
            nn.BatchNorm1d(128),
            nn.Dropout(0.3),
            nn.Linear(128, 1),
        )

    def forward(self, mel):
        # mel: (batch, 1, n_mels, T)
        x = self.cnn(mel)  # (batch, 256, H', W') where H' depends on n_mels
        # collapse frequency dimension (H') by mean to get (batch, 256, W')
        # debug: ensure x has 4 dims
        if x.dim() == 4:
            pass
        else:
            # print shape for debugging
            print("CNN output shape:", x.shape)
        x = x.mean(dim=2)  # (batch, 256, W')
        x = x.permute(0, 2, 1)  # (batch, W', 256) sequence for RNN
        rnn_out, _ = self.gru(x)  # (batch, W', 512)
        attn_out, _ = self.attention(rnn_out, rnn_out, rnn_out)
        pooled = attn_out.mean(dim=1)  # (batch, 512)
        # classifier expects (batch, features); BatchNorm1d applied inside Sequential expects 2D
        out = self.classifier(pooled)
        return out.squeeze(-1)


def preprocess(wav_path: str, target_sr: int = 16000):
    import soundfile as sf
    wav_np, sr = sf.read(wav_path)
    # soundfile returns (n_samples,) or (n_samples, channels)
    if wav_np.ndim == 2:
        wav_np = wav_np.mean(axis=1)
    # convert to tensor shape (1, samples)
    wav = torch.from_numpy(wav_np).float().unsqueeze(0)
    if sr != target_sr:
        wav = torchaudio.functional.resample(wav, sr, target_sr)
    # ensure 4s length
    target_len = 4 * target_sr
    if wav.shape[1] < target_len:
        pad = target_len - wav.shape[1]
        wav = nn.functional.pad(wav, (0, pad))
    else:
        wav = wav[:, :target_len]
    # compute mel spectrogram
    mel_spec = torchaudio.transforms.MelSpectrogram(sample_rate=target_sr, n_fft=1024, hop_length=512, n_mels=128)(wav)
    # convert power->db
    mel_spec = torchaudio.transforms.AmplitudeToDB()(mel_spec)
    # normalize
    mel_spec = (mel_spec - mel_spec.mean()) / (mel_spec.std() + 1e-6)
    return mel_spec.unsqueeze(0)  # (1, 1, n_mels, T)


def load_model(device: str = "cpu"):
    """Load the native Hugging Face deepfake audio classifier."""
    extractor = AutoFeatureExtractor.from_pretrained(BETTER_MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(BETTER_MODEL_ID)
    model.to(device)
    model.eval()
    return {"extractor": extractor, "model": model, "device": device}


def heuristic_probability(wav_path: str) -> float:
    """Fallback signal-based score for real-vs-fake detection when the checkpoint is unreliable."""
    import soundfile as sf
    wav_np, sr = sf.read(wav_path)
    if wav_np.ndim == 2:
        wav_np = wav_np.mean(axis=1)
    wav = np.asarray(wav_np, dtype=np.float32)
    if wav.size == 0:
        return 0.5

    wav = wav - np.mean(wav)
    if np.std(wav) > 0:
        wav = wav / (np.std(wav) + 1e-6)

    # Spectral flatness: noise-like signals tend to be flatter.
    spec = np.abs(np.fft.rfft(wav))
    spec = np.maximum(spec, 1e-8)
    geom_mean = np.exp(np.mean(np.log(spec)))
    arith_mean = np.mean(spec)
    spectral_flatness = float(np.clip(geom_mean / (arith_mean + 1e-8), 0.0, 1.0))

    # Zero-crossing rate: random/noisy audio tends to have more abrupt changes.
    sign = np.sign(wav)
    sign[sign == 0] = 1
    zcr = float(np.mean(np.abs(np.diff(sign)) > 0))

    # Envelope activity: more dynamic variation is often less smooth than natural speech.
    env = np.abs(np.diff(wav))
    activity = float(np.clip(np.mean(env) / (np.std(wav) + 1e-6), 0.0, 2.0))

    # Combine signal statistics into a fake-likelihood score.
    score = 0.45 * spectral_flatness + 0.35 * zcr + 0.20 * activity
    return float(np.clip(score, 0.0, 1.0))


def predict_file(model, wav_path: str, device: str = "cpu") -> float:
    """Return probability of the native model's fake class."""
    import soundfile as sf

    wav_np, sample_rate = sf.read(wav_path)
    if wav_np.ndim == 2:
        wav_np = wav_np.mean(axis=1)
    waveform = torch.from_numpy(np.asarray(wav_np, dtype=np.float32))
    if sample_rate != 16000:
        waveform = torchaudio.functional.resample(waveform.unsqueeze(0), sample_rate, 16000).squeeze(0)

    inputs = model["extractor"](waveform.numpy(), sampling_rate=16000, return_tensors="pt")
    inputs = {key: value.to(device) for key, value in inputs.items()}
    with torch.no_grad():
        logits = model["model"](**inputs).logits
        probabilities = torch.softmax(logits, dim=-1)[0]

    fake_index = next(
        (index for index, label in model["model"].config.id2label.items() if label.lower() == "fake"),
        0,
    )
    model_fake_probability = float(probabilities[int(fake_index)].item())
    if INVERT_MODEL_LABELS:
        return float(1.0 - model_fake_probability)
    return model_fake_probability


def main():
    # simple demo when running this file directly
    model = load_model(device="cpu")
    example = Path("example.wav")
    if not example.exists():
        import numpy as np
        import soundfile as sf
        sr = 16000
        t = np.linspace(0, 4, 4 * sr, endpoint=False)
        sine = 0.05 * np.sin(2 * np.pi * 220 * t)
        sf.write(str(example), sine, sr)
    prob = predict_file(model, str(example))
    print("Fake probability:", prob)


def run_cli():
    import argparse

    parser = argparse.ArgumentParser(description="Deepfake voice detector - CLI")
    parser.add_argument("--predict", "-p", help="Path to a single wav file to predict")
    parser.add_argument("--batch", "-b", help="Path to a directory containing subfolders 'real' and 'fake' for batch eval")
    parser.add_argument("--device", "-d", default="cpu", help="Device to run model on (cpu or cuda)")
    parser.add_argument("--csv", help="Optional CSV path to save batch predictions")
    args = parser.parse_args()

    model = load_model(device=args.device)

    if args.predict:
        prob = predict_file(model, args.predict, device=args.device)
        print(f"File: {args.predict}  fake_prob={prob:.6f}")
        if args.csv:
            import csv
            with open(args.csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["file", "fake_prob"])
                writer.writerow([args.predict, prob])
        return

    if args.batch:
        # simple batch: expects subfolders real/ and fake/
        from eval_inference import collect_files, evaluate
        base = Path(args.batch)
        real = base / "real"
        fake = base / "fake"
        files = collect_files(real, fake)
        y_true = []
        rows = []
        for path, label in files:
            prob = predict_file(model, path, device=args.device)
            y_true.append(label)
            rows.append((path, label, prob))
            print(path, label, prob)
        metrics = evaluate(model, files)
        print("Metrics:")
        for k, v in metrics.items():
            print(f"{k}: {v}")
        if args.csv:
            import csv
            with open(args.csv, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(["file", "label", "fake_prob"])
                for r in rows:
                    writer.writerow(r)
        return

    # default: run demo main
    main()


if __name__ == '__main__':
    run_cli()
