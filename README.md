# Deepfake Voice Detector — local test harness

Quick scripts to run the `koyelog/deepfake-voice-detector-sota` model locally for testing.

Files added:
- `inference_custom.py` — native `MelodyMachine/Deepfake-audio-detection-V2` loader, `load_model()`, `predict_file()` and CLI (`--predict`, `--batch`).
- `eval_inference.py` — small batch evaluation harness that creates `test_data/real` and `test_data/fake` if missing and computes metrics.
- `test_pipeline.py` — minimal HF pipeline test.
- `requirements.txt` — runtime requirements.
- 
- `test_pipeline.py` — minimal HF pipeline test.
- `requirements.txt` — runtime requirements.

Basic usage (PowerShell):

Create venv and install deps:
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
.venv\Scripts\python -m pip install --upgrade pip
.venv\Scripts\python -m pip install -r requirements.txt
```

Single-file prediction:
```powershell
.venv\Scripts\python inference_custom.py --predict example.wav
```

Batch evaluation (expects `dir/real/*.wav` and `dir/fake/*.wav`):
```powershell
.venv\Scripts\python inference_custom.py --batch test_data --csv results.csv
```

Web UI (upload or record):
```powershell
.venv\Scripts\python app.py
```
Then open:
```text
http://localhost:5000
```
Use either the file upload box or the microphone recording button to analyze audio in the browser.

### Use from another device on the same Wi-Fi

Find this computer's local IPv4 address:
```powershell
ipconfig
```
Start the app, then open `http://YOUR_IPV4_ADDRESS:5000` on a phone, tablet, or another computer. For example:
```text
http://192.168.1.25:5000
```
If Windows Firewall prompts you, allow Python access on **Private networks**. The responsive interface works on mobile and desktop browsers.

Browser microphone recording normally requires a secure context. `localhost` is allowed, but a plain `http://192.168...` address may block microphone access on some mobile browsers. Uploading audio works over the local network. For recording from another device, run the app behind HTTPS (for example with a reverse proxy or a trusted local certificate).

### Publish the app publicly with Render

The repository includes [render.yaml](C:/Users/adshy/OneDrive/Documents/SIH/render.yaml), which configures a public HTTPS web service.

1. Push this project to a GitHub repository.
2. Create an account at [Render](https://render.com/) and choose **New > Blueprint**.
3. Connect the GitHub repository and deploy the detected `render.yaml`.
4. Wait for the first build. It downloads the approximately 378 MB Hugging Face model.
5. Open the generated `https://...onrender.com` URL and share it.

Render provides HTTPS automatically, so microphone recording works on supported mobile browsers. The service uses one worker because the model is large; the Starter plan is recommended for enough memory. The free plan may run out of memory or sleep when idle.

The public service accepts audio uploads up to 16 MB. Do not commit secrets or model tokens to the repository. If Hugging Face rate limits the build, add an `HF_TOKEN` environment variable in Render using a secret value.

### Free public demo with Cloudflare Tunnel

Cloudflare Tunnel can expose the local app through a temporary public HTTPS URL without a paid hosting plan. Your computer must remain on, and the app must keep running.

1. Install `cloudflared` from the [official downloads page](https://developers.cloudflare.com/cloudflare-one/connections/connect-networks/downloads/), then make sure `cloudflared` works in PowerShell:
   ```powershell
   cloudflared --version
   ```
2. Start the public app:
   ```powershell
   .\start_public.ps1
   ```
3. Copy the `https://*.trycloudflare.com` URL printed by Cloudflare and share it.

If PowerShell blocks the script, run:
```powershell
Set-ExecutionPolicy -Scope Process Bypass
.\start_public.ps1
```

The URL changes whenever the tunnel restarts. Never expose private files or secrets through the public app.

Notes:
- The app uses the native Hugging Face Wav2Vec2 classifier `MelodyMachine/Deepfake-audio-detection-V2`. Its label mapping is `fake=0`, `real=1`.
- The inference boundary applies the calibrated inversion enabled by `INVERT_MODEL_LABELS` because the published mapping was reversed on the user's labeled real/fake samples.
Notes:
- The app uses the native Hugging Face Wav2Vec2 classifier `MelodyMachine/Deepfake-audio-detection-V2`. Its label mapping is `fake=0`, `real=1`.
- The inference boundary applies the calibrated inversion enabled by `INVERT_MODEL_LABELS` because the published mapping was reversed on the user's labeled real/fake samples.
- The first run downloads approximately 378 MB of model weights into the Hugging Face cache.
- The repo's model is optimized for 4s clips at 16kHz; ensure audio is preprocessed accordingly. The scripts produce demo 4s files if none exist.
- Results near the 0.5 threshold are reported as `Uncertain` instead of forcing a binary decision.
- For meaningful evaluation replace `test_data` with real labeled datasets (ASVspoof, WaveFake, etc.).
