from huggingface_hub import hf_hub_download
import torch
import numpy as np


def main():
    print("Downloading pytorch_model.pth from HF repo...")
    path = hf_hub_download(repo_id="koyelog/deepfake-voice-detector-sota", filename="pytorch_model.pth")
    print("Loaded file at:", path)
    print("Inspecting checkpoint keys (top-level)...")
    # Allowlist numpy scalar for safe unpickling if needed
    try:
        torch.serialization.add_safe_globals([np.core.multiarray.scalar])
    except Exception:
        pass
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    if isinstance(ckpt, dict):
        print("Top-level keys:", list(ckpt.keys())[:50])
        if "model_state_dict" in ckpt:
            msd = ckpt["model_state_dict"]
            keys = list(msd.keys())
            print(f"model_state_dict total keys: {len(keys)}")
            print("model_state_dict keys sample (first 60):", keys[:60])
            print("model_state_dict keys sample (last 60):", keys[-60:])
        elif "state_dict" in ckpt:
            sd = ckpt["state_dict"]
            print("state_dict keys sample:", list(sd.keys())[:50])
        else:
            print("No model_state_dict or state_dict found; keys may be top-level model parameters.")
    else:
        print("Checkpoint is not a dict; type:", type(ckpt))


if __name__ == "__main__":
    main()
