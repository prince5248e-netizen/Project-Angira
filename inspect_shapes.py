from huggingface_hub import hf_hub_download
import torch


def main():
    path = hf_hub_download(repo_id="koyelog/deepfake-voice-detector-sota", filename="pytorch_model.pth")
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    msd = ckpt.get("model_state_dict", ckpt.get("state_dict", ckpt))
    keys = list(msd.keys())
    print("total keys:", len(keys))
    # print a few selected keys and their shapes
    want = ["conv1.weight", "gru.weight_ih_l0", "gru.weight_hh_l0", "gru.weight_ih_l0_reverse", "attention.in_proj_weight", "classifier.0.weight", "classifier.8.weight"]
    for k in want:
        if k in msd:
            print(k, msd[k].shape)
        else:
            print(k, "MISSING")


if __name__ == '__main__':
    main()
