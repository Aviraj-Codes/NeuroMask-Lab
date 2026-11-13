import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from brain_tumor_seg.data import discover_cases, load_case
from brain_tumor_seg.model import UNet


def main():
    parser = argparse.ArgumentParser(description="Visualize a model prediction on one labeled case")
    parser.add_argument("--data", default="data/Task01_BrainTumour")
    parser.add_argument("--checkpoint", default="outputs/run/best_model.pt")
    parser.add_argument("--case-index", type=int, default=0)
    parser.add_argument("--output", default="outputs/prediction.png")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cases = discover_cases(args.data)
    image_path, label_path = cases[args.case_index]
    image, truth = load_case(image_path, label_path)
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model = UNet().to(device); model.load_state_dict(checkpoint["model"]); model.eval()
    z = int(np.argmax(truth.sum(axis=(0, 1))))
    x = torch.from_numpy(image[:, :, z, :].transpose(2, 0, 1)[None].copy()).to(device)
    with torch.no_grad(): pred = (torch.sigmoid(model(x))[0, 0].cpu().numpy() >= args.threshold)
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    axes[0].imshow(image[:, :, z, 0].T, cmap="gray", origin="lower"); axes[0].set_title("FLAIR")
    axes[1].imshow(image[:, :, z, 0].T, cmap="gray", origin="lower")
    axes[1].imshow(truth[:, :, z].T, cmap="Greens", alpha=.45, origin="lower"); axes[1].set_title("Reference")
    axes[2].imshow(image[:, :, z, 0].T, cmap="gray", origin="lower")
    axes[2].imshow(pred.T, cmap="Reds", alpha=.45, origin="lower"); axes[2].set_title("U-Net prediction")
    for ax in axes: ax.axis("off")
    fig.suptitle(f"{image_path.name} | axial slice {z}"); fig.tight_layout()
    output = ROOT / args.output; output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160); print(f"Saved {output}")


if __name__ == "__main__": main()
