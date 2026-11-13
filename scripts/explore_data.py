import argparse
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from brain_tumor_seg.data import discover_cases, load_case


def main():
    parser = argparse.ArgumentParser(description="Explore MSD BrainTumour NIfTI cases")
    parser.add_argument("--data", default="data/Task01_BrainTumour")
    parser.add_argument("--output", default="outputs/data_overview.png")
    parser.add_argument("--max-cases", type=int, default=5)
    args = parser.parse_args()
    pairs = discover_cases(args.data)
    print(f"Paired training cases: {len(pairs)}")
    for image_path, label_path in pairs[:args.max_cases]:
        image, mask = load_case(image_path, label_path)
        tumor_slices = np.flatnonzero(mask.sum(axis=(0, 1)) > 0)
        print(f"{image_path.name}: shape={image.shape}, tumor voxels={int(mask.sum())}, "
              f"tumor axial slices={len(tumor_slices)}")
    image_path, label_path = pairs[0]
    image, mask = load_case(image_path, label_path)
    z = int(np.argmax(mask.sum(axis=(0, 1))))
    fig, axes = plt.subplots(1, 3, figsize=(12, 4))
    for ax, channel, name in zip(axes[:2], (0, 3), ("FLAIR", "T2w")):
        ax.imshow(image[:, :, z, channel].T, cmap="gray", origin="lower")
        ax.set_title(name)
        ax.axis("off")
    axes[2].imshow(image[:, :, z, 0].T, cmap="gray", origin="lower")
    axes[2].imshow(mask[:, :, z].T, cmap="Reds", alpha=0.45, origin="lower")
    axes[2].set_title("FLAIR + whole-tumor label")
    axes[2].axis("off")
    fig.tight_layout()
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=160)
    print(f"Saved example visualization to {output}")


if __name__ == "__main__":
    main()
