"""Evaluate a saved checkpoint on its reproducible patient-level test split."""
import argparse
import json
import sys
from pathlib import Path
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from brain_tumor_seg.data import AxialSliceDataset, discover_cases, patient_split
from brain_tumor_seg.metrics import segmentation_metrics
from brain_tumor_seg.model import UNet

def case_id(path):
    return path.name.removesuffix(".nii.gz").removesuffix(".nii")

def main():
    parser = argparse.ArgumentParser(description="Evaluate Dice, IoU, precision, and recall on held-out patients")
    parser.add_argument("--data", default="data/Task01_BrainTumour")
    parser.add_argument("--checkpoint", default="outputs/run/best_model.pt")
    parser.add_argument("--output", default="outputs/run/test_metrics.json")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    run_args = checkpoint.get("args", {})
    seed = int(checkpoint.get("seed", run_args.get("seed", 42)))
    max_cases = int(run_args.get("max_cases", 0))
    pairs = discover_cases(args.data, max_cases or None)
    train_pairs, val_pairs, test_pairs = patient_split(pairs, seed)
    loader = DataLoader(AxialSliceDataset(test_pairs), batch_size=args.batch_size,
                        num_workers=args.workers, pin_memory=device.type == "cuda")
    model = UNet().to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    totals = {key: 0.0 for key in ("dice", "iou", "precision", "recall")}
    slice_count = 0
    with torch.inference_mode():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            batch_metrics = segmentation_metrics(model(x), y)
            batch_size = y.shape[0]
            for key, value in batch_metrics.items():
                totals[key] += value * batch_size
            slice_count += batch_size
    result = {
        "device": str(device),
        "threshold": 0.5,
        "averaging": "mean over sampled axial slices",
        "empty_slice_convention": "Dice/IoU/precision are 1 when both masks are empty; recall is 1 when reference is empty",
        "patient_counts": {"train": len(train_pairs), "validation": len(val_pairs), "test": len(test_pairs)},
        "test_patients": [case_id(image) for image, _ in test_pairs],
        "metrics": {key: value / slice_count for key, value in totals.items()},
    }
    output = ROOT / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2))
    split = {
        "seed": seed,
        "train": [case_id(image) for image, _ in train_pairs],
        "validation": [case_id(image) for image, _ in val_pairs],
        "test": [case_id(image) for image, _ in test_pairs],
    }
    (output.parent / "split.json").write_text(json.dumps(split, indent=2))
    print(json.dumps(result, indent=2))
    print(f"Saved metrics to {output}")

if __name__ == "__main__":
    main()