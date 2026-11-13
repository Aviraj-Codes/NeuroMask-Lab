import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from brain_tumor_seg.data import AxialSliceDataset, discover_cases, patient_split
from brain_tumor_seg.metrics import dice_iou
from brain_tumor_seg.model import UNet, dice_loss


def seed_everything(seed):
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    if torch.cuda.is_available(): torch.cuda.manual_seed_all(seed)


def evaluate(model, loader, device):
    model.eval(); losses, dices, ious = [], [], []
    bce = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            logits = model(x)
            losses.append((bce(logits, y) + dice_loss(logits, y)).item())
            d, i = dice_iou(logits, y); dices.append(d); ious.append(i)
    return {"loss": float(np.mean(losses)), "dice": float(np.mean(dices)), "iou": float(np.mean(ious))}


def main():
    parser = argparse.ArgumentParser(description="Train a 2D U-Net on MSD Task01 BrainTumour")
    parser.add_argument("--data", default="data/Task01_BrainTumour")
    parser.add_argument("--output", default="outputs/run")
    parser.add_argument("--max-cases", type=int, default=30, help="Small portfolio subset; 0 means all cases")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--workers", type=int, default=0)
    args = parser.parse_args()
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    pairs = discover_cases(args.data, args.max_cases or None)
    train_pairs, val_pairs, test_pairs = patient_split(pairs, args.seed)
    print(f"Device: {device}; patient cases: train={len(train_pairs)}, val={len(val_pairs)}, test={len(test_pairs)}")
    train_loader = DataLoader(AxialSliceDataset(train_pairs), batch_size=args.batch_size, shuffle=False,
                              num_workers=args.workers, pin_memory=device.type == "cuda")
    val_loader = DataLoader(AxialSliceDataset(val_pairs), batch_size=args.batch_size,
                            num_workers=args.workers, pin_memory=device.type == "cuda")
    test_loader = DataLoader(AxialSliceDataset(test_pairs), batch_size=args.batch_size,
                             num_workers=args.workers, pin_memory=device.type == "cuda")
    if not len(train_loader) or not len(val_loader) or not len(test_loader):
        raise ValueError("Need at least three labeled patient cases with nonempty slice sets; use --max-cases >= 3")
    model = UNet().to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr)
    bce = nn.BCEWithLogitsLoss()
    output = ROOT / args.output; output.mkdir(parents=True, exist_ok=True)
    best_val_dice = -1.0; history = []
    for epoch in range(1, args.epochs + 1):
        model.train(); train_losses = []
        for x, y in tqdm(train_loader, desc=f"Epoch {epoch}/{args.epochs}"):
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            logits = model(x); loss = bce(logits, y) + dice_loss(logits, y)
            loss.backward(); optimizer.step(); train_losses.append(loss.item())
        val = evaluate(model, val_loader, device)
        row = {"epoch": epoch, "train_loss": float(np.mean(train_losses)), **{f"val_{k}": v for k, v in val.items()}}
        history.append(row); print(json.dumps(row))
        if val["dice"] > best_val_dice:
            best_val_dice = val["dice"]
            torch.save({"model": model.state_dict(), "seed": args.seed, "epoch": epoch,
                        "val_dice": best_val_dice, "args": vars(args)}, output / "best_model.pt")
    checkpoint = torch.load(output / "best_model.pt", map_location=device, weights_only=False)
    model.load_state_dict(checkpoint["model"])
    test = evaluate(model, test_loader, device)
    (output / "history.json").write_text(json.dumps(history, indent=2))
    (output / "test_metrics.json").write_text(json.dumps(test, indent=2))
    print("Held-out patient test metrics:", json.dumps(test))


if __name__ == "__main__":
    main()
