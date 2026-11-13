"""Create portfolio-ready training-loss and segmentation-score plots."""
import argparse
import json
from pathlib import Path
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]

def read_json(path):
    path = ROOT / path
    if not path.exists():
        raise FileNotFoundError(f"Could not find {path}. Train or evaluate the model first.")
    return json.loads(path.read_text(encoding="utf-8"))

def plot_training_curves(history, output):
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(epochs, [row["train_loss"] for row in history], marker="o", label="Training loss")
    if any("val_loss" in row for row in history):
        ax.plot(epochs, [row.get("val_loss") for row in history], marker="o", label="Validation loss")
    ax.set(title="Training and validation loss", xlabel="Epoch", ylabel="Loss")
    ax.grid(alpha=0.25)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output / "training_curves.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def plot_dice_scores(history, test_result, output):
    test_metrics = test_result.get("metrics", test_result)
    has_val_dice = any("val_dice" in row for row in history)
    has_test_scores = any(key in test_metrics for key in ("dice", "iou", "precision", "recall"))
    panels = int(has_val_dice) + int(has_test_scores)
    if panels == 0:
        raise ValueError("No validation Dice or held-out test scores found in the input JSON files.")

    fig, axes = plt.subplots(1, panels, figsize=(7 * panels, 5))
    if panels == 1:
        axes = [axes]
    panel = 0
    if has_val_dice:
        rows = [row for row in history if "val_dice" in row]
        axes[panel].plot([r["epoch"] for r in rows], [r["val_dice"] for r in rows], marker="o")
        axes[panel].set(title="Validation Dice by epoch", xlabel="Epoch", ylabel="Dice")
        axes[panel].set_ylim(0, 1)
        axes[panel].grid(alpha=0.25)
        panel += 1
    if has_test_scores:
        names = [name for name in ("dice", "iou", "precision", "recall") if name in test_metrics]
        values = [test_metrics[name] for name in names]
        bars = axes[panel].bar([name.upper() for name in names], values, color="#3976a8")
        axes[panel].bar_label(bars, fmt="%.3f", padding=3)
        axes[panel].set(title="Held-out test metrics", ylabel="Score")
        axes[panel].set_ylim(0, 1.08)
        axes[panel].grid(axis="y", alpha=0.25)
    fig.tight_layout()
    fig.savefig(output / "dice_scores.png", dpi=180, bbox_inches="tight")
    plt.close(fig)

def main():
    parser = argparse.ArgumentParser(description="Plot saved training history and held-out test scores")
    parser.add_argument("--history", default="outputs/run/history.json")
    parser.add_argument("--metrics", default="outputs/run/test_metrics.json")
    parser.add_argument("--output", default="results")
    args = parser.parse_args()
    history = read_json(args.history)
    test_result = read_json(args.metrics)
    output = ROOT / args.output
    output.mkdir(parents=True, exist_ok=True)
    plot_training_curves(history, output)
    plot_dice_scores(history, test_result, output)
    print(f"Saved {output / 'training_curves.png'}")
    print(f"Saved {output / 'dice_scores.png'}")

if __name__ == "__main__":
    main()