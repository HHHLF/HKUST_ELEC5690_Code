"""Report figures: learning curves, confusion matrix, and ROC curves."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
import numpy as np

_FONT_DIR = Path(__file__).resolve().parents[2] / "fonts"


def use_times_new_roman() -> None:
    """Use the project Times New Roman files for every Matplotlib figure."""
    for font_path in sorted(_FONT_DIR.glob("*.[Tt][Tt][Ff]")):
        font_manager.fontManager.addfont(str(font_path))
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.serif"] = ["Times New Roman"]
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


use_times_new_roman()


def _save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    if path.suffix.lower() == ".png":
        fig.savefig(path.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_history(history: list[dict], path: Path, loss_title: str = "Cross-entropy loss") -> None:
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["train_loss"] for row in history], marker="o", label="train")
    axes[0].plot(epochs, [row["val_loss"] for row in history], marker="o", label="val")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title(loss_title)
    axes[0].legend()
    axes[1].plot(epochs, [row["train_accuracy"] for row in history], marker="o", label="train")
    axes[1].plot(epochs, [row["val_accuracy"] for row in history], marker="o", label="val")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Accuracy")
    axes[1].set_title("Accuracy")
    axes[1].legend()
    fig.suptitle("Checkpoint selection uses validation accuracy")
    _save(fig, path)


def plot_confusion(matrix, class_names, path: Path) -> None:
    array = np.asarray(matrix, dtype=np.float64)
    row_sum = array.sum(axis=1, keepdims=True)
    rate = np.divide(array, row_sum, out=np.zeros_like(array), where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(10.2, 9.0))
    image = ax.imshow(rate, cmap="Blues", vmin=0.0, vmax=1.0, aspect="equal")
    colorbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.03)
    colorbar.set_label("Share of true class", fontsize=19)
    colorbar.ax.tick_params(labelsize=17)
    colorbar.set_ticks([0, 0.25, 0.5, 0.75, 1])
    colorbar.set_ticklabels(["0%", "25%", "50%", "75%", "100%"])
    ax.set_xticks(range(len(class_names)), class_names, fontsize=18)
    ax.set_yticks(range(len(class_names)), class_names, fontsize=18)
    ax.set_xticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(class_names), 1), minor=True)
    ax.grid(which="minor", color="white", linestyle="-", linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    ax.tick_params(length=0)
    ax.set_xlabel("Predicted", fontsize=20)
    ax.set_ylabel("True", fontsize=20)
    ax.set_title("Confusion matrix", fontsize=21, pad=12)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            count = int(array[i, j])
            if count == 0:
                label = "0"
            else:
                label = f"{count}\n{rate[i, j] * 100:.1f}%"
            ax.text(
                j,
                i,
                label,
                ha="center",
                va="center",
                color="white" if rate[i, j] >= 0.55 else "#1a1a1a",
                fontsize=16,
                linespacing=1.15,
            )
    _save(fig, path)


def plot_roc(roc_rows: list[dict], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(9.5, 8.2))
    ax.plot([0, 1], [0, 1], linestyle="--", color="0.6", linewidth=2.0, label="random")
    fpr_grid = np.linspace(0, 1, 201)
    interpolated = []
    for row in roc_rows:
        if row["fpr"] is None or row["auroc"] is None:
            continue
        fpr = np.asarray(row["fpr"], dtype=np.float64)
        tpr = np.asarray(row["tpr"], dtype=np.float64)
        ax.plot(fpr, tpr, label=f"{row['class']} AUROC={row['auroc']:.3f}")
        interpolated.append(np.interp(fpr_grid, fpr, tpr))
    if interpolated:
        mean_tpr = np.mean(interpolated, axis=0)
        mean_tpr[0] = 0.0
        mean_tpr[-1] = 1.0
        ax.plot(fpr_grid, mean_tpr, color="black", linewidth=2.8, label="Average ROC")
    ax.set_xlabel("False positive rate", fontsize=20)
    ax.set_ylabel("True positive rate", fontsize=20)
    ax.set_title("Test one-vs-rest ROC", fontsize=22, pad=12)
    ax.tick_params(labelsize=16)
    ax.legend(loc="lower right", fontsize=16, framealpha=0.95)
    _save(fig, path)
