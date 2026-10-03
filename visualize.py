"""Learning curves, test ROC is produced by evaluate.py. This script builds Grad-CAM cases."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from PIL import Image

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from isic_baseline import CLASS_NAMES, IMAGENET_MEAN, IMAGENET_STD
from isic_baseline.data import ISICDataset
from isic_baseline.gradcam import generate_gradcam
from isic_baseline.model import build_model
from isic_baseline.plots import plot_history
from isic_baseline.utils import get_device, load_config, load_json


def parse_args():
    parser = argparse.ArgumentParser(description="Plot curves and Grad-CAM panels for the selected model.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--predictions")
    parser.add_argument("--history")
    parser.add_argument("--output-dir")
    parser.add_argument("--device")
    return parser.parse_args()


def denormalize(image: torch.Tensor) -> np.ndarray:
    mean = image.new_tensor(IMAGENET_MEAN).view(3, 1, 1)
    std = image.new_tensor(IMAGENET_STD).view(3, 1, 1)
    array = (image * std + mean).clamp(0, 1)
    return array.permute(1, 2, 0).cpu().numpy()


def heatmap_stats(cam: np.ndarray) -> dict:
    total = float(cam.sum()) + 1e-8
    height, width = cam.shape
    peak_y, peak_x = np.unravel_index(int(cam.argmax()), cam.shape)
    quads = {
        "top_left": cam[: height // 2, : width // 2].sum() / total,
        "top_right": cam[: height // 2, width // 2 :].sum() / total,
        "bottom_left": cam[height // 2 :, : width // 2].sum() / total,
        "bottom_right": cam[height // 2 :, width // 2 :].sum() / total,
    }
    y0, y1 = height // 4, height - height // 4
    x0, x1 = width // 4, width - width // 4
    center = float(cam[y0:y1, x0:x1].sum()) / total
    hottest = max(quads, key=quads.get)
    return {
        "peak_y_frac": float(peak_y / max(height - 1, 1)),
        "peak_x_frac": float(peak_x / max(width - 1, 1)),
        "hottest_quadrant": hottest,
        "quadrant_mass": {key: float(value) for key, value in quads.items()},
        "center_mass": center,
        "border_mass": float(1.0 - center),
    }


def choose_cases(frame: pd.DataFrame) -> list[dict]:
    work = frame.copy()
    prob_cols = [f"prob_{name}" for name in CLASS_NAMES]
    work["confidence"] = work[prob_cols].max(axis=1)
    ranked = np.sort(work[prob_cols].to_numpy(), axis=1)
    work["margin"] = ranked[:, -1] - ranked[:, -2]
    work["correct"] = work["true_label"] == work["pred_label"]
    correct = work[work["correct"]].sort_values(["confidence", "image_id"], ascending=[False, True])
    wrong = work[~work["correct"]].sort_values(["confidence", "image_id"], ascending=[False, True])
    chosen = []
    used = set()
    for _, row in correct.iterrows():
        if row["true_label"] in used and len(chosen) > 0:
            continue
        chosen.append(row)
        used.add(row["true_label"])
        if len(chosen) == 2:
            break
    if len(chosen) < 2:
        for _, row in correct.iterrows():
            if row["image_id"] in {item["image_id"] for item in chosen}:
                continue
            chosen.append(row)
            if len(chosen) == 2:
                break
    incorrect = []
    if len(wrong):
        incorrect.append(wrong.iloc[0])
        ambiguous = wrong.sort_values(["margin", "image_id"], ascending=[True, True])
        for _, row in ambiguous.iterrows():
            if row["image_id"] != incorrect[0]["image_id"]:
                incorrect.append(row)
                break
    cases = [{"kind": "correct", "row": row} for row in chosen] + [
        {"kind": "incorrect", "row": row} for row in incorrect
    ]
    return cases


def write_analysis(path: Path, records: list[dict], layer_name: str) -> None:
    lines = [
        "# Grad-CAM case notes",
        "",
        "These test images were chosen after the checkpoint was selected by validation accuracy.",
        f"Heatmap layer: `{layer_name}`. The CLS token is removed and the patch tokens are reshaped onto the patch grid.",
        "High-response locations below are computed from the heatmap. They are not a medical diagnosis. Check the original image before writing an interpretation.",
        "",
    ]
    if len(records) < 4:
        lines.append(
            f"Only {len(records)} usable cases were found (2 correct and 2 incorrect are required). Do not invent images."
        )
        lines.append("")
    for index, record in enumerate(records, start=1):
        row = record["row"]
        stats = record["stats"]
        probs = {name: float(row[f"prob_{name}"]) for name in CLASS_NAMES}
        ranked = sorted(probs, key=probs.get, reverse=True)
        lines.extend(
            [
                f"## Case {index} ({'correct' if record['kind'] == 'correct' else 'incorrect'})",
                f"- image ID: {row['image_id']}",
                f"- true label: {row['true_label']}",
                f"- predicted label: {row['pred_label']}",
                f"- predicted probability: {probs[row['pred_label']]:.4f}",
                f"- second-highest class: {ranked[1]} ({probs[ranked[1]]:.4f}), margin {probs[ranked[0]] - probs[ranked[1]]:.4f}",
                (
                    f"- observed high-response region: peak at about {stats['peak_y_frac']:.0%} of the image height "
                    f"and {stats['peak_x_frac']:.0%} of the width; "
                    f"hottest quadrant is {stats['hottest_quadrant']} "
                    f"({stats['quadrant_mass'][stats['hottest_quadrant']]:.0%}); "
                    f"the center 50% holds {stats['center_mass']:.0%} of the activation mass "
                    f"and the border holds {stats['border_mass']:.0%}."
                ),
                "- Fill in after looking at the image. Do not write a pathological conclusion before that: does the high response fall on the lesion interior, the lesion border, or a non-lesion region such as a ruler, hair, or black edge?",
            ]
        )
        if record["kind"] == "correct":
            lines.append("- Possible misclassification reason: this prediction is correct, so none is listed. Record only whether the visible pattern matches the high-response location.")
        else:
            second_is_true = ranked[1] == row["true_label"]
            lines.append(
                "- Possible misclassification reason (check the numbers, then the image; do not invent a histological cause): "
                f"predicted {row['pred_label']} but the true label is {row['true_label']}."
                + (
                    " The true class is the second-highest probability, so this is a confusion between nearby classes."
                    if second_is_true
                    else " The true class is not the second-highest probability; most of the probability went to another class."
                )
                + (
                    " The heatmap mass is toward the border, so check whether the model responds to the frame or an artifact."
                    if stats["border_mass"] >= 0.6
                    else " The heatmap mass is toward the center, so check whether the center color and border look more like the predicted class."
                )
            )
            lines.append("- Manual note (write this after viewing the original, heatmap, and overlay): ")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, {"device": args.device})
    train_dir = Path(cfg["output_dir"])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else train_dir / "best.pt"
    history_path = Path(args.history) if args.history else train_dir / "history.json"
    predictions_path = Path(args.predictions) if args.predictions else train_dir / "test" / "predictions.csv"
    output_dir = Path(args.output_dir) if args.output_dir else train_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)

    if history_path.is_file():
        loss_title = "Cross-entropy loss"
        snapshot_path = history_path.parent / "config_snapshot.json"
        if snapshot_path.is_file():
            loss_name = load_json(snapshot_path).get("loss", {}).get("name", "cross_entropy")
            loss_title = {
                "cross_entropy": "Cross-entropy loss",
                "weighted_cross_entropy": "Weighted cross-entropy loss",
                "focal": "Focal loss",
            }.get(loss_name, "Loss")
        plot_history(load_json(history_path), output_dir / "loss_acc_curves.png", loss_title=loss_title)
        print(f"Saved curves to {output_dir / 'loss_acc_curves.png'}")
    else:
        print(f"History not found, skipped curves: {history_path}")

    if not predictions_path.is_file():
        raise SystemExit(f"Predictions not found: {predictions_path}. Run evaluate.py first.")
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")

    frame = pd.read_csv(predictions_path)
    cases = choose_cases(frame)
    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved = checkpoint["config"]
    device = get_device(cfg["device"])
    model = build_model(saved["variant"], weights_path=None, load_pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)
    model.eval()

    by_id = {row["image_id"]: row for _, row in frame.iterrows()}
    records_for_template = []
    panel_images = []
    layer_name = ""
    gradcam_dir = output_dir / "gradcam"
    gradcam_dir.mkdir(parents=True, exist_ok=True)
    test_split_records = None
    from isic_baseline.data import load_split

    test_split_records = {
        record["image_id"]: record
        for record in load_split("test", cfg["test_image_dir"], cfg["test_csv"])["records"]
    }
    dataset = ISICDataset(list(test_split_records.values()), int(saved["image_size"]), train=False)

    for case in cases:
        row = case["row"]
        image_id = row["image_id"]
        record = test_split_records[image_id]
        tensor, label, _ = dataset[list(test_split_records).index(image_id)]
        class_index = CLASS_NAMES.index(row["pred_label"])
        batch = tensor.unsqueeze(0).to(device)
        heatmap, layer_name = generate_gradcam(model, batch, class_index)
        original = denormalize(tensor)
        color = plt.get_cmap("jet")(heatmap)[..., :3]
        overlay = np.clip(0.55 * original + 0.45 * color, 0, 1)
        Image.fromarray((original * 255).astype(np.uint8)).save(gradcam_dir / f"{image_id}_original.png")
        Image.fromarray((color * 255).astype(np.uint8)).save(gradcam_dir / f"{image_id}_heatmap.png")
        Image.fromarray((overlay * 255).astype(np.uint8)).save(gradcam_dir / f"{image_id}_overlay.png")
        stats = heatmap_stats(heatmap)
        panel_images.append((original, color, overlay, row, stats))
        records_for_template.append({"kind": case["kind"], "row": by_id[image_id], "stats": stats})
        pred_prob = float(row[f"prob_{row['pred_label']}"])
        print(
            f"{image_id} true={row['true_label']} pred={row['pred_label']} "
            f"prob={pred_prob:.3f} layer={layer_name} "
            f"cam_sum={float(heatmap.sum()):.2f} cam_std={float(heatmap.std()):.3f}"
        )

    columns = 3
    fig, axes = plt.subplots(len(panel_images), columns, figsize=(9, 3 * max(len(panel_images), 1)))
    if len(panel_images) == 1:
        axes = np.expand_dims(axes, 0)
    for row_axes, (original, color, overlay, row, _stats) in zip(axes, panel_images):
        row_axes[0].imshow(original)
        row_axes[1].imshow(color)
        row_axes[2].imshow(overlay)
        prob = float(row[f"prob_{row['pred_label']}"])
        row_axes[0].set_ylabel(row["image_id"], fontsize=17)
        titles = [
            f"true={row['true_label']}",
            f"pred={row['pred_label']} p={prob:.2f}",
            "overlay",
        ]
        for axis, title in zip(row_axes, titles):
            axis.set_title(title, fontsize=19)
            axis.set_xticks([])
            axis.set_yticks([])
    fig.suptitle("Grad-CAM of Success and Failure Cases", fontsize=23)
    fig.tight_layout()
    fig.savefig(gradcam_dir / "panel.png", dpi=160, bbox_inches="tight")
    fig.savefig(gradcam_dir / "panel.pdf", bbox_inches="tight")
    plt.close(fig)
    write_analysis(gradcam_dir / "case_analysis.md", records_for_template, layer_name)
    print(f"Wrote {gradcam_dir / 'panel.png'} and case_analysis.md")


if __name__ == "__main__":
    main()
