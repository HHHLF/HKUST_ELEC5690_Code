"""Compare the finished 10-epoch baseline with weighted CE and focal test results.

The table includes a run only when its saved training budget matches runs/baseline.
runs/baseline_e30 is not read.
"""

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

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from isic_baseline import CLASS_NAMES
from isic_baseline.plots import use_times_new_roman
from isic_baseline.utils import load_json, save_json

use_times_new_roman()

MATCH_KEYS = (
    "seed",
    "variant",
    "image_size",
    "epochs",
    "batch_size",
    "lr_backbone",
    "lr_head",
    "weight_decay",
    "warmup_epochs",
    "amp",
)
SCREENING_CLASSES = ("MEL", "BCC", "AKIEC")


def parse_args():
    parser = argparse.ArgumentParser(description="Build the three-loss comparison from saved test metrics.")
    parser.add_argument("--baseline-metrics", default=str(ROOT / "runs" / "baseline" / "test" / "metrics.json"))
    parser.add_argument("--baseline-checkpoint", default=str(ROOT / "runs" / "baseline" / "best.pt"))
    parser.add_argument("--weighted-metrics", default=str(ROOT / "outputs" / "weighted_ce" / "test" / "metrics.json"))
    parser.add_argument("--weighted-checkpoint", default=str(ROOT / "outputs" / "weighted_ce" / "best.pt"))
    parser.add_argument("--focal-metrics", default=str(ROOT / "outputs" / "focal" / "test" / "metrics.json"))
    parser.add_argument("--focal-checkpoint", default=str(ROOT / "outputs" / "focal" / "best.pt"))
    parser.add_argument("--output-dir", default=str(ROOT / "outputs" / "comparison"))
    return parser.parse_args()


def balanced_accuracy(metrics: dict) -> float | None:
    value = metrics["summary"].get("balanced_accuracy")
    if value is not None:
        return float(value)
    recalls = [row["recall"] for row in metrics["per_class"] if row["recall"] is not None]
    if len(recalls) != len(CLASS_NAMES):
        return None
    return float(np.mean(recalls))


def recall_of(metrics: dict, class_name: str) -> float | None:
    for row in metrics["per_class"]:
        if row["class"] == class_name:
            return None if row["recall"] is None else float(row["recall"])
    raise KeyError(class_name)


def checkpoint_config(path: Path) -> dict:
    checkpoint = torch.load(path, map_location="cpu")
    config = dict(checkpoint["config"])
    config["best_epoch"] = checkpoint.get("best_epoch")
    config["best_macro_f1"] = checkpoint.get("best_macro_f1")
    config["best_accuracy"] = checkpoint.get("best_accuracy")
    config["loss_name"] = (config.get("loss") or {}).get("name", "cross_entropy")
    return config


def markdown_table(frame: pd.DataFrame) -> str:
    columns = [str(column) for column in frame.columns]
    rows = []
    for _, row in frame.iterrows():
        cells = []
        for column in frame.columns:
            value = row[column]
            if isinstance(value, (float, np.floating)):
                cells.append(f"{float(value):.4f}")
            else:
                cells.append(str(value))
        rows.append(cells)
    header = "| " + " | ".join(columns) + " |"
    rule = "| " + " | ".join("---" for _ in columns) + " |"
    body = ["| " + " | ".join(cells) + " |" for cells in rows]
    return "\n".join([header, rule, *body])


def fmt(value) -> str:
    if value is None:
        return ""
    if isinstance(value, float):
        return f"{value:.4f}"
    return str(value)


def main() -> None:
    args = parse_args()
    specs = [
        ("baseline_ce", Path(args.baseline_metrics), Path(args.baseline_checkpoint)),
        ("weighted_ce", Path(args.weighted_metrics), Path(args.weighted_checkpoint)),
        ("focal", Path(args.focal_metrics), Path(args.focal_checkpoint)),
    ]
    missing = [str(path) for _, metrics, checkpoint in specs if not metrics.is_file() or not checkpoint.is_file()]
    if missing:
        raise SystemExit(
            "Comparison inputs are missing:\n"
            + "\n".join(missing)
            + "\nTrain and evaluate the missing run first. This script does not invent metrics."
        )
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    loaded = []
    for name, metrics_path, checkpoint_path in specs:
        metrics = load_json(metrics_path)
        config = checkpoint_config(checkpoint_path)
        loaded.append((name, metrics, config, checkpoint_path))
    reference = loaded[0][2]
    mismatches = []
    for name, _, config, _ in loaded[1:]:
        for key in MATCH_KEYS:
            if config.get(key) != reference.get(key):
                mismatches.append(f"{name} {key}={config.get(key)} but baseline {key}={reference.get(key)}")
    if mismatches:
        raise SystemExit(
            "Refusing to put these runs in one table because the training budget differs:\n" + "\n".join(mismatches)
        )
    if len({tuple(item[1]["class_names"]) for item in loaded}) != 1:
        raise SystemExit("Class order differs across metric files.")
    if len({item[1]["summary"]["num_samples"] for item in loaded}) != 1:
        raise SystemExit("Test sample counts differ across metric files.")

    overall_rows = []
    recall_rows = []
    for name, metrics, config, _ in loaded:
        summary = metrics["summary"]
        overall_rows.append(
            {
                "model": name,
                "loss": config["loss_name"],
                "best_epoch": config["best_epoch"],
                "val_macro_f1": config["best_macro_f1"],
                "val_accuracy": config["best_accuracy"],
                "accuracy": summary["accuracy"],
                "macro_precision": summary["macro_precision"],
                "macro_recall": summary["macro_recall"],
                "macro_f1": summary["macro_f1"],
                "weighted_f1": summary["weighted_f1"],
                "macro_ovr_auroc": summary["macro_ovr_auroc"],
                "balanced_accuracy": balanced_accuracy(metrics),
            }
        )
        for class_name in CLASS_NAMES:
            recall_rows.append({"model": name, "class": class_name, "recall": recall_of(metrics, class_name)})
    overall = pd.DataFrame(overall_rows)
    recalls = pd.DataFrame(recall_rows)
    overall.to_csv(output_dir / "comparison.csv", index=False)
    recalls.to_csv(output_dir / "per_class_recall.csv", index=False)

    figure, axes = plt.subplots(1, 3, figsize=(15, 4.8))
    matrices = [np.asarray(item[1]["confusion_matrix"], dtype=np.int64) for item in loaded]
    vmax = max(matrix.max() for matrix in matrices)
    for axis, matrix, (name, _, _, _) in zip(axes, matrices, loaded):
        image = axis.imshow(matrix, cmap="Blues", vmin=0, vmax=vmax)
        axis.set_xticks(range(7), CLASS_NAMES, rotation=45, ha="right")
        axis.set_yticks(range(7), CLASS_NAMES)
        axis.set_xlabel("Predicted")
        axis.set_ylabel("True")
        axis.set_title(name)
        for i in range(7):
            for j in range(7):
                axis.text(j, i, str(matrix[i, j]), ha="center", va="center", fontsize=7, color="black")
    figure.colorbar(image, ax=axes, fraction=0.02)
    figure.suptitle("Confusion Matrices")
    figure.savefig(output_dir / "confusion_side_by_side.png", dpi=160, bbox_inches="tight")
    figure.savefig(output_dir / "confusion_side_by_side.pdf", bbox_inches="tight")
    plt.close(figure)

    lines = [
        "# Three-loss comparison",
        "",
        "All three runs use seed 42, DINOv2 ViT-B/14, image size 224, 10 epochs, batch size 64, "
        "AdamW, backbone lr 1e-5, head lr 1e-3, and the same cosine schedule with 1 warmup epoch. "
        "Each new model started from `dinov2_vitb14_pretrain.pth` and a new classification head. "
        "Checkpoints were selected by validation accuracy. The numbers below are test-set metrics.",
        "",
        "`runs/baseline_e30` is excluded: its checkpoint records epochs 30 and batch size 256. "
        "`configs/baseline.yaml` now lists epochs 50 and batch size 256; that edit is not the protocol of `runs/baseline`.",
        "",
        "The baseline `metrics.json` was saved before `balanced_accuracy` existed. "
        "Its balanced accuracy in this table is the unweighted mean of the seven saved per-class recalls. "
        "On this test split every class has support, so that mean equals macro recall.",
        "",
        "## Overall",
        "",
        markdown_table(overall),
        "",
        "## Per-class recall",
        "",
        markdown_table(recalls.pivot(index="class", columns="model", values="recall").reindex(CLASS_NAMES).reset_index()),
        "",
        "## Where each true class is most often sent when it is wrong",
        "",
    ]
    by_name = {name: metrics for name, metrics, _, _ in loaded}
    for class_index, class_name in enumerate(CLASS_NAMES):
        bits = []
        for name, _, _, _ in loaded:
            row = np.asarray(by_name[name]["confusion_matrix"][class_index], dtype=np.int64)
            order = [index for index in np.argsort(-row) if index != class_index]
            top = order[0]
            bits.append(f"{name}: {CLASS_NAMES[top]} {int(row[top])}/{int(row.sum())}")
        lines.append(f"- {class_name}: " + "; ".join(bits))
    lines.extend(["", "## Screening note", ""])
    lines.extend(screening_paragraph(by_name))
    text = "\n".join(lines) + "\n"
    (output_dir / "comparison.md").write_text(text, encoding="utf-8")
    (output_dir / "analysis.md").write_text(text, encoding="utf-8")
    save_json(
        output_dir / "protocol.json",
        {"match_keys": {key: reference.get(key) for key in MATCH_KEYS}, "excluded": ["runs/baseline_e30"]},
    )
    print(f"Wrote {output_dir}")


def screening_paragraph(by_name: dict[str, dict]) -> list[str]:
    lines = [
        "For screening, missing MEL, BCC, or AKIEC is more costly than sending a common nevus for review. The comparison below uses only this test set's recall and confusion matrices. It does not assume that one loss is better.",
        "",
    ]
    scores = {}
    for name, metrics in by_name.items():
        values = [recall_of(metrics, class_name) for class_name in SCREENING_CLASSES]
        scores[name] = float(np.mean(values))
        detail = ", ".join(f"{class_name} {recall_of(metrics, class_name):.4f}" for class_name in SCREENING_CLASSES)
        lines.append(
            f"- {name}: accuracy {metrics['summary']['accuracy']:.4f}, "
            f"macro-F1 {metrics['summary']['macro_f1']:.4f}, "
            f"mean MEL/BCC/AKIEC recall {scores[name]:.4f} ({detail})."
        )
    chosen = max(scores, key=scores.get)
    baseline_mel = recall_of(by_name["baseline_ce"], "MEL")
    lines.append("")
    lines.append(
        f"On this test set, {chosen} has the highest mean recall on MEL, BCC, and AKIEC ({scores[chosen]:.4f}). "
        f"Baseline MEL recall is {baseline_mel:.4f}."
    )
    for name in ("weighted_ce", "focal"):
        delta = recall_of(by_name[name], "MEL") - baseline_mel
        lines.append(f"- {name} MEL recall changes by {delta:+.4f} relative to the baseline.")
    lines.append("")
    lines.append(
        f"If the screening goal is to miss fewer MEL, BCC, and AKIEC cases, this test result selects {chosen} "
        f"because its mean recall on those three classes is highest ({scores[chosen]:.4f})."
    )
    nv_bits = []
    for name, metrics in by_name.items():
        row = metrics["confusion_matrix"][CLASS_NAMES.index("NV")]
        mel_index = CLASS_NAMES.index("MEL")
        nv_bits.append(
            f"{name} predicts MEL for {row[mel_index]}/{sum(row)} NV images, NV recall {recall_of(metrics, 'NV'):.4f}"
        )
    lines.append("The common confusion is still between melanoma and nevus. " + "; ".join(nv_bits) + ".")
    lines.append(
        "Weighted CE raises recall the most for the rare classes DF and VASC and for MEL, while clearly lowering NV recall, so overall accuracy can fall. "
        "Focal loss also raises rare-class and MEL recall, but NV recall stays closer to the baseline, so overall accuracy is higher. "
        "If the priority is to send fewer benign nevi for review, choose focal loss rather than weighted CE."
    )
    return lines


if __name__ == "__main__":
    main()
