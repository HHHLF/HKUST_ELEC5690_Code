"""Evaluate a selected checkpoint on the test set. Do not use this loop to pick the model."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from isic_baseline import CLASS_NAMES
from isic_baseline.data import audit_splits, load_split, make_loader
from isic_baseline.engine import evaluate_loader, prediction_rows
from isic_baseline.losses import build_criterion
from isic_baseline.metrics import compute_metrics, metrics_to_rows
from isic_baseline.model import build_model
from isic_baseline.plots import plot_confusion, plot_roc
from isic_baseline.utils import get_device, load_config, save_json, set_seed, start_log


def parse_args():
    parser = argparse.ArgumentParser(description="Test-set evaluation for the selected ISIC checkpoint.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--checkpoint")
    parser.add_argument("--output-dir")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--device")
    parser.add_argument("--test-image-dir")
    parser.add_argument("--test-csv")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(
        args.config,
        {
            "batch_size": args.batch_size,
            "num_workers": args.num_workers,
            "device": args.device,
            "test_image_dir": args.test_image_dir,
            "test_csv": args.test_csv,
        },
    )
    train_dir = Path(cfg["output_dir"])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else train_dir / "best.pt"
    if not checkpoint_path.is_file():
        raise SystemExit(f"Checkpoint not found: {checkpoint_path}")
    output_dir = Path(args.output_dir) if args.output_dir else train_dir / "test"
    output_dir.mkdir(parents=True, exist_ok=True)
    start_log(train_dir.with_name(train_dir.name + "_eval.log"))

    checkpoint = torch.load(checkpoint_path, map_location="cpu")
    saved = checkpoint["config"]
    set_seed(int(saved.get("seed", cfg.get("seed", 0))))
    device = get_device(cfg["device"])
    model = build_model(saved["variant"], weights_path=None, load_pretrained=False)
    model.load_state_dict(checkpoint["model"])
    model.to(device)

    test_split = load_split("test", cfg["test_image_dir"], cfg["test_csv"])
    audit = audit_splits({"test": test_split})
    save_json(output_dir / "data_audit_test.json", audit)
    loader = make_loader(
        test_split["records"],
        int(saved["image_size"]),
        int(cfg["batch_size"]),
        False,
        int(cfg["num_workers"]),
        int(saved.get("seed", 0)),
    )
    loss_spec = saved.get("loss") or {"name": "cross_entropy"}
    criterion = build_criterion(loss_spec, device)
    result = evaluate_loader(model, loader, device, use_amp=True, criterion=criterion)
    if result["y_prob"].shape != (result["num_samples"], len(CLASS_NAMES)):
        raise RuntimeError(f"Unexpected probability shape {result['y_prob'].shape}")
    metrics = compute_metrics(result["y_true"], result["y_prob"], list(CLASS_NAMES))
    metrics["loss"] = loss_spec
    metrics["checkpoint"] = str(checkpoint_path)
    metrics["selection_rule"] = (
        "This checkpoint was selected by validation accuracy before test evaluation. "
        f"Saved val accuracy={checkpoint.get('best_accuracy')}, val macro-F1={checkpoint.get('best_macro_f1')}, "
        f"epoch={checkpoint.get('best_epoch')}."
    )
    save_json(output_dir / "metrics.json", metrics)
    pd.DataFrame(metrics_to_rows(metrics)).to_csv(output_dir / "metrics.csv", index=False)
    pd.DataFrame(prediction_rows(result)).to_csv(output_dir / "predictions.csv", index=False)
    pd.DataFrame(metrics["confusion_matrix"], index=CLASS_NAMES, columns=CLASS_NAMES).to_csv(
        output_dir / "confusion_matrix.csv"
    )
    plot_confusion(metrics["confusion_matrix"], CLASS_NAMES, output_dir / "confusion_matrix.png")
    plot_roc(metrics["roc"], output_dir / "roc.png")
    summary = metrics["summary"]
    print(
        f"Test samples={summary['num_samples']} accuracy={summary['accuracy']:.4f} "
        f"macro_f1={summary['macro_f1']:.4f} balanced_accuracy={summary['balanced_accuracy']} "
        f"macro_ovr_auroc={summary['macro_ovr_auroc']}"
    )
    print(f"Wrote {output_dir}")
    if metrics["notes"]:
        print("Notes:")
        for note in metrics["notes"]:
            print(" -", note)


if __name__ == "__main__":
    main()
