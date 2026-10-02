"""Train the ISIC 2018 baseline. The test set is not used for model selection."""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from isic_baseline.data import audit_splits, load_split, make_loader
from isic_baseline.engine import evaluate_loader, train_one_epoch
from isic_baseline.losses import build_criterion, format_loss_spec, prepare_loss_spec
from isic_baseline.model import DEFAULT_WEIGHTS, build_model
from isic_baseline.plots import plot_history
from isic_baseline.utils import get_device, load_config, save_json, set_seed, start_log


def parse_args():
    parser = argparse.ArgumentParser(description="Fine-tune DINOv2 on ISIC 2018 Task 3 with unweighted CE.")
    parser.add_argument("--config", default=str(ROOT / "configs" / "baseline.yaml"))
    parser.add_argument("--variant", choices=["vitb14", "vits14"])
    parser.add_argument("--weights")
    parser.add_argument("--image-size", type=int)
    parser.add_argument("--epochs", type=int)
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--num-workers", type=int)
    parser.add_argument("--lr-backbone", type=float)
    parser.add_argument("--lr-head", type=float)
    parser.add_argument("--weight-decay", type=float)
    parser.add_argument("--output-dir")
    parser.add_argument("--resume")
    parser.add_argument("--seed", type=int)
    parser.add_argument("--device")
    parser.add_argument("--train-image-dir")
    parser.add_argument("--train-csv")
    parser.add_argument("--val-image-dir")
    parser.add_argument("--val-csv")
    parser.add_argument("--amp", dest="amp", action="store_true", default=None)
    parser.add_argument("--no-amp", dest="amp", action="store_false")
    return parser.parse_args()


def overrides_from_args(args) -> dict:
    mapping = {
        "variant": args.variant,
        "weights": args.weights,
        "image_size": args.image_size,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "num_workers": args.num_workers,
        "lr_backbone": args.lr_backbone,
        "lr_head": args.lr_head,
        "weight_decay": args.weight_decay,
        "output_dir": args.output_dir,
        "resume": args.resume,
        "seed": args.seed,
        "device": args.device,
        "train_image_dir": args.train_image_dir,
        "train_csv": args.train_csv,
        "val_image_dir": args.val_image_dir,
        "val_csv": args.val_csv,
        "amp": args.amp,
    }
    return mapping


def set_epoch_lr(optimizer, epoch: int, warmup: int, epochs: int) -> float:
    if epoch < warmup:
        factor = float(epoch + 1) / max(warmup, 1)
    else:
        progress = (epoch - warmup) / max(epochs - warmup, 1)
        factor = 0.5 * (1.0 + math.cos(math.pi * progress))
    for group in optimizer.param_groups:
        group["lr"] = group["initial_lr"] * factor
    return factor


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config, overrides_from_args(args))
    if args.variant and not args.weights:
        cfg["weights"] = str((ROOT / DEFAULT_WEIGHTS[cfg["variant"]]).resolve())
    if cfg["image_size"] % 14 != 0:
        raise SystemExit(f"image_size must be a multiple of 14, got {cfg['image_size']}")

    set_seed(int(cfg["seed"]))
    device = get_device(cfg["device"])
    output_dir = Path(cfg["output_dir"])
    output_dir.mkdir(parents=True, exist_ok=True)
    start_log(output_dir.with_name(output_dir.name + "_train.log"))

    train_split = load_split("train", cfg["train_image_dir"], cfg["train_csv"])
    val_split = load_split("val", cfg["val_image_dir"], cfg["val_csv"])
    audit = audit_splits({"train": train_split, "val": val_split})
    save_json(output_dir / "data_audit_train_val.json", audit)
    print("Model selection: highest validation accuracy. Validation macro-F1 is logged only.")
    print("The test split is not loaded during training.")
    loss_spec = prepare_loss_spec(cfg, train_split["records"])
    cfg["loss"] = loss_spec
    print(format_loss_spec(loss_spec))
    save_json(output_dir / "loss_spec.json", loss_spec)
    save_json(output_dir / "config_snapshot.json", cfg)
    if loss_spec["name"] != "cross_entropy" and cfg.get("resume"):
        print("Resume is set. This run will continue from that checkpoint instead of a fresh DINOv2 initialization.")
    else:
        print(f"Initializing backbone from pretrained weights only: {cfg['weights']}")

    model = build_model(cfg["variant"], cfg["weights"], load_pretrained=True)
    criterion = build_criterion(loss_spec, device)
    model.to(device)
    optimizer = torch.optim.AdamW(
        model.param_groups(cfg["lr_backbone"], cfg["lr_head"], cfg["weight_decay"])
    )
    for group in optimizer.param_groups:
        group["initial_lr"] = group["lr"]
    scaler = torch.amp.GradScaler("cuda", enabled=bool(cfg["amp"]) and device.type == "cuda")

    start_epoch = 0
    best_f1 = -1.0
    best_acc = -1.0
    best_epoch = -1
    history = []
    if cfg.get("resume"):
        checkpoint = torch.load(cfg["resume"], map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        if checkpoint.get("scaler"):
            scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_f1 = float(checkpoint.get("best_macro_f1", -1.0))
        best_acc = float(checkpoint.get("best_accuracy", -1.0))
        best_epoch = int(checkpoint.get("best_epoch", -1))
        history = list(checkpoint.get("history", []))
        print(f"Resumed {cfg['resume']} at epoch {start_epoch} (best accuracy {best_acc:.4f} @ epoch {best_epoch})")

    train_loader = make_loader(
        train_split["records"], cfg["image_size"], cfg["batch_size"], True, cfg["num_workers"], cfg["seed"]
    )
    val_loader = make_loader(
        val_split["records"], cfg["image_size"], cfg["batch_size"], False, cfg["num_workers"], cfg["seed"]
    )

    for epoch in range(start_epoch, int(cfg["epochs"])):
        factor = set_epoch_lr(optimizer, epoch, int(cfg["warmup_epochs"]), int(cfg["epochs"]))
        train_stats = train_one_epoch(
            model, train_loader, optimizer, scaler, device, bool(cfg["amp"]), criterion
        )
        val_stats = evaluate_loader(model, val_loader, device, bool(cfg["amp"]), criterion)
        row = {
            "epoch": epoch + 1,
            "lr_factor": factor,
            "lr_backbone": optimizer.param_groups[0]["lr"],
            "lr_head": optimizer.param_groups[-1]["lr"],
            "train_loss": train_stats["loss"],
            "train_accuracy": train_stats["accuracy"],
            "val_loss": val_stats["loss"],
            "val_accuracy": val_stats["accuracy"],
            "val_macro_f1": val_stats["macro_f1"],
        }
        history.append(row)
        print(
            f"epoch {row['epoch']}/{cfg['epochs']} "
            f"train_loss={row['train_loss']:.4f} train_acc={row['train_accuracy']:.4f} "
            f"val_loss={row['val_loss']:.4f} val_acc={row['val_accuracy']:.4f} "
            f"val_macro_f1={row['val_macro_f1']:.4f} lr_head={row['lr_head']:.2e}"
        )
        last = {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scaler": scaler.state_dict(),
            "best_macro_f1": best_f1,
            "best_accuracy": best_acc,
            "best_epoch": best_epoch,
            "history": history,
            "config": cfg,
        }
        val_f1 = -1.0 if val_stats["macro_f1"] is None else float(val_stats["macro_f1"])
        improved = val_stats["accuracy"] > best_acc + 1e-12 or (
            abs(val_stats["accuracy"] - best_acc) <= 1e-12 and val_f1 > best_f1 + 1e-12
        )
        if improved:
            best_f1 = val_f1
            best_acc = float(val_stats["accuracy"])
            best_epoch = epoch + 1
            last["best_macro_f1"] = best_f1
            last["best_accuracy"] = best_acc
            last["best_epoch"] = best_epoch
            torch.save(
                {
                    "epoch": epoch,
                    "model": model.state_dict(),
                    "best_macro_f1": best_f1,
                    "best_accuracy": best_acc,
                    "best_epoch": best_epoch,
                    "history": history,
                    "config": cfg,
                },
                output_dir / "best.pt",
            )
            print(f"  saved best.pt by validation accuracy={best_acc:.4f} (val macro-F1={best_f1:.4f})")
        torch.save(last, output_dir / "last.pt")
        save_json(output_dir / "history.json", history)

    if history:
        loss_title = {
            "cross_entropy": "Cross-entropy loss",
            "weighted_cross_entropy": "Weighted cross-entropy loss",
            "focal": "Focal loss",
        }.get(cfg["loss"]["name"], "Loss")
        plot_history(history, output_dir / "figures" / "loss_acc_curves.png", loss_title=loss_title)
    pd.DataFrame(history).to_csv(output_dir / "history.csv", index=False)
    print(
        f"Training finished. Best epoch={best_epoch}, val accuracy={best_acc:.4f}, val macro-F1={best_f1:.4f}. "
        "Run evaluate.py on the test set after this selection. Test metrics were not computed."
    )


if __name__ == "__main__":
    main()
