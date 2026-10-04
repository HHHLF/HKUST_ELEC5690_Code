"""2D training. Checkpoints follow full-volume validation Dice, not slice Dice."""

from __future__ import annotations

import csv
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from la_seg2d.data import SliceCropDataset, collate_crops, open_store, slice_manifest
from la_seg2d.infer import predict_volume_from_slices
from la_seg2d.losses import ce_dice_loss
from la_seg2d.model import UNet2D, count_parameters
from la_seg2d.protocol import build_protocol
from la_seg3d.utils import runtime_info, seed_everything, write_json


def volume_overlap_metrics(prediction: np.ndarray, ground_truth: np.ndarray) -> dict:
    """Dice and Jaccard branch of la_seg3d.metrics.binary_segmentation_metrics.

    Empty-mask rules match that function. ASD and HD95 are left out on purpose.
    """
    pred = np.asarray(prediction).astype(bool)
    gt = np.asarray(ground_truth).astype(bool)
    if pred.shape != gt.shape or pred.ndim != 3:
        raise ValueError(f"expected matching 3D masks, got {pred.shape} vs {gt.shape}")
    pred_empty = not bool(pred.any())
    gt_empty = not bool(gt.any())
    intersection = int(np.logical_and(pred, gt).sum())
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    union = pred_sum + gt_sum - intersection
    if pred_empty and gt_empty:
        dice, jaccard, empty_status = 1.0, 1.0, "both_empty"
    elif pred_empty or gt_empty:
        dice, jaccard, empty_status = 0.0, 0.0, "pred_empty" if pred_empty else "gt_empty"
    else:
        dice = (2.0 * intersection) / (pred_sum + gt_sum)
        jaccard = intersection / union if union else 0.0
        empty_status = "none"
    return {
        "dice": float(dice),
        "jaccard": float(jaccard),
        "pred_foreground_voxels": pred_sum,
        "gt_foreground_voxels": gt_sum,
        "empty_status": empty_status,
    }


def build_model(cfg: dict) -> UNet2D:
    model_cfg = cfg["model"]
    return UNet2D(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        channels=[int(v) for v in model_cfg["channels"]],
        norm=model_cfg["norm"],
    )


def make_loader(dataset: SliceCropDataset, cfg: dict) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=True,
        num_workers=int(cfg["train"]["num_workers"]),
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        collate_fn=collate_crops,
    )


def evaluate_cases(model, store, case_ids: list[str], cfg: dict, device: torch.device) -> list[dict]:
    if cfg["infer"]["postprocess"] != "none":
        raise RuntimeError("postprocess must stay none, matching the 3D run")
    rows = []
    infer_cfg = cfg["infer"]
    crop = tuple(int(v) for v in cfg["data"]["crop_size_xy"])
    model.eval()
    for case_id in case_ids:
        case = store.cases[case_id]
        prediction, window_info = predict_volume_from_slices(
            model,
            case["image"],
            z_axis=int(cfg["data"]["z_axis"]),
            crop_size=crop,
            full_slice_max_side=int(infer_cfg["full_slice_max_side"]),
            overlap=float(infer_cfg["overlap"]),
            sw_batch_size=int(infer_cfg["sw_batch_size"]),
            slice_batch_size=int(infer_cfg["slice_batch_size"]),
            device=device,
            amp=bool(cfg["train"]["amp"]),
            sigma_scale=float(infer_cfg["gaussian_sigma_scale"]),
        )
        if tuple(prediction.shape) != tuple(case["label"].shape):
            raise RuntimeError(f"{case_id} prediction {prediction.shape} != label {case['label'].shape}")
        # Checkpoint selection uses full-volume Dice only. Surface distances are
        # computed at test time by binary_segmentation_metrics; on a speckled
        # early prediction they dominate validation time and do not change the
        # selected checkpoint.
        metrics = volume_overlap_metrics(prediction, case["label"])
        metrics.update(
            {
                "case_id": case_id,
                "shape_xyz": case["shape_xyz"],
                "spacing_xyz": case["spacing_xyz"],
                "distance_unit": "voxel",
                "infer_mode": window_info["mode"],
                "n_slices": window_info["n_slices"],
                "n_windows": window_info["n_windows"],
            }
        )
        rows.append(metrics)
    return rows


def _checkpoint_payload(model, optimizer, scheduler, scaler, epoch, global_step, best, stale, history, cfg):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "global_step": global_step,
        "best_val_dice": best,
        "stale_validations": stale,
        "history": history,
        "selected_on": "val_full_volume_mean_dice",
        "config": cfg,
    }


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _write_history(out_dir: Path, history: list[dict]) -> None:
    write_json(out_dir / "history.json", {"history": history})
    fields = [
        "epoch",
        "global_step",
        "loss",
        "ce_loss",
        "dice_loss",
        "fg_fraction",
        "background_crop_fraction",
        "foreground_sampler_fraction",
        "lr",
        "val_dice",
        "val_jaccard",
    ]
    with (out_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key) for key in fields})


def _plot_history(history: list[dict], path: Path) -> None:
    if not history:
        return
    steps = [row["global_step"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(steps, [row["loss"] for row in history], label="total")
    axes[0].plot(steps, [row["ce_loss"] for row in history], label="ce")
    axes[0].plot(steps, [row["dice_loss"] for row in history], label="dice")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    val_steps = [row["global_step"] for row in history if row.get("val_dice") is not None]
    val_dice = [row["val_dice"] for row in history if row.get("val_dice") is not None]
    if val_steps:
        axes[1].plot(val_steps, val_dice, marker="o")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("full-volume val Dice")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=300)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


def _assert_same_split(store, split_path: Path) -> None:
    saved = json_split(split_path)
    for key in ("train_ids", "val_ids", "test_ids"):
        if list(getattr(store, key)) != list(saved[key]):
            raise RuntimeError(f"2D {key} do not match {split_path}")


def json_split(path: Path) -> dict:
    import json

    return json.loads(path.read_text())


def train(cfg: dict, device: torch.device, resume: str | None, root: Path) -> None:
    seed_everything(int(cfg["seed"]))
    out_dir = root / cfg["output"]["dir"]
    comparison_dir = root / cfg["output"]["comparison_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    started = time.time()

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    if int(cfg["data"]["z_axis"]) != 2:
        raise RuntimeError("z_axis must be 2 for the (X, Y, Z) NRRD arrays")
    if cfg["infer"]["postprocess"] != "none":
        raise RuntimeError("infer.postprocess must be none")

    protocol = build_protocol(cfg, root)
    write_json(out_dir / "comparison_protocol.json", protocol)
    write_json(comparison_dir / "comparison_protocol.json", protocol)
    if not protocol["budget_alignment"]["matched"]:
        log(
            "2D max_steps does not equal the computed 3D optimizer updates: "
            f"{protocol['budget_alignment']}"
        )

    store = open_store(cfg, retain="train_val", keep_raw=False)
    _assert_same_split(store, root / cfg["reference_3d"]["split"])
    write_json(out_dir / "split.json", store.split_payload())
    write_json(out_dir / "slice_manifest.json", slice_manifest(store))
    write_json(out_dir / "runtime.json", runtime_info(device))
    write_json(out_dir / "config_snapshot.json", cfg)

    model = build_model(cfg).to(device)
    n_params = count_parameters(model)
    log(
        f"UNet2D parameters={n_params} UNet3D parameters={protocol['unet3d']['parameters']} "
        f"cases train={len(store.train_ids)} val={len(store.val_ids)} test={len(store.test_ids)}"
    )
    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(train_cfg["scheduler_t_max"]),
        eta_min=float(train_cfg["eta_min"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg["amp"]) and device.type == "cuda")
    dataset = SliceCropDataset(store, store.train_ids, cfg)
    loader = make_loader(dataset, cfg)

    start_epoch = 0
    global_step = 0
    best_dice = -1.0
    stale_vals = 0
    history: list[dict] = []
    if resume:
        checkpoint = torch.load(resume, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        scheduler.load_state_dict(checkpoint["scheduler"])
        scaler.load_state_dict(checkpoint["scaler"])
        start_epoch = int(checkpoint["epoch"])
        global_step = int(checkpoint["global_step"])
        best_dice = float(checkpoint["best_val_dice"])
        stale_vals = int(checkpoint["stale_validations"])
        history = list(checkpoint["history"])
        log(f"resumed from {resume} at step {global_step} best_val_dice={best_dice:.4f}")

    loss_cfg = cfg["loss"]
    accum = int(train_cfg["grad_accum_steps"])
    max_steps = int(train_cfg["max_steps"])
    val_every = int(train_cfg["val_every_steps"])
    scheduler_every = int(train_cfg["scheduler_every_steps"])
    epoch = start_epoch
    micro = 0
    optimizer.zero_grad(set_to_none=True)

    while global_step < max_steps:
        epoch += 1
        dataset.set_epoch(epoch)
        model.train()
        totals, ces, dices, fg_fracs = [], [], [], []
        n_background = 0
        n_fg_sampler = 0
        n_crops = 0
        lr_used = float(optimizer.param_groups[0]["lr"])
        val_row = None
        for batch in loader:
            images = batch["image"].to(device, non_blocking=True)
            labels = batch["label"].to(device, non_blocking=True)
            with torch.autocast(device_type=device.type, enabled=bool(train_cfg["amp"]) and device.type == "cuda"):
                logits = model(images)
                if logits.shape[2:] != images.shape[2:]:
                    raise RuntimeError(f"logits {tuple(logits.shape)} != input {tuple(images.shape)}")
                total, ce, dice = ce_dice_loss(
                    logits,
                    labels,
                    ce_weight=float(loss_cfg["ce_weight"]),
                    dice_weight=float(loss_cfg["dice_weight"]),
                    dice_smooth=float(loss_cfg["dice_smooth"]),
                )
            scaler.scale(total / accum).backward()
            micro += 1
            n_crops += int(labels.shape[0])
            n_background += sum(1 for flag in batch["background_slice"] if flag)
            n_fg_sampler += sum(1 for flag in batch["used_foreground_sampler"] if flag)
            stepped = micro % accum == 0
            if stepped:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1
                if global_step % scheduler_every == 0:
                    scheduler.step()
                totals.append(float(total.detach()))
                ces.append(float(ce.detach()))
                dices.append(float(dice.detach()))
                fg_fracs.append(float((labels == 1).float().mean()))
                if global_step % val_every == 0 or global_step >= max_steps:
                    val_row = _validate(model, store, cfg, device, log, global_step)
                    model.train()
                if global_step >= max_steps:
                    break
        if micro % accum != 0:
            raise RuntimeError("epoch ended with a partial accumulation and max_steps was not reached")
        row = {
            "epoch": epoch,
            "global_step": global_step,
            "loss": float(np.mean(totals)) if totals else None,
            "ce_loss": float(np.mean(ces)) if ces else None,
            "dice_loss": float(np.mean(dices)) if dices else None,
            "fg_fraction": float(np.mean(fg_fracs)) if fg_fracs else None,
            "background_crop_fraction": n_background / max(n_crops, 1),
            "foreground_sampler_fraction": n_fg_sampler / max(n_crops, 1),
            "lr": lr_used,
            "val_dice": None if val_row is None else val_row["val_dice"],
            "val_jaccard": None if val_row is None else val_row["val_jaccard"],
            "n_pred_empty": None if val_row is None else val_row["n_pred_empty"],
        }
        if val_row is not None:
            improved = val_row["val_dice"] > best_dice
            if improved:
                best_dice = val_row["val_dice"]
                stale_vals = 0
                _save_checkpoint(
                    out_dir / "best.pt",
                    _checkpoint_payload(
                        model, optimizer, scheduler, scaler, epoch, global_step, best_dice, stale_vals, history + [row], cfg
                    ),
                )
                write_json(out_dir / "val_best_metrics.json", {"mean_dice": best_dice, "cases": val_row["cases"]})
            else:
                stale_vals += 1
            log(
                f"epoch {epoch} step {global_step} loss={row['loss']:.4f} "
                f"val_dice={val_row['val_dice']:.4f} empty={val_row['n_pred_empty']}/{len(store.val_ids)} "
                f"bg_crops={row['background_crop_fraction']:.3f}"
            )
        else:
            log(
                f"epoch {epoch} step {global_step} loss={row['loss']:.4f} "
                f"bg_crops={row['background_crop_fraction']:.3f} fg_sampler={row['foreground_sampler_fraction']:.3f}"
            )
        history.append(row)
        _write_history(out_dir, history)
        if val_row is not None or epoch % 20 == 0 or global_step >= max_steps:
            _plot_history(history, out_dir / "figures" / "loss_val_dice.png")
        _save_checkpoint(
            out_dir / "last.pt",
            _checkpoint_payload(model, optimizer, scheduler, scaler, epoch, global_step, best_dice, stale_vals, history, cfg),
        )
        if val_row is not None and stale_vals >= int(train_cfg["early_stopping_patience"]):
            log(f"early stop at step {global_step}: {stale_vals} validations without improvement")
            break

    elapsed = time.time() - started
    write_json(
        out_dir / "train_finished.json",
        {
            "seconds": elapsed,
            "best_val_dice": best_dice,
            "epochs_ran": history[-1]["epoch"] if history else 0,
            "optimizer_updates": global_step,
            "batch_size": int(train_cfg["batch_size"]),
            "samples_per_epoch": int(train_cfg["samples_per_epoch"]),
            "samples_drawn": global_step * int(train_cfg["batch_size"]) * accum,
            "note": "optimizer_updates match the 3D count only as configured. They are not equal FLOPs.",
        },
    )
    log(f"finished in {elapsed / 3600:.2f} h, best full-volume val Dice {best_dice:.4f}, steps {global_step}")


def _validate(model, store, cfg, device, log, global_step) -> dict:
    t0 = time.time()
    rows = evaluate_cases(model, store, store.val_ids, cfg, device)
    val_dice = float(np.mean([item["dice"] for item in rows]))
    val_jaccard = float(np.mean([item["jaccard"] for item in rows]))
    n_empty = sum(item["empty_status"] == "pred_empty" for item in rows)
    log(f"validation step {global_step} dice={val_dice:.4f} in {time.time() - t0:.1f}s")
    return {"val_dice": val_dice, "val_jaccard": val_jaccard, "n_pred_empty": n_empty, "cases": rows}
