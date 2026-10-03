"""Training loop. Checkpoints are chosen by full-volume validation Dice."""

from __future__ import annotations

import csv
import queue
import threading
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import DataLoader

from la_seg3d.data import PatchDataset, VolumeStore, write_json
from la_seg3d.infer import predict_volume
from la_seg3d.losses import ce_dice_loss
from la_seg3d.metrics import binary_segmentation_metrics
from la_seg3d.model import UNet3D
from la_seg3d.utils import runtime_info, seed_everything


def build_model(cfg: dict) -> UNet3D:
    model_cfg = cfg["model"]
    return UNet3D(
        in_channels=int(model_cfg["in_channels"]),
        out_channels=int(model_cfg["out_channels"]),
        channels=[int(v) for v in model_cfg["channels"]],
        norm=model_cfg["norm"],
    )


def _loader_worker_init(_worker_id: int) -> None:
    # Workers only crop numpy patches. One thread each avoids oversubscribing the CPU.
    import os

    os.environ["OMP_NUM_THREADS"] = "1"


def make_loader(dataset: PatchDataset, cfg: dict, shuffle: bool) -> DataLoader:
    workers = int(cfg["train"]["num_workers"])
    kwargs = {}
    if workers > 0:
        kwargs["persistent_workers"] = True
        kwargs["prefetch_factor"] = 2
        kwargs["multiprocessing_context"] = "fork"
        kwargs["worker_init_fn"] = _loader_worker_init
    return DataLoader(
        dataset,
        batch_size=int(cfg["train"]["batch_size"]),
        shuffle=shuffle,
        num_workers=workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        **kwargs,
    )


class _BatchPrefetcher:
    """Prepare the next batch on a thread while the GPU runs the current step.

    DataLoader workers stay at 0. Forking them after CUDA starts is unsafe, and
    the volumes already sit in RAM. The thread only overlaps patch sampling and
    the host-to-device copy with the GPU step.
    """

    def __init__(self, loader: DataLoader, device: torch.device):
        self.loader = loader
        self.device = device

    def __iter__(self):
        if self.device.type != "cuda":
            for images, labels in self.loader:
                yield images.to(self.device), labels.to(self.device)
            return
        batches: queue.Queue = queue.Queue(maxsize=2)
        stream = torch.cuda.Stream(device=self.device)
        sentinel = object()

        def produce() -> None:
            try:
                with torch.cuda.stream(stream):
                    for images, labels in self.loader:
                        images = images.to(self.device, non_blocking=True)
                        labels = labels.to(self.device, non_blocking=True)
                        ready = torch.cuda.Event()
                        ready.record(stream)
                        batches.put((images, labels, ready))
            except Exception as exc:
                batches.put(exc)
            finally:
                batches.put(sentinel)

        worker = threading.Thread(target=produce, daemon=True)
        worker.start()
        while True:
            item = batches.get()
            if item is sentinel:
                break
            if isinstance(item, Exception):
                worker.join()
                raise item
            images, labels, ready = item
            ready.wait()
            yield images, labels
        worker.join()

    def __len__(self) -> int:
        return len(self.loader)


def evaluate_cases(model, store: VolumeStore, case_ids: list[str], cfg: dict, device: torch.device) -> list[dict]:
    rows = []
    infer_cfg = cfg["infer"]
    for case_id in case_ids:
        case = store.cases[case_id]
        prediction, window_info = predict_volume(
            model,
            case["image"],
            patch_size=[int(v) for v in cfg["data"]["patch_size"]],
            overlap=float(infer_cfg["overlap"]),
            sw_batch_size=int(infer_cfg["sw_batch_size"]),
            device=device,
            amp=bool(cfg["train"]["amp"]),
            sigma_scale=float(infer_cfg["gaussian_sigma_scale"]),
        )
        if tuple(prediction.shape) != tuple(case["label"].shape):
            raise RuntimeError(f"{case_id} prediction shape {prediction.shape} != {case['label'].shape}")
        metrics = binary_segmentation_metrics(
            prediction,
            case["label"],
            spacing=tuple(case["spacing_xyz"]),
        )
        metrics.update(
            {
                "case_id": case_id,
                "shape_xyz": case["shape_xyz"],
                "spacing_xyz": case["spacing_xyz"],
                "distance_unit": "voxel",
                "n_windows": window_info["n_windows"],
                "pad_after": window_info["pad_after"],
            }
        )
        rows.append(metrics)
    return rows


def _save_checkpoint(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def _plot_history(history: list[dict], path: Path) -> None:
    if not history:
        return
    epochs = [row["epoch"] for row in history]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(epochs, [row["loss"] for row in history], label="total")
    axes[0].plot(epochs, [row["ce_loss"] for row in history], label="ce")
    axes[0].plot(epochs, [row["dice_loss"] for row in history], label="dice")
    axes[0].set_xlabel("epoch")
    axes[0].set_ylabel("loss")
    axes[0].legend()
    val_epochs = [row["epoch"] for row in history if row.get("val_dice") is not None]
    val_dice = [row["val_dice"] for row in history if row.get("val_dice") is not None]
    if val_epochs:
        axes[1].plot(val_epochs, val_dice, marker="o")
    axes[1].set_xlabel("epoch")
    axes[1].set_ylabel("full-volume val Dice")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=120)
    plt.close(fig)


def train(cfg: dict, device: torch.device, resume: str | None) -> None:
    seed_everything(int(cfg["seed"]))
    out_dir = Path(cfg["output"]["dir"])
    out_dir.mkdir(parents=True, exist_ok=True)
    log_path = out_dir / "train.log"
    started = time.time()

    def log(message: str) -> None:
        line = f"[{time.strftime('%Y-%m-%d %H:%M:%S')}] {message}"
        print(line, flush=True)
        with log_path.open("a") as handle:
            handle.write(line + "\n")

    store = VolumeStore(cfg)
    store.prepare(retain="train_val", keep_raw=False)
    write_json(out_dir / "data_audit.json", store.audit_payload())
    write_json(out_dir / "split.json", store.split_payload())
    split_info = store.split_payload()
    if split_info["overlap"]["has_overlap"]:
        raise RuntimeError(f"split overlap: {split_info['overlap']}")
    log(
        f"cases train={len(store.train_ids)} val={len(store.val_ids)} test={len(store.test_ids)} "
        f"device={device} info={runtime_info(device)}"
    )
    write_json(out_dir / "runtime.json", runtime_info(device))
    write_json(out_dir / "config_snapshot.json", cfg)

    model = build_model(cfg).to(device)
    train_cfg = cfg["train"]
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(train_cfg["lr"]),
        weight_decay=float(train_cfg["weight_decay"]),
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=int(train_cfg["epochs"]),
        eta_min=float(train_cfg["eta_min"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(train_cfg["amp"]) and device.type == "cuda")
    dataset = PatchDataset(store, store.train_ids, training=True)
    loader = make_loader(dataset, cfg, shuffle=True)

    start_epoch = 0
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
        best_dice = float(checkpoint["best_val_dice"])
        stale_vals = int(checkpoint["stale_validations"])
        history = list(checkpoint["history"])
        log(f"resumed from {resume} at epoch {start_epoch} best_val_dice={best_dice:.4f}")

    loss_cfg = cfg["loss"]
    accum = int(train_cfg["grad_accum_steps"])
    epochs = int(train_cfg["epochs"])
    for epoch in range(start_epoch, epochs):
        dataset.set_epoch(epoch)
        model.train()
        optimizer.zero_grad(set_to_none=True)
        totals = []
        ces = []
        dices = []
        fg_fracs = []
        for step, (images, labels) in enumerate(_BatchPrefetcher(loader, device), start=1):
            with torch.autocast(device_type=device.type, enabled=bool(train_cfg["amp"]) and device.type == "cuda"):
                logits = model(images)
                if logits.shape[2:] != images.shape[2:]:
                    raise RuntimeError(f"logits {logits.shape} != input {images.shape}")
                total, ce, dice = ce_dice_loss(
                    logits,
                    labels,
                    ce_weight=float(loss_cfg["ce_weight"]),
                    dice_weight=float(loss_cfg["dice_weight"]),
                    dice_smooth=float(loss_cfg["dice_smooth"]),
                )
            scaler.scale(total / accum).backward()
            if step % accum == 0 or step == len(loader):
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), float(train_cfg["grad_clip"]))
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            totals.append(float(total.detach()))
            ces.append(float(ce.detach()))
            dices.append(float(dice.detach()))
            fg_fracs.append(float((labels == 1).float().mean()))
        scheduler.step()
        row = {
            "epoch": epoch + 1,
            "loss": float(np.mean(totals)),
            "ce_loss": float(np.mean(ces)),
            "dice_loss": float(np.mean(dices)),
            "fg_fraction": float(np.mean(fg_fracs)),
            "lr": float(optimizer.param_groups[0]["lr"]),
            "val_dice": None,
        }
        do_val = (epoch + 1) % int(train_cfg["val_interval"]) == 0
        if do_val:
            rows = evaluate_cases(model, store, store.val_ids, cfg, device)
            val_dice = float(np.mean([item["dice"] for item in rows]))
            row["val_dice"] = val_dice
            row["val_jaccard"] = float(np.mean([item["jaccard"] for item in rows]))
            n_empty = sum(item["empty_status"] == "pred_empty" for item in rows)
            log(
                f"epoch {epoch + 1} loss={row['loss']:.4f} ce={row['ce_loss']:.4f} "
                f"dice_loss={row['dice_loss']:.4f} fg={row['fg_fraction']:.3f} "
                f"val_dice={val_dice:.4f} empty_preds={n_empty}/{len(rows)}"
            )
            improved = val_dice > best_dice
            if improved:
                best_dice = val_dice
                stale_vals = 0
                _save_checkpoint(
                    out_dir / "best.pt",
                    _checkpoint_payload(
                        model, optimizer, scheduler, scaler, epoch + 1, best_dice, stale_vals, history + [row], cfg
                    ),
                )
                write_json(out_dir / "val_best_metrics.json", {"mean_dice": best_dice, "cases": rows})
            else:
                stale_vals += 1
        else:
            log(
                f"epoch {epoch + 1} loss={row['loss']:.4f} ce={row['ce_loss']:.4f} "
                f"dice_loss={row['dice_loss']:.4f} fg={row['fg_fraction']:.3f}"
            )
        history.append(row)
        _write_history(out_dir, history)
        _plot_history(history, out_dir / "figures" / "loss_val_dice.png")
        _save_checkpoint(
            out_dir / "last.pt",
            _checkpoint_payload(model, optimizer, scheduler, scaler, epoch + 1, best_dice, stale_vals, history, cfg),
        )
        if do_val and stale_vals >= int(train_cfg["early_stopping_patience"]):
            log(f"early stop at epoch {epoch + 1}: {stale_vals} validations without improvement")
            break
    elapsed = time.time() - started
    write_json(
        out_dir / "train_finished.json",
        {"seconds": elapsed, "best_val_dice": best_dice, "epochs_ran": history[-1]["epoch"] if history else 0},
    )
    log(f"finished in {elapsed / 3600:.2f} h, best full-volume val Dice {best_dice:.4f}")


def _checkpoint_payload(model, optimizer, scheduler, scaler, epoch, best_val_dice, stale_vals, history, cfg):
    return {
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "scheduler": scheduler.state_dict(),
        "scaler": scaler.state_dict(),
        "epoch": epoch,
        "best_val_dice": best_val_dice,
        "stale_validations": stale_vals,
        "history": history,
        "selected_on": "val_full_volume_mean_dice",
        "config": cfg,
    }


def _write_history(out_dir: Path, history: list[dict]) -> None:
    write_json(out_dir / "history.json", {"history": history})
    fields = ["epoch", "loss", "ce_loss", "dice_loss", "fg_fraction", "lr", "val_dice", "val_jaccard"]
    with (out_dir / "history.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in history:
            writer.writerow({key: row.get(key) for key in fields})
