#!/usr/bin/env python
"""Correctness checks for the 3D U-Net, sliding window, and surface metrics."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from la_seg3d.data import augment_patch, map_label, nonzero_zscore, sample_patch
from la_seg3d.infer import predict_volume
from la_seg3d.losses import ce_dice_loss
from la_seg3d.metrics import binary_segmentation_metrics
from la_seg3d.model import UNet3D
from la_seg3d.nrrd_io import read_nrrd
from la_seg3d.utils import load_config, pick_device


def check_metrics() -> None:
    spacing = (1.0, 1.0, 1.0)
    mask = np.zeros((24, 20, 16), dtype=np.uint8)
    mask[4:18, 4:16, 3:13] = 1
    same = binary_segmentation_metrics(mask, mask, spacing)
    assert same["dice"] == 1.0 and same["jaccard"] == 1.0
    assert same["asd"] == 0.0 and same["hd95"] == 0.0, same
    empty = np.zeros_like(mask)
    both = binary_segmentation_metrics(empty, empty, spacing)
    assert both["dice"] == 1.0 and both["jaccard"] == 1.0 and both["asd"] == 0.0 and both["hd95"] == 0.0
    one = binary_segmentation_metrics(empty, mask, spacing)
    assert one["dice"] == 0.0 and one["jaccard"] == 0.0
    assert np.isinf(one["asd"]) and np.isinf(one["hd95"])
    shifted = np.roll(mask, 1, axis=0)
    moved = binary_segmentation_metrics(shifted, mask, spacing)
    assert 0.0 < moved["dice"] < 1.0
    assert moved["asd"] > 0.0 and moved["hd95"] > 0.0
    print("metrics ok", {k: same[k] for k in ("dice", "jaccard", "asd", "hd95")})


def check_model_and_window(device: torch.device) -> None:
    model = UNet3D(1, 2, [8, 8, 8, 8, 8], norm="instance").to(device)
    odd = torch.randn(2, 1, 48, 50, 36, device=device)
    logits = model(odd)
    assert logits.shape == (2, 2, 48, 50, 36), logits.shape
    loss, ce, dice = ce_dice_loss(logits, torch.zeros(2, 48, 50, 36, dtype=torch.long, device=device), 1.0, 1.0, 1.0)
    loss.backward()
    assert all(p.grad is not None for p in model.parameters() if p.requires_grad)
    print(f"forward/backward ok loss={float(loss):.4f} ce={float(ce):.4f} dice={float(dice):.4f}")

    volume = np.random.randn(40, 50, 30).astype(np.float32)
    pred, info = predict_volume(
        model, volume, [48, 48, 40], overlap=0.5, sw_batch_size=1, device=device, amp=False, sigma_scale=0.125
    )
    assert pred.shape == volume.shape, (pred.shape, info)
    assert info["pad_after"] == [8, 0, 10]
    full = np.random.randn(64, 64, 48).astype(np.float32)
    pred_full, info_full = predict_volume(
        model, full, [32, 32, 32], overlap=0.5, sw_batch_size=2, device=device, amp=device.type == "cuda", sigma_scale=0.125
    )
    assert pred_full.shape == full.shape and info_full["pad_after"] == [0, 0, 0]
    print("sliding window restored", pred.shape, "and", pred_full.shape, "windows", info_full["n_windows"])


def check_real_case(cfg: dict) -> None:
    root = Path(cfg["data"]["root"]) / cfg["data"]["train_dirname"]
    case = next(path for path in sorted(root.iterdir()) if (path / "lgemri.nrrd").is_file())
    image, spacing, _meta = read_nrrd(case / "lgemri.nrrd")
    label, label_spacing, _ = read_nrrd(case / "laendo.nrrd")
    assert image.shape == label.shape
    assert spacing == label_spacing == (1.0, 1.0, 1.0)
    label01 = map_label(label, case.name)
    assert set(np.unique(label01).tolist()) <= {0, 1}
    norm = nonzero_zscore(image)
    assert np.isfinite(norm).all()
    rng = np.random.RandomState(0)
    patch_size = tuple(cfg["data"]["patch_size"])
    img_p, lab_p = sample_patch(norm, label01, patch_size, 1.0, (0, 0, 0), rng)
    assert img_p.shape == patch_size and lab_p.sum() > 0
    img_a, lab_a = augment_patch(img_p, lab_p, cfg["augment"], rng)
    assert img_a.shape == patch_size and lab_a.shape == patch_size
    print("real case", case.name, "shape", image.shape, "fg", int(label01.sum()), "spacing", spacing)


def check_loss_decreases(device: torch.device) -> None:
    model = UNet3D(1, 2, [8, 16, 16, 16, 16], norm="instance").to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    image = torch.zeros(2, 1, 48, 48, 32, device=device)
    label = torch.zeros(2, 48, 48, 32, dtype=torch.long, device=device)
    label[:, 12:36, 12:36, 8:24] = 1
    image[:, :, 12:36, 12:36, 8:24] = 1
    first = None
    last = None
    for _ in range(15):
        optimizer.zero_grad(set_to_none=True)
        total, _, _ = ce_dice_loss(model(image), label, 1.0, 1.0, 1.0)
        total.backward()
        optimizer.step()
        last = float(total)
        first = last if first is None else first
    assert last < first, (first, last)
    print(f"overfit loss {first:.4f} -> {last:.4f}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = pick_device(cfg)
    check_metrics()
    check_model_and_window(device)
    check_real_case(cfg)
    check_loss_decreases(device)
    print("smoke test passed")


if __name__ == "__main__":
    main()
