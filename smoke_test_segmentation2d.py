#!/usr/bin/env python
"""Forward/backward, Z-index, and volume-shape checks. Does not train."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from la_seg2d.data import Z_AXIS, foreground_slice_indices, open_store, pad_slice, sample_slice_crop
from la_seg2d.engine import build_model
from la_seg2d.infer import predict_slice_sliding, predict_slices_batched, predict_volume_from_slices
from la_seg2d.losses import ce_dice_loss
from la_seg2d.model import count_parameters
from la_seg2d.protocol import build_protocol
from la_seg3d.model import UNet3D
from la_seg3d.utils import load_config


def main() -> None:
    cfg = load_config(ROOT / "configs/segmentation2d.yaml")
    protocol = build_protocol(cfg, ROOT)
    if protocol["unknown_or_not_stored"]:
        raise RuntimeError(protocol["unknown_or_not_stored"])
    if not protocol["budget_alignment"]["matched"]:
        raise RuntimeError(protocol["budget_alignment"])
    split = json.loads((ROOT / cfg["reference_3d"]["split"]).read_text())
    store = open_store(cfg, retain="train_val", keep_raw=False)
    for key in ("train_ids", "val_ids", "test_ids"):
        if list(getattr(store, key)) != list(split[key]):
            raise RuntimeError(f"split mismatch on {key}")

    case_id = store.train_ids[0]
    case = store.cases[case_id]
    image, label = case["image"], case["label"]
    if image.shape[Z_AXIS] != case["shape_xyz"][Z_AXIS]:
        raise RuntimeError("Z axis is not shape_xyz[2]")
    stacked = np.stack([image[:, :, index] for index in range(image.shape[Z_AXIS])], axis=2)
    if not np.array_equal(stacked, image):
        raise RuntimeError("stacking Z slices did not restore the volume")
    fg = foreground_slice_indices(label)
    rng = np.random.RandomState(0)
    crop, crop_label, slice_index, used_fg, background = sample_slice_crop(
        image,
        label,
        fg,
        tuple(cfg["data"]["crop_size_xy"]),
        float(cfg["data"]["fg_sample_prob"]),
        tuple(cfg["data"]["fg_center_jitter_xy"]),
        rng,
    )
    if crop.shape != tuple(cfg["data"]["crop_size_xy"]):
        raise RuntimeError(f"crop {crop.shape}")
    if not (0 <= slice_index < image.shape[Z_AXIS]):
        raise RuntimeError("slice_index out of range")
    small = np.zeros((40, 30), dtype=np.float32)
    small_label = np.zeros((40, 30), dtype=np.uint8)
    padded_image, padded_label, pad = pad_slice(small, small_label, (112, 112))
    if padded_image.shape != (112, 112) or padded_label.shape != (112, 112) or pad != [72, 82]:
        raise RuntimeError(f"padding failed {padded_image.shape} {pad}")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    model.train()
    for shape in ((2, 1, 112, 112), (1, 1, 111, 97), (1, 1, 113, 113)):
        batch = torch.randn(*shape, device=device)
        target = torch.randint(0, 2, shape[:1] + shape[2:], device=device)
        logits = model(batch)
        if tuple(logits.shape) != (shape[0], 2, shape[2], shape[3]):
            raise RuntimeError(f"{shape} -> {tuple(logits.shape)}")
        loss, ce, dice = ce_dice_loss(logits, target, 1.0, 1.0, 1.0)
        loss.backward()
        model.zero_grad(set_to_none=True)
    model.eval()
    synthetic = np.random.randn(180, 200).astype(np.float32)
    pred, info = predict_slice_sliding(
        model,
        synthetic,
        (112, 112),
        0.5,
        4,
        device,
        amp=False,
        sigma_scale=0.125,
    )
    if pred.shape != synthetic.shape:
        raise RuntimeError(f"sliding window {pred.shape} != {synthetic.shape}")
    volume = np.stack([synthetic, synthetic[::-1]], axis=2)
    tiny_cfg_side = 64
    restored, vol_info = predict_volume_from_slices(
        model,
        volume,
        z_axis=2,
        crop_size=(112, 112),
        full_slice_max_side=tiny_cfg_side,
        overlap=0.5,
        sw_batch_size=4,
        slice_batch_size=2,
        device=device,
        amp=False,
        sigma_scale=0.125,
    )
    if restored.shape != volume.shape or vol_info["mode"] != "sliding_window":
        raise RuntimeError(f"volume restore {restored.shape} {vol_info}")
    full, full_info = predict_volume_from_slices(
        model,
        image[:96, :80, :2],
        z_axis=2,
        crop_size=(112, 112),
        full_slice_max_side=112,
        overlap=0.5,
        sw_batch_size=4,
        slice_batch_size=2,
        device=device,
        amp=False,
        sigma_scale=0.125,
    )
    if full.shape != (96, 80, 2) or full_info["mode"] != "full_slice":
        raise RuntimeError(f"full-slice path {full.shape} {full_info}")
    plane = image[:, :, int(fg[0]) if len(fg) else 0]
    pred_plane = predict_slices_batched(model, plane[None].astype(np.float32), device, amp=False)
    if pred_plane.shape != (1,) + plane.shape:
        raise RuntimeError(f"full axial slice {pred_plane.shape} != {(1,) + plane.shape}")

    n_background = 0
    for index in range(300):
        _c, _l, _z, _fg, is_background = sample_slice_crop(
            image,
            label,
            fg,
            tuple(cfg["data"]["crop_size_xy"]),
            float(cfg["data"]["fg_sample_prob"]),
            tuple(cfg["data"]["fg_center_jitter_xy"]),
            np.random.RandomState(index),
        )
        n_background += int(is_background)
    if len(fg) < image.shape[Z_AXIS] and n_background == 0:
        raise RuntimeError("foreground sampling excluded every background slice")
    unet3d = UNet3D(1, 2, [16, 32, 64, 128, 256], "instance")
    print(
        json.dumps(
            {
                "case_id": case_id,
                "shape_xyz": list(case["shape_xyz"]),
                "slice_index": slice_index,
                "used_foreground_sampler": used_fg,
                "background_slice": background,
                "crop_shape": list(crop.shape),
                "unet2d_parameters": count_parameters(model),
                "unet3d_parameters": count_parameters(unet3d),
                "background_samples_in_300": n_background,
                "optimizer_updates_3d": protocol["unet3d"]["budget"]["optimizer_updates"],
                "max_steps_2d": protocol["unet2d"]["max_steps"],
                "sliding_windows": info["n_windows"],
                "split_counts": split["counts"],
            }
        ),
        flush=True,
    )
    print("smoke test passed", flush=True)


if __name__ == "__main__":
    main()
