#!/usr/bin/env python
"""Full-volume evaluation. The test entry loads the validation-selected best.pt."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from la_seg3d.data import VolumeStore
from la_seg3d.engine import build_model
from la_seg3d.infer import predict_volume
from la_seg3d.metrics import binary_segmentation_metrics
from la_seg3d.utils import load_config, pick_device, summarize_cases, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_config(args.config)
    if cfg["infer"]["postprocess"] != "none":
        raise RuntimeError("postprocess is disabled for the reported metrics; set infer.postprocess: none")
    device = pick_device(cfg)
    out_root = Path(cfg["output"]["dir"])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else out_root / "best.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("selected_on") != "val_full_volume_mean_dice":
        raise RuntimeError(f"{checkpoint_path} was not selected on full-volume validation Dice")
    if args.split == "test" and checkpoint_path.name != "best.pt" and args.checkpoint is None:
        raise RuntimeError("test evaluation must load best.pt")

    store = VolumeStore(cfg)
    retain = "test" if args.split == "test" else "train_val"
    store.prepare(retain=retain, keep_raw=True)
    case_ids = store.test_ids if args.split == "test" else store.val_ids
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    save_dir = out_root / args.split
    pred_dir = save_dir / "predictions"
    preview_dir = save_dir / "previews"
    nifti_dir = save_dir / "nifti"
    for folder in (pred_dir, preview_dir, nifti_dir):
        folder.mkdir(parents=True, exist_ok=True)

    rows = []
    patch_size = [int(v) for v in cfg["data"]["patch_size"]]
    for case_id in case_ids:
        case = store.cases[case_id]
        prediction, window_info = predict_volume(
            model,
            case["image"],
            patch_size=patch_size,
            overlap=float(cfg["infer"]["overlap"]),
            sw_batch_size=int(cfg["infer"]["sw_batch_size"]),
            device=device,
            amp=bool(cfg["train"]["amp"]),
            sigma_scale=float(cfg["infer"]["gaussian_sigma_scale"]),
        )
        if tuple(prediction.shape) != tuple(case["label"].shape):
            raise RuntimeError(f"{case_id} shape was not restored")
        row = binary_segmentation_metrics(prediction, case["label"], tuple(case["spacing_xyz"]))
        row.update(
            {
                "case_id": case_id,
                "shape_xyz": case["shape_xyz"],
                "spacing_xyz": case["spacing_xyz"],
                "distance_unit": "voxel",
                "n_windows": window_info["n_windows"],
                "pad_after": window_info["pad_after"],
                "pad_restore": window_info["pad_mode"],
            }
        )
        rows.append(row)
        np.savez_compressed(
            pred_dir / f"{case_id}.npz",
            prediction=prediction,
            image=case["image_raw"],
            label=case["label"],
            spacing_xyz=np.array(case["spacing_xyz"], dtype=np.float32),
            shape_xyz=np.array(case["shape_xyz"], dtype=np.int32),
            axis_order=np.array(["X", "Y", "Z"]),
            distance_unit=np.array("voxel"),
        )
        _write_nifti(nifti_dir / f"{case_id}_pred.nii.gz", prediction, case["spacing_xyz"])
        _write_nifti(nifti_dir / f"{case_id}_gt.nii.gz", case["label"], case["spacing_xyz"])
        _write_nifti(nifti_dir / f"{case_id}_mri.nii.gz", case["image_raw"], case["spacing_xyz"])
        _write_preview(preview_dir / f"{case_id}.png", case["image_raw"], case["label"], prediction, row)
        print(
            f"{case_id} dice={row['dice']:.4f} jaccard={row['jaccard']:.4f} "
            f"asd={row['asd']:.4f} hd95={row['hd95']:.4f}",
            flush=True,
        )

    summary = summarize_cases(rows)
    summary.update(
        {
            "split": args.split,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "best_val_dice_recorded_in_checkpoint": checkpoint.get("best_val_dice"),
            "postprocess": "none",
            "distance_unit": "voxel",
            "spacing_source": "NRRD space directions; no space units in the files",
        }
    )
    write_json(save_dir / "per_case_metrics.json", rows)
    write_json(save_dir / "summary.json", summary)
    _write_table(save_dir / "per_case_metrics.csv", rows)
    _write_summary_csv(save_dir / "summary.csv", summary)
    print(
        f"n={summary['n_cases']} mean_dice={summary['mean_dice']:.4f} "
        f"exceeds_0.70={summary['mean_dice_exceeds_0.70']}",
        flush=True,
    )


def _write_nifti(path: Path, volume: np.ndarray, spacing: list[float]) -> None:
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0], affine[1, 1], affine[2, 2] = spacing
    image = nib.Nifti1Image(np.ascontiguousarray(volume), affine)
    image.header["descrip"] = "axis XYZ; spacing unit voxel, not mm"
    nib.save(image, str(path))


def _write_preview(path: Path, image: np.ndarray, label: np.ndarray, prediction: np.ndarray, row: dict) -> None:
    axis_sum = label.sum(axis=(0, 1))
    index = int(np.argmax(axis_sum))
    fig, axes = plt.subplots(1, 3, figsize=(9, 3))
    panels = [
        (image[:, :, index], "MRI"),
        (label[:, :, index], "ground truth"),
        (prediction[:, :, index], "prediction"),
    ]
    for axis, (panel, title) in zip(axes, panels):
        axis.imshow(panel.T, cmap="gray", origin="lower")
        axis.set_title(title)
        axis.axis("off")
    fig.suptitle(
        f"{row['case_id']} z={index}  Dice={row['dice']:.3f}  HD95={row['hd95']:.2f} voxel"
    )
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def _write_table(path: Path, rows: list[dict]) -> None:
    fields = [
        "case_id",
        "dice",
        "jaccard",
        "asd",
        "hd95",
        "shape_xyz",
        "spacing_xyz",
        "distance_unit",
        "pred_foreground_voxels",
        "gt_foreground_voxels",
        "empty_status",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def _write_summary_csv(path: Path, summary: dict) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std", "n_nonfinite", "finite_mean"])
        for key in ("dice", "jaccard", "asd", "hd95"):
            item = summary[key]
            writer.writerow([key, item["mean"], item["std"], item["n_nonfinite"], item["finite_mean"]])
        writer.writerow(["n_cases", summary["n_cases"], "", "", ""])
        writer.writerow(["mean_dice_exceeds_0.70", summary["mean_dice_exceeds_0.70"], "", "", ""])
        writer.writerow(["n_pred_empty", summary["n_pred_empty"], "", "", ""])
        writer.writerow(["n_gt_empty", summary["n_gt_empty"], "", "", ""])
        writer.writerow(["n_both_empty", summary["n_both_empty"], "", "", ""])


if __name__ == "__main__":
    main()
