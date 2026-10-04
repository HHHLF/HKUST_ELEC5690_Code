#!/usr/bin/env python
"""Full-volume 2D evaluation. Test loads the validation-selected best.pt."""

from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import nibabel as nib
import numpy as np
import torch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from la_seg2d.data import open_store
from la_seg2d.engine import build_model
from la_seg2d.infer import predict_volume_from_slices
from la_seg3d.metrics import binary_segmentation_metrics
from la_seg3d.utils import load_config, pick_device, summarize_cases, write_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/segmentation2d.yaml")
    parser.add_argument("--split", choices=["test", "val"], default="test")
    parser.add_argument("--checkpoint", default=None)
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    if cfg["infer"]["postprocess"] != "none":
        raise RuntimeError("postprocess must be none")
    device = pick_device(cfg)
    out_root = ROOT / cfg["output"]["dir"]
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else out_root / "best.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(checkpoint_path)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    if checkpoint.get("selected_on") != "val_full_volume_mean_dice":
        raise RuntimeError(f"{checkpoint_path} was not selected on full-volume validation Dice")

    store = open_store(cfg, retain="test" if args.split == "test" else "train_val", keep_raw=True)
    case_ids = store.test_ids if args.split == "test" else store.val_ids
    model = build_model(cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()

    save_dir = out_root / args.split
    pred_dir = save_dir / "predictions"
    nifti_dir = save_dir / "nifti"
    for folder in (pred_dir, nifti_dir):
        folder.mkdir(parents=True, exist_ok=True)

    rows = []
    crop = tuple(int(v) for v in cfg["data"]["crop_size_xy"])
    for case_id in case_ids:
        case = store.cases[case_id]
        prediction, window_info = predict_volume_from_slices(
            model,
            case["image"],
            z_axis=int(cfg["data"]["z_axis"]),
            crop_size=crop,
            full_slice_max_side=int(cfg["infer"]["full_slice_max_side"]),
            overlap=float(cfg["infer"]["overlap"]),
            sw_batch_size=int(cfg["infer"]["sw_batch_size"]),
            slice_batch_size=int(cfg["infer"]["slice_batch_size"]),
            device=device,
            amp=bool(cfg["train"]["amp"]),
            sigma_scale=float(cfg["infer"]["gaussian_sigma_scale"]),
        )
        if tuple(prediction.shape) != tuple(case["label"].shape):
            raise RuntimeError(f"{case_id} shape was not restored: {prediction.shape} vs {case['label'].shape}")
        row = binary_segmentation_metrics(prediction, case["label"], tuple(case["spacing_xyz"]))
        row.update(
            {
                "case_id": case_id,
                "shape_xyz": case["shape_xyz"],
                "spacing_xyz": case["spacing_xyz"],
                "distance_unit": "voxel",
                "infer_mode": window_info["mode"],
                "n_slices": window_info["n_slices"],
                "n_windows": window_info["n_windows"],
                "pad_after_xy": window_info["pad_after_xy"],
                "z_axis": window_info["z_axis"],
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
        print(
            f"{case_id} dice={row['dice']:.4f} jaccard={row['jaccard']:.4f} "
            f"asd={row['asd']:.4f} hd95={row['hd95']:.4f} mode={row['infer_mode']}",
            flush=True,
        )

    expected = list(case_ids)
    got = [row["case_id"] for row in rows]
    if got != expected:
        raise RuntimeError("evaluation dropped or reordered cases")
    summary = summarize_cases(rows)
    summary.update(
        {
            "split": args.split,
            "checkpoint": str(checkpoint_path),
            "checkpoint_epoch": checkpoint.get("epoch"),
            "checkpoint_step": checkpoint.get("global_step"),
            "best_val_dice_recorded_in_checkpoint": checkpoint.get("best_val_dice"),
            "postprocess": "none",
            "distance_unit": "voxel",
            "spacing_source": "NRRD space directions; no space units in the files",
            "metrics_implementation": "la_seg3d.metrics.binary_segmentation_metrics",
        }
    )
    write_json(save_dir / "per_case_metrics.json", rows)
    write_json(save_dir / "summary.json", summary)
    _write_table(save_dir / "per_case_metrics.csv", rows)
    _write_summary_csv(save_dir / "summary.csv", summary)
    print(
        f"n={summary['n_cases']} mean_dice={summary['mean_dice']} "
        f"n_pred_empty={summary['n_pred_empty']} n_nonfinite_hd95={summary['hd95']['n_nonfinite']}",
        flush=True,
    )


def _write_nifti(path: Path, volume: np.ndarray, spacing: list[float]) -> None:
    affine = np.eye(4, dtype=np.float64)
    affine[0, 0], affine[1, 1], affine[2, 2] = spacing
    image = nib.Nifti1Image(np.ascontiguousarray(volume), affine)
    image.header["descrip"] = "axis XYZ; spacing unit voxel, not mm"
    nib.save(image, str(path))


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
        "infer_mode",
        "n_slices",
    ]
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row[key] for key in fields})


def _write_summary_csv(path: Path, summary: dict) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["metric", "mean", "std", "n_nonfinite", "finite_mean", "n_cases"])
        for key in ("dice", "jaccard", "asd", "hd95"):
            item = summary[key]
            writer.writerow([key, item["mean"], item["std"], item["n_nonfinite"], item["finite_mean"], summary["n_cases"]])
        writer.writerow(["n_cases", summary["n_cases"], "", "", "", ""])
        writer.writerow(["n_pred_empty", summary["n_pred_empty"], "", "", "", ""])
        writer.writerow(["n_gt_empty", summary["n_gt_empty"], "", "", "", ""])
        writer.writerow(["n_both_empty", summary["n_both_empty"], "", "", "", ""])
        writer.writerow(["n_failed_cases", summary["n_failed_cases"], "", "", "", ""])


if __name__ == "__main__":
    main()
