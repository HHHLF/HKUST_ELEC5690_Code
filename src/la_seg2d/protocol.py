"""Read the finished 3D run and describe what the 2D run is allowed to change.

Numbers that are not stored in the 3D artifacts are computed from those
artifacts and marked as computed. Nothing is filled in from a default guess.
"""

from __future__ import annotations

import json
import math
from pathlib import Path

from la_seg2d.data import AXIS_NOTE, Z_AXIS
from la_seg2d.model import UNet2D, count_parameters
from la_seg3d.model import UNet3D


def _read(path: Path):
    return json.loads(path.read_text())


def three_d_optimizer_budget(snapshot: dict, split: dict, finished: dict) -> dict:
    n_train = int(split["counts"]["train"])
    patches = int(snapshot["data"]["patches_per_case_per_epoch"])
    batch = int(snapshot["train"]["batch_size"])
    accum = int(snapshot["train"]["grad_accum_steps"])
    epochs = int(finished["epochs_ran"])
    samples_per_epoch = n_train * patches
    batches_per_epoch = math.ceil(samples_per_epoch / batch)
    updates_per_epoch = math.ceil(batches_per_epoch / accum)
    return {
        "recorded_directly": False,
        "source": (
            "Computed from outputs/segmentation3d/config_snapshot.json, split.json counts, "
            "and train_finished.json epochs_ran. src/la_seg3d/engine.py uses drop_last=False "
            "and one optimizer step per grad_accum_steps batches, including a short last batch. "
            "history.csv has one row per epoch and no optimizer-step column."
        ),
        "n_train_cases": n_train,
        "patches_per_case_per_epoch": patches,
        "batch_size": batch,
        "grad_accum_steps": accum,
        "samples_per_epoch": samples_per_epoch,
        "batches_per_epoch": batches_per_epoch,
        "optimizer_updates_per_epoch": updates_per_epoch,
        "epochs_ran": epochs,
        "optimizer_updates": epochs * updates_per_epoch,
        "samples_drawn": epochs * samples_per_epoch,
        "best_checkpoint_epoch": None,
        "optimizer_updates_at_best_checkpoint": None,
        "wall_seconds": finished.get("seconds"),
        "best_val_dice": finished.get("best_val_dice"),
        "early_stopping_triggered": epochs != int(snapshot["train"]["epochs"]),
    }


def build_protocol(cfg: dict, root: Path, parameter_counts: dict | None = None) -> dict:
    ref = cfg["reference_3d"]
    snapshot = _read(root / ref["config_snapshot"])
    split = _read(root / ref["split"])
    finished = _read(root / ref["train_finished"])
    summary = _read(root / ref["test_summary"])
    audit = _read(root / ref["data_audit"])
    selection = _read(root / ref["problem2b_selection"])
    budget = three_d_optimizer_budget(snapshot, split, finished)
    best_epoch = summary.get("checkpoint_epoch")
    if best_epoch is not None:
        budget["best_checkpoint_epoch"] = int(best_epoch)
        budget["optimizer_updates_at_best_checkpoint"] = int(best_epoch) * budget["optimizer_updates_per_epoch"]

    if parameter_counts is None:
        parameter_counts = {
            "unet3d": count_parameters(
                UNet3D(
                    int(snapshot["model"]["in_channels"]),
                    int(snapshot["model"]["out_channels"]),
                    [int(v) for v in snapshot["model"]["channels"]],
                    snapshot["model"]["norm"],
                )
            ),
            "unet2d": count_parameters(
                UNet2D(
                    int(cfg["model"]["in_channels"]),
                    int(cfg["model"]["out_channels"]),
                    [int(v) for v in cfg["model"]["channels"]],
                    cfg["model"]["norm"],
                )
            ),
        }

    unknown = []
    if audit.get("h5_file_count", 0) != 0:
        unknown.append("HDF5 files were reported; the 2D loader does not implement a second axis convention.")
    if summary.get("distance_unit") != "voxel":
        unknown.append(f"3D summary distance_unit is {summary.get('distance_unit')}")
    if audit.get("axis_order", {}).get("array_and_model_axes") is None:
        unknown.append("3D data_audit.json has no axis_order.array_and_model_axes")

    return {
        "purpose": "Problem 2(d) 2D U-Net versus the finished Problem 2(a) 3D U-Net.",
        "unknown_or_not_stored": unknown,
        "distance_unit": {
            "unit": summary.get("distance_unit"),
            "source": summary.get("spacing_source"),
            "audit_note": audit.get("spacing_note"),
            "mm_not_used": True,
        },
        "axis": AXIS_NOTE,
        "z_axis_index": Z_AXIS,
        "split": {
            "source": ref["split"],
            "seed": split["seed"],
            "val_ratio": split["val_ratio"],
            "counts": split["counts"],
            "policy": split["policy"],
            "overlap": split["overlap"],
            "same_case_ids_required": True,
        },
        "shared": {
            "seed": snapshot["seed"],
            "normalization": audit.get("normalization"),
            "resample": audit.get("resample"),
            "loss": snapshot["loss"],
            "dice_reduction": "per-sample soft foreground Dice, then batch mean; CE is mean CrossEntropyLoss",
            "optimizer": snapshot["train"]["optimizer"],
            "lr": snapshot["train"]["lr"],
            "weight_decay": snapshot["train"]["weight_decay"],
            "scheduler": snapshot["train"]["scheduler"],
            "eta_min": snapshot["train"]["eta_min"],
            "scheduler_t_max_steps_of_scheduler": snapshot["train"]["epochs"],
            "amp": snapshot["train"]["amp"],
            "grad_clip": snapshot["train"]["grad_clip"],
            "checkpoint_rule": "strict improvement of mean full-volume validation Dice; test set is not used",
            "postprocess": snapshot["infer"]["postprocess"],
            "metrics": "la_seg3d.metrics.binary_segmentation_metrics",
            "surface_definition": "medpy-style 6-neighborhood erosion surface; ASD mean and HD95 of the concatenated directed distances",
            "empty_mask_rule": "both empty -> dice/jaccard 1 and distances 0; one empty -> dice/jaccard 0 and distances inf",
            "aggregation": "per-case metrics, then equal-weight mean; non-finite distances make the official mean inf",
        },
        "unet3d": {
            "config_snapshot": ref["config_snapshot"],
            "best_checkpoint": str(Path(snapshot["output"]["dir"]) / "best.pt"),
            "channels": snapshot["model"]["channels"],
            "norm": snapshot["model"]["norm"],
            "parameters": parameter_counts["unet3d"],
            "patch_size_xyz": snapshot["data"]["patch_size"],
            "fg_sample_prob": snapshot["data"]["fg_sample_prob"],
            "fg_center_jitter_xyz": snapshot["data"]["fg_center_jitter"],
            "augment": snapshot["augment"],
            "batch_size": snapshot["train"]["batch_size"],
            "epochs_configured": snapshot["train"]["epochs"],
            "val_interval_epochs": snapshot["train"]["val_interval"],
            "early_stopping_patience_validations": snapshot["train"]["early_stopping_patience"],
            "infer": snapshot["infer"],
            "budget": budget,
            "test_summary_file": ref["test_summary"],
            "test_n_cases": summary.get("n_cases"),
            "test_checkpoint_epoch": summary.get("checkpoint_epoch"),
        },
        "unet2d": {
            "input": "[B,1,H,W] one axial slice; no neighbor stack",
            "output": "[B,2,H,W]",
            "channels": cfg["model"]["channels"],
            "norm": cfg["model"]["norm"],
            "parameters": parameter_counts["unet2d"],
            "crop_size_xy": cfg["data"]["crop_size_xy"],
            "crop_matches_3d_patch_xy": list(cfg["data"]["crop_size_xy"]) == list(snapshot["data"]["patch_size"][:2]),
            "fg_sample_prob": cfg["data"]["fg_sample_prob"],
            "fg_center_jitter_xy": cfg["data"]["fg_center_jitter_xy"],
            "background_slice_policy": (
                "The non-foreground branch draws the Z index uniformly from all slices of the case, "
                "so pure-background slices stay in the training mixture. Validation and test predict every Z slice."
            ),
            "augment": cfg["augment"],
            "augment_difference": (
                "In-plane flips and rot90 follow the 3D XY augmentation. Intensity scale and noise use the same "
                "probabilities and ranges. The 3D flip along Z is not applied, because one axial slice has no Z extent. "
                "No spatial rescaling."
            ),
            "batch_size": cfg["train"]["batch_size"],
            "samples_per_epoch": cfg["train"]["samples_per_epoch"],
            "max_steps": cfg["train"]["max_steps"],
            "val_every_steps": cfg["train"]["val_every_steps"],
            "scheduler_every_steps": cfg["train"]["scheduler_every_steps"],
            "epoch_meaning": (
                "One 2D epoch draws samples_per_epoch crops. One 3D epoch draws "
                "n_train * patches_per_case_per_epoch patches. Equal epoch counts are not equal budgets."
            ),
            "infer": cfg["infer"],
            "cache": {
                "mode": cfg["data"]["cache_mode"],
                "dir": cfg["data"]["cache_dir"],
                "note": "Reuses the 3D per-case numpy cache in memory. Training does not re-read NRRD each epoch.",
            },
        },
        "budget_alignment": {
            "target": "2D max_steps equals the computed 3D optimizer update count",
            "unet3d_optimizer_updates": budget["optimizer_updates"],
            "unet2d_max_steps": int(cfg["train"]["max_steps"]),
            "matched": int(cfg["train"]["max_steps"]) == budget["optimizer_updates"],
            "not_the_same_compute": (
                "Matching optimizer updates does not match FLOPs. A 2D crop is one XY plane; "
                "a 3D patch is 112x112x80. Batch sizes also differ."
            ),
        },
        "visualization_cases": {
            "source": ref["problem2b_selection"],
            "slice_axis": selection.get("slice_axis", "Z"),
            "cases": [
                {"role": row["role"], "case_id": row["case_id"]}
                for row in selection["cases"]
            ],
            "slice_indices_file": ref["problem2b_metrics"],
        },
    }
