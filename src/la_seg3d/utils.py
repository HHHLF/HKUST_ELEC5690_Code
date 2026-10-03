"""Config, seeding, devices, and metric summaries."""

from __future__ import annotations

import json
import platform
import random
from pathlib import Path

import numpy as np
import torch
import yaml


def load_config(path: str | Path) -> dict:
    with Path(path).open() as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return cfg


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # The seed fixes the split, initialization, and patch sampling.
    # cuDNN benchmark is enabled so 3D convolutions use a fast kernel.
    torch.backends.cudnn.benchmark = True


def pick_device(cfg: dict) -> torch.device:
    want = cfg.get("device", "cuda")
    if want == "cuda" and torch.cuda.is_available():
        return torch.device("cuda")
    return torch.device("cpu")


def runtime_info(device: torch.device) -> dict:
    info = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "torch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
        "device": str(device),
        "gpu_count_visible": torch.cuda.device_count() if torch.cuda.is_available() else 0,
    }
    if device.type == "cuda":
        info["gpu_name"] = torch.cuda.get_device_name(device)
    return info


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


def summarize_cases(rows: list[dict]) -> dict:
    """Equal-weight mean over cases. Non-finite distances are kept.

    If any ASD or HD95 is non-finite, that metric's official mean is inf.
    `finite_mean` is reported beside it and is not a replacement.
    """
    summary = {
        "n_cases": len(rows),
        "distance_mean_policy": (
            "Metrics are computed per case, then averaged with equal case weight. "
            "Voxels are not pooled across cases. Non-finite ASD/HD95 stay in the set; "
            "the official mean is inf when any case is non-finite. finite_mean excludes "
            "those cases and is only a diagnostic."
        ),
        "std_policy": "sample standard deviation (ddof=1) over finite values when at least two exist",
        "n_both_empty": sum(row["empty_status"] == "both_empty" for row in rows),
        "n_pred_empty": sum(row["empty_status"] == "pred_empty" for row in rows),
        "n_gt_empty": sum(row["empty_status"] == "gt_empty" for row in rows),
        "n_failed_cases": 0,
    }
    for key in ("dice", "jaccard", "asd", "hd95"):
        values = np.array([row[key] for row in rows], dtype=np.float64)
        finite = values[np.isfinite(values)]
        n_nonfinite = int(values.size - finite.size)
        official_mean = float(np.mean(values)) if n_nonfinite == 0 and values.size else float("inf")
        summary[key] = {
            "mean": official_mean,
            "std": float(np.std(finite, ddof=1)) if finite.size >= 2 else None,
            "n_nonfinite": n_nonfinite,
            "finite_mean": float(np.mean(finite)) if finite.size else None,
        }
    summary["mean_dice"] = summary["dice"]["mean"]
    summary["mean_dice_exceeds_0.70"] = bool(np.isfinite(summary["mean_dice"]) and summary["mean_dice"] > 0.70)
    return summary
