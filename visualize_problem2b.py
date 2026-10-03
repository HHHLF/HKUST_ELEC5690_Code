#!/usr/bin/env python
"""Problem 2(b): training-loss curve, four test cases, and error analysis.

Reads the Problem 2(a) history, per-case metrics, and raw full-volume predictions.
Does not train, does not change the checkpoint, and does not rewrite the test metrics.
"""

from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import font_manager as _font_manager
from matplotlib.patches import Patch
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from la_seg3d.metrics import binary_segmentation_metrics, surface_distances  # noqa: E402
from la_seg3d.nrrd_io import read_nrrd  # noqa: E402


AXIS_INDEX = {"X": 0, "Y": 1, "Z": 2}
ROLE_ORDER = (
    "high_dice_low_hd95",
    "median_dice",
    "low_dice",
    "high_hd95_decent_dice",
)
_TIMES_FONT = "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf"
# Matplotlib sizes below are 4 pt above the previous figure text.
_FONT_AXIS = 14
_FONT_TICK = 14
_FONT_LEGEND = 14
_FONT_CAPTION = 14
_FONT_GRID_CAPTION = 19


def _use_times() -> None:
    """Use the installed Times-compatible face. This machine has no Times New Roman file."""
    if not Path(_TIMES_FONT).is_file():
        raise RuntimeError(f"Times-compatible font is missing: {_TIMES_FONT}")
    _font_manager.fontManager.addfont(_TIMES_FONT)
    plt.rcParams["font.family"] = "Nimbus Roman"
    plt.rcParams["font.size"] = _FONT_CAPTION
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


ROLE_LABEL = {
    "high_dice_low_hd95": "high Dice, low HD95",
    "median_dice": "Dice near the test median",
    "low_dice": "low Dice",
    "high_hd95_decent_dice": "Dice still reasonable, HD95 elevated",
}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/la_seg3d/problem2b.yaml")
    args = parser.parse_args()
    cfg = json.loads(json.dumps(_load_yaml(ROOT / args.config)))
    out_dir = ROOT / cfg["output_dir"]
    out_dir.mkdir(parents=True, exist_ok=True)

    history = _read_history(ROOT / cfg["history_csv"])
    rows = _read_metrics(ROOT / cfg["metrics_csv"])
    summary = json.loads((ROOT / cfg["summary_json"]).read_text())
    split = json.loads((ROOT / cfg["split_json"]).read_text())
    test_ids = list(split["test_ids"])
    if set(row["case_id"] for row in rows) != set(test_ids):
        raise RuntimeError("per-case metrics do not match split.json test ids")
    if summary.get("postprocess") != "none":
        raise RuntimeError(f"Problem 2(a) postprocess is {summary.get('postprocess')}, expected none")
    if summary.get("distance_unit") != "voxel":
        raise RuntimeError(f"Problem 2(a) distance unit is {summary.get('distance_unit')}")

    _plot_training_loss(history, out_dir, int(cfg["dpi"]))
    selected = _select_cases(rows, cfg.get("case_ids") or [])
    volumes = _load_volumes(selected, ROOT / cfg["predictions_dir"], cfg, summary)
    prepared = [_prepare_case(item, cfg) for item in volumes]
    _check_against_saved_metrics(prepared)

    _plot_main_grid(prepared, out_dir, int(cfg["dpi"]))
    _plot_error_grid(prepared, out_dir, int(cfg["dpi"]))
    _write_metrics_table(prepared, out_dir)
    _write_selection(selected, rows, cfg, summary, out_dir)
    _write_report(prepared, history, summary, out_dir)
    _verify_outputs(prepared, out_dir, test_ids)
    print(f"wrote {out_dir}", flush=True)
    for item in prepared:
        print(
            f"{item['role']} {item['case_id']} z={item['slice_index']} "
            f"dice={item['dice']:.4f} hd95={item['hd95']:.4f} {item['distance_unit']}",
            flush=True,
        )


def _load_yaml(path: Path) -> dict:
    import yaml

    with path.open() as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError(f"{path} did not contain a mapping")
    return cfg


def _read_history(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise RuntimeError(f"{path} has no epoch rows")
    epochs = [int(row["epoch"]) for row in rows]
    if epochs != list(range(1, len(rows) + 1)):
        raise RuntimeError(f"{path} epochs are not contiguous from 1")
    parsed = []
    for row in rows:
        total = float(row["loss"])
        ce = float(row["ce_loss"])
        dice = float(row["dice_loss"])
        parsed.append({"epoch": int(row["epoch"]), "loss": total, "ce_loss": ce, "dice_loss": dice})
    gap = max(abs(item["loss"] - (item["ce_loss"] + item["dice_loss"])) for item in parsed)
    if gap > 1e-6:
        raise RuntimeError(f"history total loss is not CE + Dice; max gap {gap}")
    return parsed


def _read_metrics(path: Path) -> list[dict]:
    with path.open() as handle:
        rows = list(csv.DictReader(handle))
    parsed = []
    for row in rows:
        item = dict(row)
        for key in ("dice", "jaccard", "asd", "hd95"):
            item[key] = float(row[key])
        item["pred_foreground_voxels"] = int(row["pred_foreground_voxels"])
        item["gt_foreground_voxels"] = int(row["gt_foreground_voxels"])
        parsed.append(item)
    return parsed


def _select_cases(rows: list[dict], requested: list[str]) -> list[dict]:
    by_id = {row["case_id"]: row for row in rows}
    if requested:
        if len(requested) != 4 or len(set(requested)) != 4:
            raise RuntimeError("case_ids must contain four different ids")
        missing = [case_id for case_id in requested if case_id not in by_id]
        if missing:
            raise RuntimeError(f"case_ids are not in the test metrics: {missing}")
        return [{"role": "manual", "case_id": case_id, "saved": by_id[case_id]} for case_id in requested]

    dices = np.array([row["dice"] for row in rows], dtype=np.float64)
    median = float(np.median(dices))
    p75 = float(np.percentile(dices, 75))
    remaining = set(by_id)
    chosen = []

    high = [row for row in rows if row["dice"] >= p75]
    high.sort(key=lambda row: (row["hd95"], -row["dice"], row["case_id"]))
    chosen.append(_take(high[0], "high_dice_low_hd95", remaining))

    median_pool = [by_id[case_id] for case_id in remaining]
    median_pool.sort(key=lambda row: (abs(row["dice"] - median), row["case_id"]))
    chosen.append(_take(median_pool[0], "median_dice", remaining))

    low_pool = [by_id[case_id] for case_id in remaining]
    low_pool.sort(key=lambda row: (row["dice"], row["case_id"]))
    chosen.append(_take(low_pool[0], "low_dice", remaining))

    hd_pool = [by_id[case_id] for case_id in remaining if by_id[case_id]["dice"] >= median]
    if not hd_pool:
        hd_pool = [by_id[case_id] for case_id in remaining]
    hd_pool.sort(key=lambda row: (-row["hd95"], row["case_id"]))
    chosen.append(_take(hd_pool[0], "high_hd95_decent_dice", remaining))

    for item in chosen:
        item["selection_stats"] = {"test_median_dice": median, "test_dice_p75": p75}
    return chosen


def _take(row: dict, role: str, remaining: set[str]) -> dict:
    remaining.remove(row["case_id"])
    return {"role": role, "case_id": row["case_id"], "saved": row}


def _load_volumes(selected: list[dict], pred_dir: Path, cfg: dict, summary: dict) -> list[dict]:
    loaded = []
    missing = []
    for item in selected:
        path = pred_dir / f"{item['case_id']}.npz"
        if not path.is_file():
            missing.append(item["case_id"])
            continue
        loaded.append(_read_prediction_npz(item, path))
    if missing:
        inferred = _infer_missing(missing, cfg, summary, ROOT / cfg["output_dir"] / "predictions")
        by_id = {item["case_id"]: item for item in inferred}
        loaded = []
        for item in selected:
            path = pred_dir / f"{item['case_id']}.npz"
            if path.is_file():
                loaded.append(_read_prediction_npz(item, path))
            else:
                loaded.append(by_id[item["case_id"]])
    return loaded


def _read_prediction_npz(item: dict, path: Path) -> dict:
    archive = np.load(path)
    axis_order = [str(v) for v in archive["axis_order"].tolist()]
    if axis_order != ["X", "Y", "Z"]:
        raise RuntimeError(f"{item['case_id']} axis_order is {axis_order}")
    unit = str(archive["distance_unit"])
    if unit != "voxel":
        raise RuntimeError(f"{item['case_id']} prediction distance unit is {unit}")
    image = archive["image"]
    label = archive["label"]
    prediction = archive["prediction"]
    spacing = tuple(float(v) for v in archive["spacing_xyz"].tolist())
    shape = tuple(int(v) for v in archive["shape_xyz"].tolist())
    if image.shape != label.shape or label.shape != prediction.shape:
        raise RuntimeError(f"{item['case_id']} MRI, GT, and prediction shapes differ")
    if tuple(image.shape) != shape:
        raise RuntimeError(f"{item['case_id']} array shape {image.shape} != stored shape {shape}")
    saved_shape = _parse_int_list(item["saved"]["shape_xyz"])
    if list(shape) != saved_shape:
        raise RuntimeError(f"{item['case_id']} shape {shape} != metrics shape {saved_shape}")
    saved_spacing = _parse_float_list(item["saved"]["spacing_xyz"])
    if [float(v) for v in spacing] != saved_spacing:
        raise RuntimeError(f"{item['case_id']} spacing does not match the metrics table")
    record = dict(item)
    record.update(
        {
            "image": image,
            "label": label,
            "prediction": prediction,
            "spacing_xyz": spacing,
            "shape_xyz": list(shape),
            "prediction_path": str(path.relative_to(ROOT)),
            "prediction_source": "saved_raw_prediction",
        }
    )
    return record


def _infer_missing(case_ids: list[str], cfg: dict, summary: dict, dest: Path) -> list[dict]:
    """Fill gaps with the same sliding-window path used for Problem 2(a). Never writes into test/."""
    import torch

    from la_seg3d.data import VolumeStore
    from la_seg3d.engine import build_model
    from la_seg3d.infer import predict_volume
    from la_seg3d.utils import load_config, pick_device

    train_cfg = load_config(ROOT / "configs/la_seg3d/unet3d_ce_dice.yaml")
    if train_cfg["infer"]["postprocess"] != "none":
        raise RuntimeError("refusing to infer with postprocess enabled")
    checkpoint_path = ROOT / summary["checkpoint"]
    device = pick_device(train_cfg)
    checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
    store = VolumeStore(train_cfg)
    store.prepare(retain="test", keep_raw=True)
    model = build_model(train_cfg).to(device)
    model.load_state_dict(checkpoint["model"])
    model.eval()
    dest.mkdir(parents=True, exist_ok=True)
    records = []
    patch_size = [int(v) for v in train_cfg["data"]["patch_size"]]
    for case_id in case_ids:
        case = store.cases[case_id]
        prediction, _info = predict_volume(
            model,
            case["image"],
            patch_size=patch_size,
            overlap=float(train_cfg["infer"]["overlap"]),
            sw_batch_size=int(train_cfg["infer"]["sw_batch_size"]),
            device=device,
            amp=bool(train_cfg["train"]["amp"]),
            sigma_scale=float(train_cfg["infer"]["gaussian_sigma_scale"]),
        )
        path = dest / f"{case_id}.npz"
        np.savez_compressed(
            path,
            prediction=prediction,
            image=case["image_raw"],
            label=case["label"],
            spacing_xyz=np.array(case["spacing_xyz"], dtype=np.float32),
            shape_xyz=np.array(case["shape_xyz"], dtype=np.int32),
            axis_order=np.array(["X", "Y", "Z"]),
            distance_unit=np.array("voxel"),
        )
        item = {
            "role": "inferred",
            "case_id": case_id,
            "saved": {"shape_xyz": str(case["shape_xyz"]), "spacing_xyz": str(case["spacing_xyz"])},
        }
        records.append(_read_prediction_npz(item, path) | {"prediction_source": "inferred_from_best_checkpoint"})
    return records


def _prepare_case(item: dict, cfg: dict) -> dict:
    image = np.asarray(item["image"])
    gt = np.asarray(item["label"]).astype(bool)
    pred = np.asarray(item["prediction"]).astype(bool)
    spacing = tuple(item["spacing_xyz"])
    recomputed = binary_segmentation_metrics(pred, gt, spacing)
    axis_name = str(cfg["slice_axis"])
    axis = AXIS_INDEX[axis_name]
    manual = _manual_slice(cfg.get("slice_indices"), item["case_id"], item["role"])
    slice_index, slice_rule = _select_slice(
        gt,
        pred,
        axis,
        float(cfg["gt_area_min_fraction_of_max"]),
        manual,
    )
    sl = _index(axis, slice_index)
    gt_slice = gt[sl]
    pred_slice = pred[sl]
    mri_slice = image[sl]
    tp_s, fp_s, fn_s = _counts(pred_slice, gt_slice)
    tp_v, fp_v, fn_v = _counts(pred, gt)
    window = _mri_window(mri_slice, [float(v) for v in cfg["mri_window_percentiles"]])
    components = _components(pred, gt, spacing, axis, slice_index)
    tail = _hd95_tail(pred, gt, spacing, recomputed["hd95"], axis, slice_index)
    nrrd_shape, nrrd_spacing, nrrd_units = _nrrd_header(item["case_id"])
    if list(nrrd_shape) != list(item["shape_xyz"]):
        raise RuntimeError(f"{item['case_id']} NRRD shape {nrrd_shape} != prediction shape {item['shape_xyz']}")
    if [float(v) for v in nrrd_spacing] != [float(v) for v in spacing]:
        raise RuntimeError(f"{item['case_id']} NRRD spacing does not match the prediction")
    unit = "voxel"
    if nrrd_units:
        raise RuntimeError(f"{item['case_id']} has space units {nrrd_units}; refusing to guess mm vs voxel")
    prepared = dict(item)
    prepared.update(
        {
            "slice_axis": axis_name,
            "slice_axis_index": axis,
            "slice_index": slice_index,
            "slice_rule": slice_rule,
            "dice": recomputed["dice"],
            "hd95": recomputed["hd95"],
            "distance_unit": unit,
            "spacing_xyz": list(spacing),
            "mri_slice": mri_slice,
            "gt_slice": gt_slice,
            "pred_slice": pred_slice,
            "window_vmin": window[0],
            "window_vmax": window[1],
            "slice_tp": tp_s,
            "slice_fp": fp_s,
            "slice_fn": fn_s,
            "volume_tp": tp_v,
            "volume_fp": fp_v,
            "volume_fn": fn_v,
            "components": components,
            "tail": tail,
            "nrrd_space_units": nrrd_units,
            "foreground_rgb": [float(v) for v in cfg["foreground_rgb"]],
            "foreground_alpha": float(cfg["foreground_alpha"]),
        }
    )
    return prepared


def _nrrd_header(case_id: str) -> tuple[list[int], tuple[float, float, float], str | None]:
    path = ROOT / ".." / "atriaseg2018" / "Testing Set" / case_id / "lgemri.nrrd"
    path = path.resolve()
    _volume, spacing, meta = read_nrrd(path)
    shape = [int(v) for v in meta["sizes"].split()]
    units = meta.get("space units")
    return shape, spacing, units


def _manual_slice(spec, case_id: str, role: str):
    if spec is None:
        return None
    if isinstance(spec, dict):
        if case_id not in spec or spec[case_id] is None:
            return None
        return int(spec[case_id])
    if isinstance(spec, list):
        if role == "manual":
            raise RuntimeError("a list of slice_indices requires case_ids in the same order")
        order = list(ROLE_ORDER)
        return int(spec[order.index(role)])
    raise RuntimeError("slice_indices must be null, a list, or a case-id mapping")


def _select_slice(gt: np.ndarray, pred: np.ndarray, axis: int, min_fraction: float, manual: int | None):
    n_slices = gt.shape[axis]
    gt_area = np.zeros(n_slices, dtype=np.int64)
    errors = np.zeros(n_slices, dtype=np.int64)
    for index in range(n_slices):
        sl = _index(axis, index)
        gt_slice = gt[sl]
        pred_slice = pred[sl]
        gt_area[index] = int(gt_slice.sum())
        errors[index] = int(np.logical_and(pred_slice, ~gt_slice).sum() + np.logical_and(~pred_slice, gt_slice).sum())
    if manual is not None:
        if not 0 <= manual < n_slices:
            raise RuntimeError(f"slice index {manual} is outside 0..{n_slices - 1}")
        return manual, "manual"
    max_area = int(gt_area.max())
    eligible = np.flatnonzero(gt_area >= min_fraction * max_area)
    if eligible.size == 0 or int(errors[eligible].max()) == 0:
        return int(np.argmax(gt_area)), "max_gt_area_because_no_eligible_error"
    best = None
    for index in eligible.tolist():
        key = (int(errors[index]), int(gt_area[index]), -int(index))
        if best is None or key > best[0]:
            best = (key, int(index))
    return best[1], "max_fp_plus_fn_among_slices_with_gt_area_at_least_20_percent_of_max"


def _index(axis: int, index: int) -> tuple:
    sl = [slice(None), slice(None), slice(None)]
    sl[axis] = index
    return tuple(sl)


def _counts(pred: np.ndarray, gt: np.ndarray) -> tuple[int, int, int]:
    tp = int(np.logical_and(pred, gt).sum())
    fp = int(np.logical_and(pred, ~gt).sum())
    fn = int(np.logical_and(~pred, gt).sum())
    return tp, fp, fn


def _mri_window(mri: np.ndarray, percentiles: list[float]) -> tuple[float, float]:
    values = mri[mri != 0]
    if values.size < 10:
        values = mri.reshape(-1)
    vmin = float(np.percentile(values, percentiles[0]))
    vmax = float(np.percentile(values, percentiles[1]))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _components(
    pred: np.ndarray,
    gt: np.ndarray,
    spacing: tuple[float, float, float],
    axis: int,
    slice_index: int,
) -> list[dict]:
    structure = generate_binary_structure(rank=3, connectivity=3)
    labeled, count = label(pred, structure=structure)
    if not gt.any():
        distance_to_gt = None
    else:
        distance_to_gt = distance_transform_edt(~gt, sampling=spacing)
    rows = []
    for index in range(1, int(count) + 1):
        region = labeled == index
        overlap = int(np.logical_and(region, gt).sum())
        coords = np.argwhere(region)
        centroid = [float(v) for v in coords.mean(axis=0)]
        min_distance = None if distance_to_gt is None else float(distance_to_gt[region].min())
        rows.append(
            {
                "label": index,
                "voxels": int(region.sum()),
                "overlap_voxels": overlap,
                "centroid_xyz": centroid,
                "min_distance_to_gt_voxel": min_distance,
                "z_min": int(coords[:, 2].min()),
                "z_max": int(coords[:, 2].max()),
                "voxels_on_selected_slice": int(np.take(region, slice_index, axis=axis).sum()),
            }
        )
    rows.sort(key=lambda row: (-row["voxels"], row["label"]))
    return rows


def _hd95_tail(pred, gt, spacing, hd95: float, axis: int, slice_index: int) -> dict:
    pred_surface, pred_dist = _surface_distance_volume(pred, gt, spacing)
    gt_surface, gt_dist = _surface_distance_volume(gt, pred, spacing)
    pred_d = pred_dist[pred_surface]
    gt_d = gt_dist[gt_surface]
    combined = np.concatenate([pred_d, gt_d])
    tail_level = float(np.percentile(combined, 95))
    pred_coords = np.argwhere(pred_surface)
    gt_coords = np.argwhere(gt_surface)
    tail_pred = pred_d >= hd95 - 1e-6
    tail_gt = gt_d >= hd95 - 1e-6
    tail_coords = np.concatenate([pred_coords[tail_pred], gt_coords[tail_gt]], axis=0)
    on_slice = int((tail_coords[:, axis] == slice_index).sum()) if len(tail_coords) else 0
    z_counts = np.bincount(tail_coords[:, 2], minlength=pred.shape[2]) if len(tail_coords) else np.zeros(pred.shape[2])
    top_z = np.argsort(z_counts)[::-1][:5]
    return {
        "recomputed_hd95": tail_level,
        "n_surface_points": int(combined.size),
        "n_tail_points": int(len(tail_coords)),
        "n_tail_points_on_selected_slice": on_slice,
        "fraction_tail_on_selected_slice": float(on_slice / len(tail_coords)) if len(tail_coords) else 0.0,
        "max_surface_distance": float(combined.max()),
        "fraction_surface_farther_than_10_voxels": float(np.mean(combined > 10.0)),
        "top_tail_z_slices": [
            {"z": int(z), "n_tail_points": int(z_counts[z])} for z in top_z.tolist() if z_counts[z] > 0
        ],
    }


def _surface_distance_volume(source: np.ndarray, target: np.ndarray, spacing):
    from la_seg3d.metrics import _surface

    source_surface = _surface(source)
    target_surface = _surface(target)
    distances = distance_transform_edt(~target_surface, sampling=spacing)
    return source_surface, distances


def _check_against_saved_metrics(prepared: list[dict]) -> None:
    for item in prepared:
        saved = item["saved"]
        if abs(item["dice"] - saved["dice"]) > 1e-10:
            raise RuntimeError(
                f"{item['case_id']} recomputed Dice {item['dice']} != saved {saved['dice']}"
            )
        if int(item["prediction"].astype(bool).sum()) != saved["pred_foreground_voxels"]:
            raise RuntimeError(f"{item['case_id']} prediction foreground count != saved metrics")
        if int(item["label"].astype(bool).sum()) != saved["gt_foreground_voxels"]:
            raise RuntimeError(f"{item['case_id']} ground-truth foreground count != saved metrics")
        if abs(item["hd95"] - saved["hd95"]) > 1e-6:
            raise RuntimeError(
                f"{item['case_id']} recomputed HD95 {item['hd95']} != saved {saved['hd95']}"
            )
        if abs(item["tail"]["recomputed_hd95"] - saved["hd95"]) > 1e-6:
            raise RuntimeError(f"{item['case_id']} surface-tail HD95 does not match the saved value")
        # Touch the public helper so a future metrics change fails this script too.
        _ = surface_distances
        if saved.get("distance_unit") != "voxel":
            raise RuntimeError(f"{item['case_id']} saved distance unit is not voxel")


def _plot_training_loss(history: list[dict], out_dir: Path, dpi: int) -> None:
    _use_times()
    epochs = [row["epoch"] for row in history]
    fig, ax = plt.subplots(figsize=(8.2, 4.8))
    ax.plot(epochs, [row["loss"] for row in history], color="#111111", lw=1.8, label="total loss = CE + Dice")
    ax.plot(epochs, [row["ce_loss"] for row in history], color="#d95f02", lw=1.2, label="CE")
    ax.plot(epochs, [row["dice_loss"] for row in history], color="#1b9e77", lw=1.2, label="Dice loss")
    ax.set_xlabel("epoch", fontsize=_FONT_AXIS)
    ax.set_ylabel("loss", fontsize=_FONT_AXIS)
    ax.tick_params(axis="both", labelsize=_FONT_TICK)
    ax.legend(frameon=False, fontsize=_FONT_LEGEND)
    ax.grid(True, color="#dddddd", lw=0.6)
    ax.set_xlim(1, epochs[-1])
    fig.tight_layout()
    fig.savefig(out_dir / "training_loss.png", dpi=dpi)
    fig.savefig(out_dir / "training_loss.pdf")
    plt.close(fig)


def _plot_main_grid(prepared: list[dict], out_dir: Path, dpi: int) -> None:
    _use_times()
    fig = plt.figure(figsize=(13.2, 17.2))
    # wspace is half of the previous 0.045. No column titles or figure title.
    grid = fig.add_gridspec(4, 3, left=0.02, right=0.99, top=0.995, bottom=0.04, hspace=0.42, wspace=0.0225)
    axes = [[fig.add_subplot(grid[row, col]) for col in range(3)] for row in range(4)]
    for row, item in enumerate(prepared):
        panels = [
            ("mri", None),
            ("gt", item["gt_slice"]),
            ("pred", item["pred_slice"]),
        ]
        for col, (_name, mask) in enumerate(panels):
            ax = axes[row][col]
            _show_mri(ax, item["mri_slice"], item["window_vmin"], item["window_vmax"])
            if mask is not None:
                _show_mask(ax, mask, item["foreground_rgb"], item["foreground_alpha"])
            _clean_axis(ax)
        caption = (
            f"Case {item['case_id']}  |  {item['slice_axis']} slice {item['slice_index']}"
            f"  |  3D Dice: {100.0 * item['dice']:.2f}%"
            f"  |  3D HD95: {item['hd95']:.2f} {item['distance_unit']}"
        )
        item["_row_caption"] = caption
    fig.canvas.draw()
    for row, item in enumerate(prepared):
        left = axes[row][0].get_position()
        right = axes[row][2].get_position()
        fig.text(
            (left.x0 + right.x1) / 2.0,
            left.y0 - 0.012,
            item["_row_caption"],
            ha="center",
            va="top",
            fontsize=_FONT_GRID_CAPTION,
            fontfamily="Nimbus Roman",
        )
    fig.savefig(out_dir / "test_cases_4x3.png", dpi=dpi)
    fig.savefig(out_dir / "test_cases_4x3.pdf")
    plt.close(fig)


def _plot_error_grid(prepared: list[dict], out_dir: Path, dpi: int) -> None:
    _use_times()
    fig = plt.figure(figsize=(12.2, 16.6))
    grid = fig.add_gridspec(4, 1, left=0.18, right=0.82, top=0.94, bottom=0.035, hspace=0.34)
    axes = []
    colors = {
        "tp": (0.15, 0.72, 0.25, 0.62),
        "fp": (0.90, 0.12, 0.12, 0.75),
        "fn": (0.15, 0.35, 0.95, 0.75),
    }
    for row, item in enumerate(prepared):
        ax = fig.add_subplot(grid[row, 0])
        axes.append(ax)
        _show_mri(ax, item["mri_slice"], item["window_vmin"], item["window_vmax"])
        overlay = np.zeros(item["gt_slice"].T.shape + (4,), dtype=np.float32)
        gt = item["gt_slice"].T
        pred = item["pred_slice"].T
        overlay[np.logical_and(pred, gt)] = colors["tp"]
        overlay[np.logical_and(pred, ~gt)] = colors["fp"]
        overlay[np.logical_and(~pred, gt)] = colors["fn"]
        ax.imshow(overlay, origin="lower", interpolation="nearest")
        _clean_axis(ax)
        item["_error_caption"] = (
            f"Case {item['case_id']}  |  {item['slice_axis']} slice {item['slice_index']}"
            f"  |  slice TP {item['slice_tp']}  FP {item['slice_fp']}  FN {item['slice_fn']}"
        )
    handles = [
        Patch(facecolor=colors["tp"], edgecolor="none", label="TP (prediction and GT)"),
        Patch(facecolor=colors["fp"], edgecolor="none", label="FP (prediction only)"),
        Patch(facecolor=colors["fn"], edgecolor="none", label="FN (GT only)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False, fontsize=_FONT_LEGEND, bbox_to_anchor=(0.5, 0.985))
    fig.canvas.draw()
    for ax, item in zip(axes, prepared):
        position = ax.get_position()
        fig.text(
            (position.x0 + position.x1) / 2.0,
            position.y0 - 0.008,
            item["_error_caption"],
            ha="center",
            va="top",
            fontsize=_FONT_CAPTION,
            fontfamily="Nimbus Roman",
        )
    fig.savefig(out_dir / "test_cases_error_overlay.png", dpi=dpi)
    fig.savefig(out_dir / "test_cases_error_overlay.pdf")
    plt.close(fig)


def _show_mri(ax, mri: np.ndarray, vmin: float, vmax: float) -> None:
    # (X, Y) slice transposed: horizontal = X, vertical = Y, origin at the lower left.
    ax.imshow(mri.T, cmap="gray", origin="lower", vmin=vmin, vmax=vmax, interpolation="nearest")
    ax.set_aspect("equal")


def _show_mask(ax, mask: np.ndarray, rgb: list[float], alpha: float) -> None:
    overlay = np.zeros(mask.T.shape + (4,), dtype=np.float32)
    overlay[mask.T] = (rgb[0], rgb[1], rgb[2], alpha)
    ax.imshow(overlay, origin="lower", interpolation="nearest")


def _clean_axis(ax) -> None:
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)


def _write_metrics_table(prepared: list[dict], out_dir: Path) -> None:
    rows = [_metrics_row(item) for item in prepared]
    fields = list(rows[0].keys())
    with (out_dir / "selected_cases_metrics.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    (out_dir / "selected_cases_metrics.json").write_text(json.dumps(rows, indent=2) + "\n")
    (out_dir / "case_evidence.json").write_text(
        json.dumps([{k: v for k, v in item.items() if k not in {"image", "label", "prediction", "mri_slice", "gt_slice", "pred_slice", "saved"}} for item in prepared], indent=2)
        + "\n"
    )


def _metrics_row(item: dict) -> dict:
    separated = [comp for comp in item["components"] if comp["overlap_voxels"] == 0]
    return {
        "case_id": item["case_id"],
        "role": item["role"],
        "slice_axis": item["slice_axis"],
        "slice_index": item["slice_index"],
        "slice_rule": item["slice_rule"],
        "dice_3d": item["dice"],
        "hd95_3d": item["hd95"],
        "hd95_unit": item["distance_unit"],
        "slice_tp": item["slice_tp"],
        "slice_fp": item["slice_fp"],
        "slice_fn": item["slice_fn"],
        "volume_tp": item["volume_tp"],
        "volume_fp": item["volume_fp"],
        "volume_fn": item["volume_fn"],
        "shape_xyz": item["shape_xyz"],
        "spacing_xyz": item["spacing_xyz"],
        "n_pred_components_26": len(item["components"]),
        "n_pred_components_without_gt_overlap": len(separated),
        "largest_nonoverlap_component_voxels": max((comp["voxels"] for comp in separated), default=0),
        "fraction_hd95_tail_on_selected_slice": item["tail"]["fraction_tail_on_selected_slice"],
        "max_surface_distance_voxel": item["tail"]["max_surface_distance"],
        "prediction_path": item["prediction_path"],
    }


def _write_selection(selected, rows, cfg, summary, out_dir: Path) -> None:
    dices = np.array([row["dice"] for row in rows], dtype=np.float64)
    payload = {
        "source_metrics": cfg["metrics_csv"],
        "checkpoint": summary.get("checkpoint"),
        "checkpoint_epoch": summary.get("checkpoint_epoch"),
        "postprocess": "none",
        "prediction_policy": "raw argmax of the sliding-window softmax; no connected-component cleanup",
        "distance_unit": "voxel",
        "distance_unit_reason": (
            "NRRD space directions are (1,0,0) (0,1,0) (0,0,1) and the headers have no space units. "
            "HD95 is therefore reported in voxels. Millimetres are not assumed."
        ),
        "test_median_dice": float(np.median(dices)),
        "test_dice_p75": float(np.percentile(dices, 75)),
        "rules": {
            "high_dice_low_hd95": "Among test cases with Dice at or above the 75th percentile, pick the lowest HD95. Ties break toward higher Dice, then case_id.",
            "median_dice": "Among the remaining cases, pick the Dice closest to the test-set median. Ties break by case_id.",
            "low_dice": "Among the remaining cases, pick the lowest Dice. Ties break by case_id.",
            "high_hd95_decent_dice": "Among the remaining cases with Dice at or above the test median, pick the highest HD95. Ties break by case_id.",
        },
        "slice_rule": (
            "Axis Z (array axis 2; NRRD sizes are X Y Z). Keep slices whose GT area is at least 20% of that case's maximum slice GT area. "
            "Among them, pick the slice with the most FP+FN voxels. Ties break toward larger GT area, then the smaller index. "
            "If none of those slices has an error, pick the slice with the largest GT area."
        ),
        "display": "Each Z slice is transposed for imshow so the horizontal axis is X and the vertical axis is Y, origin lower. All three columns share that orientation and the MRI window.",
        "cases": [
            {
                "role": item["role"],
                "case_id": item["case_id"],
                "dice": item["saved"]["dice"],
                "hd95": item["saved"]["hd95"],
            }
            for item in selected
        ],
    }
    (out_dir / "case_selection.json").write_text(json.dumps(payload, indent=2) + "\n")


def _write_report(prepared: list[dict], history: list[dict], summary: dict, out_dir: Path) -> None:
    loss_text = _loss_paragraph(history)
    rows = "\n".join(_latex_table_row(item) for item in prepared)
    analyses = "\n\n".join(_latex_case(item) for item in prepared)
    comparison = _comparison_paragraph(prepared)
    text = rf"""\documentclass{{article}}
\usepackage{{graphicx}}
\usepackage[margin=0.85in]{{geometry}}
\title{{Problem 2(b): Visualization and Error Analysis}}
\author{{}}
\date{{}}
\begin{{document}}
\maketitle

\section{{Training loss}}
The curve uses the epoch-mean training loss already stored in \texttt{{outputs/segmentation3d/history.csv}} for the Problem 2(a) run (\texttt{{best.pt}}, epoch {summary.get("checkpoint_epoch")}). Each point is the mean of the iteration losses in that epoch. The total is CE + Dice with both weights equal to 1. CE and Dice loss are plotted as well. The curves are the raw epoch values, with no smoothing and no resampling.

\begin{{figure}}[ht]
\centering
\includegraphics[width=\textwidth]{{training_loss.pdf}}
\caption{{Training loss of the 3D U-Net. The black curve is the epoch-mean total loss, equal to CE + Dice. The orange and green curves are the CE and Dice components.}}
\end{{figure}}

{loss_text}

\section{{Four test cases}}
The four cases are distinct members of the official test split. Predictions are the raw full-volume argmax masks saved by Problem 2(a) (\texttt{{postprocess: none}}), restored on the original \((X,Y,Z)\) grid. No connected-component filter was added. Dice and HD95 below are the full-volume values and match \texttt{{per\_case\_metrics.csv}}. HD95 is in voxels: the NRRD \texttt{{space directions}} have unit length and the files contain no \texttt{{space units}}.

\begin{{figure}}[ht]
\centering
\includegraphics[width=0.96\textwidth]{{test_cases_4x3.pdf}}
\caption{{One axial (Z) slice from each of four test cases. Columns are the MRI, the same MRI with the ground-truth mask, and the same MRI with the predicted mask. The foreground color and the grayscale window are shared within a row. The printed Dice and HD95 are 3D volume scores, not scores of the displayed slice.}}
\end{{figure}}

\begin{{table}}[ht]
\centering
\resizebox{{\textwidth}}{{!}}{{%
\begin{{tabular}}{{llrrrr}}
\hline
Case & Role & Z slice & 3D Dice (\%) & 3D HD95 (voxel) & Volume FP / FN \\
\hline
{rows}
\hline
\end{{tabular}}}}
\caption{{Full-volume Dice and HD95 for the visualized test cases. FP and FN counts are also full-volume voxel counts. Slice-wise counts are in \texttt{{selected\_cases\_metrics.csv}} and are not a substitute for these 3D scores.}}
\end{{table}}

\section{{Error analysis}}
The auxiliary overlay marks true positives in green, false positives in red, and false negatives in blue on the same slices. It is evidence for the notes below. It does not replace the 4$\times$3 figure, and a slice count is not the 3D score.

\begin{{figure}}[ht]
\centering
\includegraphics[width=0.48\textwidth]{{test_cases_error_overlay.pdf}}
\caption{{Error overlay for the same four Z slices. Green is overlap, red is predicted foreground outside the ground truth, and blue is missed ground truth.}}
\end{{figure}}

{analyses}

\section{{Dice and HD95}}
{comparison}

\end{{document}}
"""
    (out_dir / "report_problem2b.tex").write_text(text)


def _loss_paragraph(history: list[dict]) -> str:
    first = history[0]
    last = history[-1]
    by_epoch = {row["epoch"]: row for row in history}
    early = by_epoch[68]
    dropped = by_epoch[70]
    epoch_100 = by_epoch[100]
    min_row = min(history, key=lambda row: row["loss"])
    late = history[199:]
    late_min = min(row["loss"] for row in late)
    late_max = max(row["loss"] for row in late)
    return (
        f"The epoch-mean total loss starts at {first['loss']:.3f} in epoch 1 "
        f"(CE {first['ce_loss']:.3f}, Dice loss {first['dice_loss']:.3f}). "
        f"It declines with visible epoch-to-epoch fluctuation, then the recorded mean falls from "
        f"{early['loss']:.3f} at epoch 68 to {dropped['loss']:.3f} at epoch 70. "
        f"The log does not by itself identify the cause of that step. "
        f"By epoch 100 the total is {epoch_100['loss']:.3f} (CE {epoch_100['ce_loss']:.3f}, Dice loss {epoch_100['dice_loss']:.3f}). "
        f"The lowest epoch mean is {min_row['loss']:.3f} at epoch {min_row['epoch']}. "
        f"From epoch 200 through epoch {last['epoch']} the total stays between {late_min:.3f} and {late_max:.3f}, "
        f"and epoch {last['epoch']} ends at {last['loss']:.3f}. "
        f"After epoch 70 the Dice component remains larger than CE, so the late total tracks the Dice loss."
    )


def _latex_table_row(item: dict) -> str:
    role = ROLE_LABEL.get(item["role"], item["role"]).replace("%", r"\%")
    return (
        f"{item['case_id']} & {role} & {item['slice_index']} & "
        f"{100.0 * item['dice']:.2f} & {item['hd95']:.2f} & "
        f"{item['volume_fp']} / {item['volume_fn']} \\\\"
    )


def _latex_case(item: dict) -> str:
    body = _analysis_sentences(item)
    return f"\\paragraph{{{item['case_id']}.}}\n{body}"


def _analysis_sentences(item: dict) -> str:
    writers = {
        "high_dice_low_hd95": _analysis_high_dice,
        "median_dice": _analysis_median,
        "low_dice": _analysis_low_dice,
        "high_hd95_decent_dice": _analysis_high_hd95,
    }
    return writers[item["role"]](item)


def _fp_share(item: dict) -> float:
    return 100.0 * item["slice_fp"] / item["volume_fp"] if item["volume_fp"] else 0.0


def _fn_share(item: dict) -> float:
    return 100.0 * item["slice_fn"] / item["volume_fn"] if item["volume_fn"] else 0.0


def _separated(item: dict) -> list[dict]:
    rows = [comp for comp in item["components"] if comp["overlap_voxels"] == 0]
    rows.sort(key=lambda comp: -comp["voxels"])
    return rows


def _analysis_high_dice(item: dict) -> str:
    return (
        f"On Z slice {item['slice_index']} the error overlay shows a thin false-positive rim around the overlapped atrium "
        f"and a small ground-truth focus that is only partly covered "
        f"(TP {item['slice_tp']}, FP {item['slice_fp']}, FN {item['slice_fn']}). "
        f"The prediction is a single 26-connected component and it overlaps the ground truth, so this is a boundary mismatch rather than a detached false-positive island. "
        f"Full-volume false positives ({item['volume_fp']}) exceed false negatives ({item['volume_fn']}); "
        f"both reduce $|P \\cap G|$ relative to $|P|+|G|$, but the overlap is still large enough for a 3D Dice of {100.0 * item['dice']:.2f}\\%. "
        f"HD95 stays {item['hd95']:.2f} voxel because the surface deviations are short: the maximum surface distance is {item['tail']['max_surface_distance']:.2f} voxel, "
        f"and {100.0 * item['tail']['fraction_surface_farther_than_10_voxels']:.3f}\\% of surface points are farther than 10 voxels. "
        f"Only {100.0 * item['tail']['fraction_tail_on_selected_slice']:.1f}\\% of the HD95 tail lies on this slice, so the same kind of rim also occurs elsewhere in the volume."
    )


def _analysis_median(item: dict) -> str:
    far = [comp for comp in _separated(item) if comp["voxels"] == 1]
    if len(far) != 1:
        raise RuntimeError(f"{item['case_id']} no longer has exactly one singleton false-positive component")
    singleton = far[0]
    if any(comp["voxels_on_selected_slice"] for comp in _separated(item)):
        raise RuntimeError(f"{item['case_id']} detached components now intersect the selected slice; rewrite the note")
    others = [comp for comp in _separated(item) if comp is not singleton]
    other_text = ", ".join(
        f"{comp['voxels']} voxels at least {comp['min_distance_to_gt_voxel']:.1f} voxels from the ground truth (Z {comp['z_min']}--{comp['z_max']})"
        for comp in others
    )
    return (
        f"On Z slice {item['slice_index']} the overlay shows added prediction on one side of the atrium and missed ground truth on another, with the center overlapped "
        f"(TP {item['slice_tp']}, FP {item['slice_fp']}, FN {item['slice_fn']}). "
        f"Those voxels belong to the main component; this slice contains none of the three predicted components that miss the ground truth. "
        f"Those components are {other_text}, plus one isolated voxel at Z {singleton['z_min']} whose nearest ground-truth voxel is {singleton['min_distance_to_gt_voxel']:.1f} voxels away. "
        f"The isolated voxel is not on the displayed slice. It sets the maximum surface distance, but it is one of {item['tail']['n_surface_points']} surface points, "
        f"so it does not move the 95th percentile: 3D HD95 remains {item['hd95']:.2f} voxel. "
        f"Only {100.0 * item['tail']['fraction_surface_farther_than_10_voxels']:.2f}\\% of surface points are farther than 10 voxels, and "
        f"{100.0 * item['tail']['fraction_tail_on_selected_slice']:.1f}\\% of the HD95 tail is on this slice. "
        f"Volume FP {item['volume_fp']} and FN {item['volume_fn']} are similar in count, and together they bring the 3D Dice to {100.0 * item['dice']:.2f}\\%."
    )


def _analysis_low_dice(item: dict) -> str:
    separated = _separated(item)
    on_slice = [comp for comp in separated if comp["voxels_on_selected_slice"] > 0]
    far = [comp for comp in separated if comp["min_distance_to_gt_voxel"] >= 50]
    near = [comp for comp in separated if comp["min_distance_to_gt_voxel"] < 20]
    island_voxels = sum(comp["voxels_on_selected_slice"] for comp in separated)
    if len(near) == 1:
        near_phrase = "1 component is closer than 20 voxels. That nearer component adds local false-positive volume"
    else:
        near_phrase = (
            f"{len(near)} components are closer than 20 voxels. "
            "The nearer components add local false-positive volume"
        )
    return (
        f"Z slice {item['slice_index']} shows the ground-truth atrium mostly covered (TP {item['slice_tp']}, FN {item['slice_fn']}) "
        f"and several red false-positive islands that do not overlap it. "
        f"Of the {item['slice_fp']} false-positive voxels on this slice, {island_voxels} lie in {len(on_slice)} components that have no ground-truth overlap anywhere; the rest are attached to the main prediction. "
        f"Across the volume there are {len(separated)} such components. "
        f"The largest has {separated[0]['voxels']} voxels and its nearest ground-truth voxel is {separated[0]['min_distance_to_gt_voxel']:.1f} voxels away; "
        f"{len(far)} components are at least 50 voxels from the ground truth, while {near_phrase}, "
        f"and the far islands place surface points hundreds of voxels away. "
        f"Full-volume FP ({item['volume_fp']}) greatly exceeds FN ({item['volume_fn']}), which enlarges $|P|$ without a matching increase in $|P \\cap G|$ and lowers the 3D Dice to {100.0 * item['dice']:.2f}\\%. "
        f"HD95 is {item['hd95']:.2f} voxel. This slice contains only {_fp_share(item):.1f}\\% of the volume FP voxels and "
        f"{100.0 * item['tail']['fraction_tail_on_selected_slice']:.1f}\\% of the HD95 tail "
        f"(the tail is heaviest on other Z slices, led by Z {item['tail']['top_tail_z_slices'][0]['z']}), "
        f"so the islands visible here illustrate the failure but do not contain it."
    )


def _analysis_high_hd95(item: dict) -> str:
    separated = _separated(item)
    if len(separated) != 1:
        raise RuntimeError(f"{item['case_id']} component structure changed")
    island = separated[0]
    return (
        f"On Z slice {item['slice_index']} the main atrium is largely overlapped, with a false-negative rim, and a separate predicted island that does not meet the ground truth on this slice "
        f"(TP {item['slice_tp']}, FP {item['slice_fp']}, FN {item['slice_fn']}; {island['voxels_on_selected_slice']} of the slice FP voxels belong to that island). "
        f"In 3D the island is one component of {island['voxels']} voxels spanning Z {island['z_min']}--{island['z_max']}, with no ground-truth overlap and a nearest ground-truth voxel {island['min_distance_to_gt_voxel']:.1f} voxels away. "
        f"It was not removed. Relative to the {item['volume_tp']} overlapping voxels, this extra region and the remaining boundary errors "
        f"(volume FP {item['volume_fp']}, FN {item['volume_fn']}) leave the 3D Dice at {100.0 * item['dice']:.2f}\\%. "
        f"HD95 is still {item['hd95']:.2f} voxel because the island's surface is far from the ground-truth surface, and "
        f"{100.0 * item['tail']['fraction_surface_farther_than_10_voxels']:.2f}\\% of all surface points are farther than 10 voxels. "
        f"Only {100.0 * item['tail']['fraction_tail_on_selected_slice']:.1f}\\% of that HD95 tail lies on Z {item['slice_index']}; "
        f"the largest shares are on Z {item['tail']['top_tail_z_slices'][0]['z']}, {item['tail']['top_tail_z_slices'][1]['z']}, and {item['tail']['top_tail_z_slices'][2]['z']}. "
        f"The displayed slice shows the island, but it is not where most of the far surface points sit."
    )


def _comparison_paragraph(prepared: list[dict]) -> str:
    by_role = {item["role"]: item for item in prepared}
    high = by_role["high_dice_low_hd95"]
    mid = by_role["median_dice"]
    low = by_role["low_dice"]
    far = by_role["high_hd95_decent_dice"]
    return (
        f"Dice responds to the total count of FP and FN voxels, while HD95 responds to how far the worse-aligned surface points are. "
        f"{high['case_id']} combines a high overlap (Dice {100.0 * high['dice']:.2f}\\%) with a short surface tail (HD95 {high['hd95']:.2f} voxel). "
        f"{mid['case_id']} sits near the test median Dice ({100.0 * mid['dice']:.2f}\\%) and its HD95 is {mid['hd95']:.2f} voxel. "
        f"{low['case_id']} has the largest count disagreement of the four (FP {low['volume_fp']}, FN {low['volume_fn']}), "
        f"and both scores suffer: Dice {100.0 * low['dice']:.2f}\\% and HD95 {low['hd95']:.2f} voxel. "
        f"{far['case_id']} still has Dice {100.0 * far['dice']:.2f}\\% with only {far['volume_fp']} FP and {far['volume_fn']} FN voxels, "
        f"but HD95 is {far['hd95']:.2f} voxel. The bulk of the atrium can overlap while a smaller set of surface points remains far enough to move the 95th percentile. "
        f"A handful of isolated voxels does not automatically do that: HD95 moves only when those far surface points make up enough of the surface distribution to reach the 95th percentile. "
        f"Here, {100.0 * far['tail']['fraction_surface_farther_than_10_voxels']:.2f}\\% of the surface points of {far['case_id']} are farther than 10 voxels, "
        f"against {100.0 * high['tail']['fraction_surface_farther_than_10_voxels']:.3f}\\% for {high['case_id']}. "
        f"{mid['case_id']} is the complementary case: one predicted voxel is {mid['tail']['max_surface_distance']:.1f} voxels from the ground truth, "
        f"but that single point is not enough to move the 95th percentile, and its HD95 stays {mid['hd95']:.2f} voxel."
    )


def _verify_outputs(prepared: list[dict], out_dir: Path, test_ids: list[str]) -> None:
    ids = [item["case_id"] for item in prepared]
    if len(ids) != 4 or len(set(ids)) != 4:
        raise RuntimeError("expected four distinct cases")
    if any(case_id not in test_ids for case_id in ids):
        raise RuntimeError("a selected case is outside the test split")
    required = [
        "training_loss.png",
        "training_loss.pdf",
        "test_cases_4x3.png",
        "test_cases_4x3.pdf",
        "test_cases_error_overlay.png",
        "test_cases_error_overlay.pdf",
        "selected_cases_metrics.csv",
        "selected_cases_metrics.json",
        "report_problem2b.tex",
    ]
    for name in required:
        path = out_dir / name
        if not path.is_file() or path.stat().st_size < 1000:
            raise RuntimeError(f"missing or empty output {path}")
    image = plt.imread(out_dir / "test_cases_4x3.png")
    if image.shape[0] < 3000 or image.shape[1] < 3000:
        raise RuntimeError(f"main figure is smaller than expected: {image.shape}")
    # The outer margin should not be a solid non-white crop of the MRI.
    if image.shape[-1] == 4:
        image = image[..., :3]
    border = np.concatenate([image[0].reshape(-1, 3), image[-1].reshape(-1, 3), image[:, 0].reshape(-1, 3), image[:, -1].reshape(-1, 3)])
    if float(border.min()) < 0.95:
        raise RuntimeError("main figure border is not the saved margin; the canvas may be clipped")


def _parse_int_list(text: str) -> list[int]:
    return [int(part) for part in text.strip("[]").split(",")]


def _parse_float_list(text: str) -> list[float]:
    return [float(part) for part in text.strip("[]").split(",")]


if __name__ == "__main__":
    main()
