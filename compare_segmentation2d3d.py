#!/usr/bin/env python
"""Numeric comparison and figures for the 2D and 3D full-volume test metrics."""

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
from matplotlib import font_manager as font_manager
from matplotlib.patches import Patch

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from la_seg3d.utils import load_config, write_json

METRICS = ("dice", "jaccard", "asd", "hd95")
HIGHER_BETTER = {"dice": True, "jaccard": True, "asd": False, "hd95": False}
LABELS = {
    "dice": "Dice",
    "jaccard": "Jaccard",
    "asd": "ASD",
    "hd95": "HD95",
}
_TIMES_FONT = "/usr/share/fonts/opentype/urw-base35/NimbusRoman-Regular.otf"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/segmentation2d.yaml")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    out_dir = ROOT / cfg["output"]["comparison_dir"]
    fig_dir = out_dir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    _use_font()

    ref = cfg["reference_3d"]
    rows_3d = _read_metric_rows(ROOT / ref["test_metrics"])
    rows_2d = _read_metric_rows(ROOT / cfg["output"]["dir"] / "test" / "per_case_metrics.json")
    by3 = {row["case_id"]: row for row in rows_3d}
    by2 = {row["case_id"]: row for row in rows_2d}
    if set(by3) != set(by2):
        missing_2d = sorted(set(by3) - set(by2))
        missing_3d = sorted(set(by2) - set(by3))
        raise RuntimeError(f"case sets differ; missing in 2D {missing_2d[:8]} missing in 3D {missing_3d[:8]}")
    if len(by3) != len(rows_3d) or len(by2) != len(rows_2d):
        raise RuntimeError("duplicate case_id in a metrics file")
    split = json.loads((ROOT / ref["split"]).read_text())
    if set(by3) != set(split["test_ids"]):
        raise RuntimeError("metrics case ids are not the 3D test split")

    unit = _one_unit(rows_2d, rows_3d)
    summary_rows = [_summary_row(name, by2, by3, unit) for name in METRICS]
    per_case = [_per_case_row(case_id, by2[case_id], by3[case_id]) for case_id in split["test_ids"]]
    write_json(out_dir / "comparison_summary.json", {"distance_unit": unit, "metrics": summary_rows})
    write_json(out_dir / "comparison_per_case.json", per_case)
    _write_csv(out_dir / "comparison_summary.csv", summary_rows)
    _write_csv(out_dir / "comparison_per_case.csv", per_case)

    _plot_boxes(per_case, unit, fig_dir, int(cfg.get("figure_dpi", 300)))
    _plot_paired(per_case, unit, fig_dir, int(cfg.get("figure_dpi", 300)))
    _plot_training(cfg, fig_dir, int(cfg.get("figure_dpi", 300)))
    alignment = _plot_cases(cfg, by2, by3, unit, fig_dir, int(cfg.get("figure_dpi", 300)))
    write_json(out_dir / "alignment_check.json", alignment)
    print(f"wrote {out_dir}", flush=True)
    for row in summary_rows:
        print(
            f"{row['metric']} 2d={row['mean_2d']} 3d={row['mean_3d']} delta={row['delta']} {row['unit']}",
            flush=True,
        )


def _use_font() -> None:
    if Path(_TIMES_FONT).is_file():
        font_manager.fontManager.addfont(_TIMES_FONT)
        plt.rcParams["font.family"] = "Nimbus Roman"
    plt.rcParams["pdf.fonttype"] = 42
    plt.rcParams["ps.fonttype"] = 42


def _as_float(value) -> float:
    if isinstance(value, (int, float)):
        return float(value)
    text = str(value).strip()
    if text.lower() in {"inf", "infinity", "+inf", "+infinity"}:
        return float("inf")
    if text.lower() in {"-inf", "-infinity"}:
        return float("-inf")
    return float(text)


def _read_metric_rows(path: Path) -> list[dict]:
    if path.suffix == ".json":
        payload = json.loads(path.read_text())
        rows = payload["cases"] if isinstance(payload, dict) and "cases" in payload else payload
    else:
        with path.open() as handle:
            rows = list(csv.DictReader(handle))
    parsed = []
    for row in rows:
        item = {
            "case_id": row["case_id"],
            "empty_status": row.get("empty_status", ""),
            "distance_unit": row.get("distance_unit", "voxel"),
        }
        for key in METRICS:
            item[key] = _as_float(row[key])
        parsed.append(item)
    return parsed


def _one_unit(rows_2d: list[dict], rows_3d: list[dict]) -> str:
    units = {row["distance_unit"] for row in rows_2d + rows_3d}
    if units != {"voxel"}:
        raise RuntimeError(f"distance units are {units}; refusing to mix them")
    return "voxel"


def _mean_std(values: list[float]) -> tuple[float, float | None, int]:
    array = np.array(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    n_nonfinite = int(array.size - finite.size)
    official = float(np.mean(array)) if n_nonfinite == 0 and array.size else float("inf")
    std = float(np.std(finite, ddof=1)) if finite.size >= 2 else None
    return official, std, n_nonfinite


def _summary_row(name: str, by2: dict, by3: dict, unit: str) -> dict:
    ids = sorted(by2)
    mean2, std2, non2 = _mean_std([by2[case_id][name] for case_id in ids])
    mean3, std3, non3 = _mean_std([by3[case_id][name] for case_id in ids])
    delta = mean2 - mean3 if np.isfinite(mean2) and np.isfinite(mean3) else float("inf")
    row = {
        "metric": name,
        "mean_2d": mean2,
        "std_2d": std2,
        "mean_3d": mean3,
        "std_3d": std3,
        "delta": delta,
        "unit": "1" if name in ("dice", "jaccard") else unit,
        "n_cases": len(ids),
        "n_nonfinite_2d": non2,
        "n_nonfinite_3d": non3,
        "delta_percentage_points": (delta * 100.0) if name in ("dice", "jaccard") and np.isfinite(delta) else "",
    }
    return row


def _per_case_row(case_id: str, row2: dict, row3: dict) -> dict:
    out = {"case_id": case_id, "empty_status_2d": row2["empty_status"], "empty_status_3d": row3["empty_status"]}
    for name in METRICS:
        out[f"{name}_2d"] = row2[name]
        out[f"{name}_3d"] = row3[name]
        a, b = row2[name], row3[name]
        out[f"{name}_delta"] = (a - b) if np.isfinite(a) and np.isfinite(b) else float("inf")
    return out


def _write_csv(path: Path, rows: list[dict]) -> None:
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _csv_value(value) for key, value in row.items()})


def _csv_value(value):
    if isinstance(value, float) and not np.isfinite(value):
        return "inf" if value > 0 else "-inf"
    return value


def _finite(values: list[float]) -> np.ndarray:
    array = np.array(values, dtype=np.float64)
    return array[np.isfinite(array)]


def _plot_boxes(per_case: list[dict], unit: str, fig_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    for ax, name in zip(axes.ravel(), METRICS):
        samples = [
            _finite([row[f"{name}_2d"] for row in per_case]),
            _finite([row[f"{name}_3d"] for row in per_case]),
        ]
        n_inf = [
            sum(not np.isfinite(row[f"{name}_2d"]) for row in per_case),
            sum(not np.isfinite(row[f"{name}_3d"]) for row in per_case),
        ]
        ax.boxplot(samples, tick_labels=["2D U-Net", "3D U-Net"], showfliers=False)
        for index, values in enumerate(samples, start=1):
            ax.scatter(np.full(values.shape, index), values, s=12, alpha=0.45, color="#333333", zorder=3)
            if values.size:
                ax.scatter([index], [float(np.mean(values))], marker="D", s=36, color="#d95f02", zorder=4)
        direction = "higher is better" if HIGHER_BETTER[name] else "lower is better"
        distance = f" ({unit})" if name in ("asd", "hd95") else ""
        ax.set_title(f"{LABELS[name]}{distance}\n{direction}; inf count 2D={n_inf[0]}, 3D={n_inf[1]}")
        ax.set_ylabel(f"{LABELS[name]}{distance}")
    fig.suptitle("Full-volume per-case metrics. Diamonds are means of finite values. inf is counted in the title.")
    fig.tight_layout()
    _save(fig, fig_dir / "metric_boxplots.png", dpi)


def _plot_paired(per_case: list[dict], unit: str, fig_dir: Path, dpi: int) -> None:
    fig, axes = plt.subplots(2, 2, figsize=(10, 8))
    n_inf = 0
    for ax, name in zip(axes.ravel(), METRICS):
        xs, ys = [], []
        skipped = 0
        for row in per_case:
            x, y = row[f"{name}_3d"], row[f"{name}_2d"]
            if np.isfinite(x) and np.isfinite(y):
                xs.append(x)
                ys.append(y)
            else:
                skipped += 1
        n_inf += skipped
        ax.scatter(xs, ys, s=16, color="#333333", alpha=0.7)
        if xs:
            low = min(min(xs), min(ys))
            high = max(max(xs), max(ys))
            ax.plot([low, high], [low, high], color="#d95f02", lw=1.0, label="y = x")
            ax.legend(frameon=False)
        distance = f" ({unit})" if name in ("asd", "hd95") else ""
        ax.set_xlabel(f"3D U-Net {LABELS[name]}{distance}")
        ax.set_ylabel(f"2D U-Net {LABELS[name]}{distance}")
        ax.set_title(f"{LABELS[name]} paired by case_id; non-finite pairs omitted here: {skipped}")
        ax.set_aspect("equal", adjustable="datalim")
    fig.suptitle("Each point is one test case. Axes are not sorted independently.")
    fig.tight_layout()
    _save(fig, fig_dir / "metric_paired.png", dpi)


def _plot_training(cfg: dict, fig_dir: Path, dpi: int) -> None:
    history_2d = json.loads((ROOT / cfg["output"]["dir"] / "history.json").read_text())["history"]
    steps = [int(row["global_step"]) for row in history_2d]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(steps, [row["loss"] for row in history_2d], color="#111111")
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("total loss")
    val_steps = [int(row["global_step"]) for row in history_2d if row.get("val_dice") not in (None, "")]
    val_dice = [float(row["val_dice"]) for row in history_2d if row.get("val_dice") not in (None, "")]
    axes[1].plot(val_steps, val_dice, marker="o", color="#1b9e77")
    axes[1].set_xlabel("optimizer step")
    axes[1].set_ylabel("full-volume validation Dice")
    axes[1].set_ylim(0, 1)
    fig.tight_layout()
    _save(fig, fig_dir / "training_curves_2d.png", dpi)

    with (ROOT / cfg["reference_3d"]["history_csv"]).open() as handle:
        history_3d = list(csv.DictReader(handle))
    finished = json.loads((ROOT / cfg["reference_3d"]["train_finished"]).read_text())
    snapshot = json.loads((ROOT / cfg["reference_3d"]["config_snapshot"]).read_text())
    split = json.loads((ROOT / cfg["reference_3d"]["split"]).read_text())
    from la_seg2d.protocol import three_d_optimizer_budget

    budget = three_d_optimizer_budget(snapshot, split, finished)
    updates_per_epoch = int(budget["optimizer_updates_per_epoch"])
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].plot(steps, [row["loss"] for row in history_2d], label="2D", color="#111111")
    axes[0].plot(
        [int(row["epoch"]) * updates_per_epoch for row in history_3d],
        [float(row["loss"]) for row in history_3d],
        label="3D",
        color="#7570b3",
    )
    axes[0].set_xlabel("optimizer step")
    axes[0].set_ylabel("total loss")
    axes[0].legend(frameon=False)
    axes[1].plot(val_steps, val_dice, marker="o", label="2D", color="#1b9e77")
    val3_steps, val3 = [], []
    for row in history_3d:
        if row.get("val_dice"):
            val3_steps.append(int(row["epoch"]) * updates_per_epoch)
            val3.append(float(row["val_dice"]))
    axes[1].plot(val3_steps, val3, marker="o", label="3D", color="#7570b3")
    axes[1].set_xlabel(f"optimizer step (3D epoch × {updates_per_epoch} updates)")
    axes[1].set_ylabel("full-volume validation Dice")
    axes[1].set_ylim(0, 1)
    axes[1].legend(frameon=False)
    fig.tight_layout()
    _save(fig, fig_dir / "training_curves_steps.png", dpi)


def _plot_cases(cfg: dict, by2: dict, by3: dict, unit: str, fig_dir: Path, dpi: int) -> dict:
    selected = json.loads((ROOT / cfg["reference_3d"]["problem2b_metrics"]).read_text())
    checks = []
    prepared = []
    for item in selected:
        case_id = item["case_id"]
        slice_index = int(item["slice_index"])
        path2 = ROOT / cfg["output"]["dir"] / "test" / "predictions" / f"{case_id}.npz"
        path3 = ROOT / cfg["reference_3d"]["predictions"] / f"{case_id}.npz"
        vol2 = np.load(path2)
        vol3 = np.load(path3)
        if tuple(vol2["shape_xyz"].tolist()) != tuple(vol3["shape_xyz"].tolist()):
            raise RuntimeError(f"{case_id} shape mismatch")
        if not np.array_equal(vol2["label"], vol3["label"]):
            raise RuntimeError(f"{case_id} labels differ between 2D and 3D prediction files")
        if not np.array_equal(vol2["image"], vol3["image"]):
            raise RuntimeError(f"{case_id} MRI arrays differ between 2D and 3D prediction files")
        if vol2["image"].shape[2] <= slice_index:
            raise RuntimeError(f"{case_id} slice {slice_index} is outside Z")
        mri = vol2["image"][:, :, slice_index]
        gt = vol2["label"][:, :, slice_index].astype(bool)
        pred2 = vol2["prediction"][:, :, slice_index].astype(bool)
        pred3 = vol3["prediction"][:, :, slice_index].astype(bool)
        window = _mri_window(mri, [1.0, 99.0])
        prepared.append(
            {
                "case_id": case_id,
                "role": item["role"],
                "slice_index": slice_index,
                "mri": mri,
                "gt": gt,
                "pred2": pred2,
                "pred3": pred3,
                "vmin": window[0],
                "vmax": window[1],
                "dice2": by2[case_id]["dice"],
                "hd2": by2[case_id]["hd95"],
                "dice3": by3[case_id]["dice"],
                "hd3": by3[case_id]["hd95"],
            }
        )
        checks.append(
            {
                "case_id": case_id,
                "slice_axis": "Z",
                "slice_axis_index": 2,
                "slice_index": slice_index,
                "shape_xyz": [int(v) for v in vol2["shape_xyz"].tolist()],
                "label_equal": True,
                "mri_equal": True,
                "displayed_shape_xy": list(mri.shape),
            }
        )
    _draw_main(prepared, unit, fig_dir, dpi)
    _draw_errors(prepared, fig_dir, dpi)
    return {"cases": checks}


def _draw_main(prepared: list[dict], unit: str, fig_dir: Path, dpi: int) -> None:
    fig = plt.figure(figsize=(16, 18))
    grid = fig.add_gridspec(4, 4, left=0.02, right=0.99, top=0.97, bottom=0.03, hspace=0.28, wspace=0.03)
    rgb = (0.90, 0.16, 0.12)
    axes = []
    for row, item in enumerate(prepared):
        panels = [
            (item["mri"], None, "MRI"),
            (item["mri"], item["gt"], "Ground truth"),
            (item["mri"], item["pred2"], "2D U-Net"),
            (item["mri"], item["pred3"], "3D U-Net"),
        ]
        row_axes = []
        for col, (image, mask, title) in enumerate(panels):
            ax = fig.add_subplot(grid[row, col])
            row_axes.append(ax)
            ax.imshow(image.T, cmap="gray", origin="lower", vmin=item["vmin"], vmax=item["vmax"], interpolation="nearest")
            if mask is not None:
                overlay = np.zeros(mask.T.shape + (4,), dtype=np.float32)
                overlay[mask.T] = (rgb[0], rgb[1], rgb[2], 0.45)
                ax.imshow(overlay, origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(title)
        item["caption"] = (
            f"{item['case_id']}  |  Z slice {item['slice_index']}  |  "
            f"full-volume 2D Dice {item['dice2']:.3f}  HD95 {_fmt_distance(item['hd2'])} {unit}  |  "
            f"full-volume 3D Dice {item['dice3']:.3f}  HD95 {_fmt_distance(item['hd3'])} {unit}"
        )
        axes.append(row_axes)
    fig.canvas.draw()
    for row_axes, item in zip(axes, prepared):
        left = row_axes[0].get_position()
        right = row_axes[-1].get_position()
        fig.text((left.x0 + right.x1) / 2.0, left.y0 - 0.012, item["caption"], ha="center", va="top", fontsize=10)
    _save(fig, fig_dir / "test_cases_4x4.png", dpi)


def _draw_errors(prepared: list[dict], fig_dir: Path, dpi: int) -> None:
    colors = {
        "tp": (0.15, 0.72, 0.25, 0.62),
        "fp": (0.90, 0.12, 0.12, 0.75),
        "fn": (0.15, 0.35, 0.95, 0.75),
    }
    fig = plt.figure(figsize=(10, 16))
    grid = fig.add_gridspec(4, 2, left=0.05, right=0.98, top=0.94, bottom=0.03, hspace=0.22, wspace=0.04)
    for row, item in enumerate(prepared):
        for col, (pred, title) in enumerate(((item["pred2"], "2D U-Net"), (item["pred3"], "3D U-Net"))):
            ax = fig.add_subplot(grid[row, col])
            ax.imshow(item["mri"].T, cmap="gray", origin="lower", vmin=item["vmin"], vmax=item["vmax"], interpolation="nearest")
            overlay = np.zeros(item["gt"].T.shape + (4,), dtype=np.float32)
            gt = item["gt"].T
            pr = pred.T
            overlay[np.logical_and(pr, gt)] = colors["tp"]
            overlay[np.logical_and(pr, ~gt)] = colors["fp"]
            overlay[np.logical_and(~pr, gt)] = colors["fn"]
            ax.imshow(overlay, origin="lower", interpolation="nearest")
            ax.set_xticks([])
            ax.set_yticks([])
            ax.set_aspect("equal")
            for spine in ax.spines.values():
                spine.set_visible(False)
            if row == 0:
                ax.set_title(title)
            ax.set_xlabel(f"{item['case_id']}  Z {item['slice_index']}", fontsize=9)
    handles = [
        Patch(facecolor=colors["tp"], edgecolor="none", label="TP (prediction and GT)"),
        Patch(facecolor=colors["fp"], edgecolor="none", label="FP (prediction only)"),
        Patch(facecolor=colors["fn"], edgecolor="none", label="FN (GT only)"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False)
    _save(fig, fig_dir / "test_cases_error_overlay.png", dpi)


def _fmt_distance(value: float) -> str:
    if not np.isfinite(value):
        return "inf"
    return f"{value:.2f}"


def _mri_window(mri: np.ndarray, percentiles: list[float]) -> tuple[float, float]:
    values = mri[mri != 0]
    if values.size < 10:
        values = mri.reshape(-1)
    vmin = float(np.percentile(values, percentiles[0]))
    vmax = float(np.percentile(values, percentiles[1]))
    if vmax <= vmin:
        vmax = vmin + 1.0
    return vmin, vmax


def _save(fig, path: Path, dpi: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    fig.savefig(path.with_suffix(".pdf"))
    plt.close(fig)


if __name__ == "__main__":
    main()
