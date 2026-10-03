"""Full-volume binary segmentation metrics.

Surface distances follow the medpy definition: the surface is the set of
voxels removed by one binary erosion with a 6-neighborhood (connectivity 1).
Distances are Euclidean distances from each surface to the other surface via
`scipy.ndimage.distance_transform_edt`. ASD is the mean of the two directions
after concatenation. HD95 is the 95th percentile of that same concatenated set.
Spacing is in array-axis order (X, Y, Z) = (D, H, W).
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, generate_binary_structure


def _surface(mask: np.ndarray) -> np.ndarray:
    footprint = generate_binary_structure(rank=3, connectivity=1)
    eroded = binary_erosion(mask, structure=footprint, iterations=1, border_value=0)
    return np.logical_xor(mask, eroded)


def surface_distances(
    source: np.ndarray,
    target: np.ndarray,
    spacing: tuple[float, float, float],
) -> np.ndarray:
    """Distances from the source surface to the target surface."""
    source_surface = _surface(source)
    target_surface = _surface(target)
    if not source_surface.any() or not target_surface.any():
        raise ValueError("surface distance requested for an empty surface")
    # EDT returns distance to the nearest zero. Zeros are the target surface.
    distances = distance_transform_edt(~target_surface, sampling=spacing)
    return distances[source_surface].astype(np.float64)


def binary_segmentation_metrics(
    prediction: np.ndarray,
    ground_truth: np.ndarray,
    spacing: tuple[float, float, float],
) -> dict:
    pred = np.asarray(prediction).astype(bool)
    gt = np.asarray(ground_truth).astype(bool)
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch {pred.shape} vs {gt.shape}")
    if pred.ndim != 3:
        raise ValueError(f"expected a 3D mask, got ndim={pred.ndim}")

    pred_empty = not bool(pred.any())
    gt_empty = not bool(gt.any())
    intersection = int(np.logical_and(pred, gt).sum())
    pred_sum = int(pred.sum())
    gt_sum = int(gt.sum())
    union = pred_sum + gt_sum - intersection

    if pred_empty and gt_empty:
        dice, jaccard, asd, hd95 = 1.0, 1.0, 0.0, 0.0
        empty_status = "both_empty"
    elif pred_empty or gt_empty:
        dice, jaccard, asd, hd95 = 0.0, 0.0, float("inf"), float("inf")
        empty_status = "pred_empty" if pred_empty else "gt_empty"
    else:
        dice = (2.0 * intersection) / (pred_sum + gt_sum)
        jaccard = intersection / union if union else 0.0
        distances = np.concatenate(
            [
                surface_distances(pred, gt, spacing),
                surface_distances(gt, pred, spacing),
            ]
        )
        asd = float(np.mean(distances))
        hd95 = float(np.percentile(distances, 95))
        empty_status = "none"

    return {
        "dice": float(dice),
        "jaccard": float(jaccard),
        "asd": float(asd),
        "hd95": float(hd95),
        "pred_foreground_voxels": pred_sum,
        "gt_foreground_voxels": gt_sum,
        "empty_status": empty_status,
    }
