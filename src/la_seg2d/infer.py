"""Full axial-slice inference, then a stack back to the original (X, Y, Z) volume.

A slice no larger than the training crop (and `full_slice_max_side`) is padded
on the trailing side if needed, run as one image, and cropped back. Larger
slices use overlapping windows of the training crop and Gaussian fusion of
softmax probabilities, the same fusion rule as the 3D sliding window. No
connected-component or other post-processing is applied.
"""

from __future__ import annotations

import numpy as np
import torch


def gaussian_weight(patch_size: tuple[int, int], sigma_scale: float) -> np.ndarray:
    axes = [np.arange(size, dtype=np.float32) - (size - 1) / 2.0 for size in patch_size]
    grid = np.meshgrid(*axes, indexing="ij")
    sigma = [max(size * sigma_scale, 1.0) for size in patch_size]
    exponent = sum((coord ** 2) / (2.0 * sig ** 2) for coord, sig in zip(grid, sigma))
    weight = np.exp(-exponent).astype(np.float32)
    weight /= weight.max()
    return np.clip(weight, 1e-3, None)


def _starts(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


def _argmax_probs(logits: torch.Tensor) -> torch.Tensor:
    return torch.argmax(torch.softmax(logits.float(), dim=1), dim=1).to(dtype=torch.uint8)


@torch.inference_mode()
def predict_slice_full(
    model: torch.nn.Module,
    image_xy: np.ndarray,
    device: torch.device,
    amp: bool,
) -> np.ndarray:
    """One XY slice. Trailing zero-pad is only applied when a caller pads first."""
    if image_xy.ndim != 2:
        raise ValueError(f"expected [X,Y], got {image_xy.shape}")
    batch = torch.from_numpy(np.ascontiguousarray(image_xy[None, None], dtype=np.float32)).to(device)
    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        logits = model(batch)
    if tuple(logits.shape[2:]) != tuple(image_xy.shape):
        raise RuntimeError(f"slice logits {tuple(logits.shape)} != input {image_xy.shape}")
    return _argmax_probs(logits)[0].cpu().numpy()


@torch.inference_mode()
def predict_slices_batched(
    model: torch.nn.Module,
    slices_xy: np.ndarray,
    device: torch.device,
    amp: bool,
) -> np.ndarray:
    """`slices_xy` is [N, X, Y]. Returns uint8 [N, X, Y]."""
    if slices_xy.ndim != 3:
        raise ValueError(f"expected [N,X,Y], got {slices_xy.shape}")
    batch = torch.from_numpy(np.ascontiguousarray(slices_xy[:, None], dtype=np.float32)).to(device)
    with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
        logits = model(batch)
    if tuple(logits.shape[2:]) != tuple(slices_xy.shape[1:]):
        raise RuntimeError(f"batched logits {tuple(logits.shape)} != input {slices_xy.shape}")
    return _argmax_probs(logits).cpu().numpy()


@torch.inference_mode()
def predict_slice_sliding(
    model: torch.nn.Module,
    image_xy: np.ndarray,
    patch_size: tuple[int, int],
    overlap: float,
    sw_batch_size: int,
    device: torch.device,
    amp: bool,
    sigma_scale: float,
) -> tuple[np.ndarray, dict]:
    original = tuple(int(v) for v in image_xy.shape)
    pad_after = [max(0, patch - size) for patch, size in zip(patch_size, original)]
    padded = image_xy
    if any(pad_after):
        padded = np.pad(image_xy, [(0, pad) for pad in pad_after], mode="constant", constant_values=0)
    stride = [max(1, int(round(patch * (1.0 - overlap)))) for patch in patch_size]
    starts = [_starts(padded.shape[axis], patch_size[axis], stride[axis]) for axis in range(2)]
    coords = [(y0, x0) for y0 in starts[0] for x0 in starts[1]]
    weight = torch.from_numpy(gaussian_weight(patch_size, sigma_scale)).to(device)
    ph, pw = patch_size
    plane = torch.from_numpy(np.ascontiguousarray(padded)).to(device)
    acc = torch.zeros((2,) + tuple(padded.shape), dtype=torch.float32, device=device)
    weight_sum = torch.zeros(padded.shape, dtype=torch.float32, device=device)
    model.eval()
    for begin in range(0, len(coords), sw_batch_size):
        batch_coords = coords[begin : begin + sw_batch_size]
        patches = torch.stack([plane[y : y + ph, x : x + pw] for y, x in batch_coords]).unsqueeze(1)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(patches)
        if tuple(logits.shape[2:]) != patch_size:
            raise RuntimeError(f"window output {tuple(logits.shape[2:])} != patch {patch_size}")
        probs = torch.softmax(logits.float(), dim=1)
        for index, (y, x) in enumerate(batch_coords):
            acc[:, y : y + ph, x : x + pw] += probs[index] * weight
            weight_sum[y : y + ph, x : x + pw] += weight
    if torch.any(weight_sum <= 0):
        raise RuntimeError("sliding window left some pixels with zero weight")
    acc /= weight_sum
    prediction = torch.argmax(acc, dim=0).to(dtype=torch.uint8).cpu().numpy()
    prediction = prediction[: original[0], : original[1]]
    info = {"n_windows": len(coords), "pad_after_xy": pad_after, "stride_xy": stride}
    return prediction, info


@torch.inference_mode()
def predict_volume_from_slices(
    model: torch.nn.Module,
    image_xyz: np.ndarray,
    z_axis: int,
    crop_size: tuple[int, int],
    full_slice_max_side: int,
    overlap: float,
    sw_batch_size: int,
    slice_batch_size: int,
    device: torch.device,
    amp: bool,
    sigma_scale: float,
) -> tuple[np.ndarray, dict]:
    """Predict every Z slice and stack on the original Z axis.

    `image_xyz` is the case-normalized volume in (X, Y, Z) order. `z_axis` must
    be 2 for this dataset. Empty slices are predicted; nothing is dropped by the
    ground-truth mask.
    """
    if z_axis != 2:
        raise ValueError(f"this dataset stores Z on axis 2, got z_axis={z_axis}")
    if image_xyz.ndim != 3:
        raise ValueError(f"expected [X,Y,Z], got {image_xyz.shape}")
    original = tuple(int(v) for v in image_xyz.shape)
    height, width, n_slices = original
    prediction = np.zeros(original, dtype=np.uint8)
    # InstanceNorm uses the statistics of the tensor it sees. Training crops are
    # crop_size, so a full 576 or 640 slice is a different input even though the
    # convolutions are fully convolutional. Windows stay at the training crop.
    trained_side = max(int(crop_size[0]), int(crop_size[1]))
    use_full = max(height, width) <= min(trained_side, int(full_slice_max_side))
    window_count = 0
    pad_after = [0, 0]
    if use_full:
        for begin in range(0, n_slices, slice_batch_size):
            end = min(n_slices, begin + slice_batch_size)
            batch = np.stack([image_xyz[:, :, index] for index in range(begin, end)], axis=0)
            pred = predict_slices_batched(model, batch, device, amp)
            prediction[:, :, begin:end] = np.transpose(pred, (1, 2, 0))
        mode = "full_slice"
    else:
        mode = "sliding_window"
        for index in range(n_slices):
            pred, info = predict_slice_sliding(
                model,
                image_xyz[:, :, index],
                crop_size,
                overlap,
                sw_batch_size,
                device,
                amp,
                sigma_scale,
            )
            prediction[:, :, index] = pred
            window_count += info["n_windows"]
            pad_after = info["pad_after_xy"]
    if prediction.shape != original:
        raise RuntimeError(f"reconstructed {prediction.shape} != original {original}")
    info = {
        "original_shape_xyz": list(original),
        "z_axis": z_axis,
        "n_slices": n_slices,
        "mode": mode,
        "n_windows": window_count if mode == "sliding_window" else n_slices,
        "pad_after_xy": pad_after,
        "postprocess": "none",
    }
    return prediction, info
