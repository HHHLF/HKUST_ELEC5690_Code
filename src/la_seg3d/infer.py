"""Sliding-window inference with Gaussian fusion of softmax probabilities."""

from __future__ import annotations

import numpy as np
import torch


def gaussian_weight(patch_size: list[int] | tuple[int, int, int], sigma_scale: float) -> np.ndarray:
    axes = [np.arange(size, dtype=np.float32) - (size - 1) / 2.0 for size in patch_size]
    grid = np.meshgrid(*axes, indexing="ij")
    sigma = [max(size * sigma_scale, 1.0) for size in patch_size]
    exponent = sum((coord ** 2) / (2.0 * sig ** 2) for coord, sig in zip(grid, sigma))
    weight = np.exp(-exponent).astype(np.float32)
    weight /= weight.max()
    weight = np.clip(weight, 1e-3, None)
    return weight


def _starts(length: int, patch: int, stride: int) -> list[int]:
    if length <= patch:
        return [0]
    starts = list(range(0, length - patch + 1, stride))
    last = length - patch
    if starts[-1] != last:
        starts.append(last)
    return starts


@torch.inference_mode()
def predict_volume(
    model: torch.nn.Module,
    image: np.ndarray,
    patch_size: list[int],
    overlap: float,
    sw_batch_size: int,
    device: torch.device,
    amp: bool,
    sigma_scale: float,
) -> tuple[np.ndarray, dict]:
    """Predict a full volume. `image` is float32 [D, H, W] in model axis order.

    Volumes smaller than the patch are zero-padded on the trailing side of each
    axis and the prediction is cropped back to the original shape.
    """
    if image.ndim != 3:
        raise ValueError(f"expected [D,H,W], got {image.shape}")
    original_shape = tuple(int(v) for v in image.shape)
    pad_after = [max(0, patch - size) for patch, size in zip(patch_size, original_shape)]
    padded = image
    if any(pad_after):
        padded = np.pad(
            image,
            [(0, pad) for pad in pad_after],
            mode="constant",
            constant_values=0,
        )
    stride = [max(1, int(round(patch * (1.0 - overlap)))) for patch in patch_size]
    starts = [_starts(padded.shape[axis], patch_size[axis], stride[axis]) for axis in range(3)]
    coords = [(d, h, w) for d in starts[0] for h in starts[1] for w in starts[2]]
    weight = torch.from_numpy(gaussian_weight(patch_size, sigma_scale)).to(device)
    pd, ph, pw = patch_size
    volume = torch.from_numpy(np.ascontiguousarray(padded)).to(device)
    acc = torch.zeros((2,) + tuple(padded.shape), dtype=torch.float32, device=device)
    weight_sum = torch.zeros(padded.shape, dtype=torch.float32, device=device)
    model.eval()
    for begin in range(0, len(coords), sw_batch_size):
        batch_coords = coords[begin : begin + sw_batch_size]
        patches = torch.stack(
            [volume[d : d + pd, h : h + ph, w : w + pw] for d, h, w in batch_coords]
        ).unsqueeze(1)
        with torch.autocast(device_type=device.type, enabled=amp and device.type == "cuda"):
            logits = model(patches)
        if logits.shape[2:] != tuple(patch_size):
            raise RuntimeError(
                f"network spatial output {tuple(logits.shape[2:])} != patch {tuple(patch_size)}"
            )
        probs = torch.softmax(logits.float(), dim=1)
        for index, (d, h, w) in enumerate(batch_coords):
            acc[:, d : d + pd, h : h + ph, w : w + pw] += probs[index] * weight
            weight_sum[d : d + pd, h : h + ph, w : w + pw] += weight
    if torch.any(weight_sum <= 0):
        raise RuntimeError("sliding window left some voxels with zero weight")
    acc /= weight_sum
    prediction = torch.argmax(acc, dim=0).to(dtype=torch.uint8).cpu().numpy()
    prediction = prediction[: original_shape[0], : original_shape[1], : original_shape[2]]
    info = {
        "original_shape": list(original_shape),
        "padded_shape": list(padded.shape),
        "pad_after": pad_after,
        "pad_mode": "constant 0 on the trailing side of each axis; crop back to original shape",
        "n_windows": len(coords),
        "overlap": overlap,
        "stride": stride,
    }
    return prediction, info
