"""Case-level volumes, then axial slices. Splits stay the 3D case split.

Z is array axis 2. That is the NRRD Z axis (sizes are X, Y, Z), which the 3D
loader stores as (X, Y, Z). It is not inferred from "the last axis of an HDF5
array". This release has no HDF5 volumes.

Normalization is the cached per-case nonzero z-score already used by the 3D
run. Slices are cut after that normalization. One sample is one XY slice crop.
"""

from __future__ import annotations

import numpy as np
import torch
from torch.utils.data import Dataset

from la_seg3d.data import VolumeStore

# Documented by outputs/segmentation3d/data_audit.json axis_order and la_seg3d.data.AXIS_NOTE.
Z_AXIS = 2
AXIS_NOTE = {
    "storage": "NRRD, not HDF5. data_audit.json reports h5_file_count = 0.",
    "array_axes": "axis 0 = X, axis 1 = Y, axis 2 = Z",
    "z_axis_index": Z_AXIS,
    "slice_plane": "image[:, :, z] is one axial slice of shape (X, Y)",
    "why_not_last_dim_default": (
        "Axis 2 is Z because the 3D loader keeps NRRD sizes in (X, Y, Z) order. "
        "The last axis is Z only as a consequence of that order."
    ),
}


def store_config(cfg: dict) -> dict:
    """Keys VolumeStore reads. patch_size stays the 3D patch so the cache manifest matches."""
    data = cfg["data"]
    ref = cfg["reference_3d_data"]
    return {
        "seed": int(cfg["seed"]),
        "data": {
            "root": data["root"],
            "train_dirname": data["train_dirname"],
            "test_dirname": data["test_dirname"],
            "image_name": data["image_name"],
            "label_name": data["label_name"],
            "val_ratio": data["val_ratio"],
            "split_seed": data["split_seed"],
            "cache_dir": data["cache_dir"],
            "cache_workers": data.get("cache_workers", 8),
            "patch_size": ref["patch_size"],
            "fg_sample_prob": ref["fg_sample_prob"],
            "fg_center_jitter": ref["fg_center_jitter"],
            "patches_per_case_per_epoch": ref["patches_per_case_per_epoch"],
        },
    }


def open_store(cfg: dict, retain: str, keep_raw: bool) -> VolumeStore:
    store = VolumeStore(store_config(cfg))
    store.prepare(retain=retain, keep_raw=keep_raw)
    return store


def foreground_slice_indices(label_xyz: np.ndarray) -> np.ndarray:
    if label_xyz.ndim != 3:
        raise ValueError(f"expected [X,Y,Z], got {label_xyz.shape}")
    counts = label_xyz.reshape(-1, label_xyz.shape[Z_AXIS]).sum(axis=0)
    return np.flatnonzero(counts > 0).astype(np.int16)


def slice_manifest(store: VolumeStore) -> dict:
    rows = []
    for case_id, record in store.cases.items():
        label = record["label"]
        if label.shape[Z_AXIS] != record["shape_xyz"][Z_AXIS]:
            raise RuntimeError(f"{case_id} Z length does not match shape_xyz")
        fg = foreground_slice_indices(label)
        split = "test" if case_id in store.test_ids else "val" if case_id in store.val_ids else "train"
        rows.append(
            {
                "case_id": case_id,
                "split": split,
                "shape_xyz": list(record["shape_xyz"]),
                "z_axis": Z_AXIS,
                "n_slices": int(label.shape[Z_AXIS]),
                "fg_slice_indices": [int(v) for v in fg.tolist()],
                "n_background_slices": int(label.shape[Z_AXIS] - len(fg)),
            }
        )
    rows.sort(key=lambda row: (row["split"], row["case_id"]))
    return {"z_axis": Z_AXIS, "axis_note": AXIS_NOTE, "cases": rows}


class SliceCropDataset(Dataset):
    """Fixed number of XY crops per epoch. Each case contributes the same count.

    With probability `fg_sample_prob` the crop comes from a foreground Z slice
    and is centered on a foreground pixel, with the 3D XY jitter. Otherwise the
    slice is uniform over every Z index, including empty slices, and the crop
    is uniform. Image and label stay on the same grid. There is no resize.
    """

    def __init__(self, store: VolumeStore, case_ids: list[str], cfg: dict):
        self.store = store
        self.case_ids = list(case_ids)
        self.crop_size = tuple(int(v) for v in cfg["data"]["crop_size_xy"])
        self.fg_prob = float(cfg["data"]["fg_sample_prob"])
        self.jitter = tuple(int(v) for v in cfg["data"]["fg_center_jitter_xy"])
        self.samples_per_epoch = int(cfg["train"]["samples_per_epoch"])
        self.aug = cfg["augment"]
        self.seed = int(cfg["seed"])
        self.epoch = 0
        if len(self.crop_size) != 2 or len(self.jitter) != 2:
            raise ValueError("crop_size_xy and fg_center_jitter_xy must have length 2")
        if self.samples_per_epoch % len(self.case_ids) != 0:
            raise ValueError(
                f"samples_per_epoch {self.samples_per_epoch} is not divisible by "
                f"{len(self.case_ids)} train cases"
            )
        self.fg_slices = {
            case_id: foreground_slice_indices(store.cases[case_id]["label"]) for case_id in self.case_ids
        }

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return self.samples_per_epoch

    def __getitem__(self, index: int):
        case_id = self.case_ids[index % len(self.case_ids)]
        case = self.store.cases[case_id]
        rng = np.random.RandomState(self.seed + self.epoch * 100_003 + index)
        image, label, slice_index, used_fg, source_is_background = sample_slice_crop(
            case["image"],
            case["label"],
            self.fg_slices[case_id],
            self.crop_size,
            self.fg_prob,
            self.jitter,
            rng,
        )
        image, label = augment_crop(image, label, self.aug, rng)
        if image.shape != self.crop_size or label.shape != self.crop_size:
            raise RuntimeError(f"crop shape {image.shape} != {self.crop_size}")
        image_t = torch.from_numpy(np.ascontiguousarray(image[None], dtype=np.float32))
        label_t = torch.from_numpy(np.ascontiguousarray(label, dtype=np.int64))
        return {
            "image": image_t,
            "label": label_t,
            "case_id": case_id,
            "slice_index": int(slice_index),
            "shape_xyz": tuple(int(v) for v in case["shape_xyz"]),
            "used_foreground_sampler": bool(used_fg),
            "background_slice": bool(source_is_background),
        }


def sample_slice_crop(
    image_xyz: np.ndarray,
    label_xyz: np.ndarray,
    fg_slices: np.ndarray,
    crop_size: tuple[int, int],
    fg_prob: float,
    jitter: tuple[int, int],
    rng: np.random.RandomState,
) -> tuple[np.ndarray, np.ndarray, int, bool, bool]:
    n_slices = int(image_xyz.shape[Z_AXIS])
    use_fg = bool(rng.rand() < fg_prob and len(fg_slices))
    if use_fg:
        slice_index = int(fg_slices[int(rng.randint(len(fg_slices)))])
    else:
        slice_index = int(rng.randint(0, n_slices))
    source_is_background = slice_index not in set(int(v) for v in fg_slices.tolist())
    image = image_xyz[:, :, slice_index]
    label = label_xyz[:, :, slice_index]
    image, label, _pad = pad_slice(image, label, crop_size)
    if use_fg:
        fg = np.argwhere(label > 0)
        if len(fg) == 0:
            raise RuntimeError("foreground slice has no foreground pixels after padding")
        center = fg[int(rng.randint(len(fg)))]
        starts = []
        for center_i, patch, limit, jit in zip(center, crop_size, image.shape, jitter):
            offset = int(rng.randint(-jit, jit + 1)) if jit else 0
            start = int(center_i) - patch // 2 + offset
            starts.append(max(0, min(start, limit - patch)))
    else:
        starts = [int(rng.randint(0, limit - patch + 1)) for limit, patch in zip(image.shape, crop_size)]
    slices = tuple(slice(start, start + patch) for start, patch in zip(starts, crop_size))
    return image[slices].copy(), label[slices].copy(), slice_index, use_fg, source_is_background


def pad_slice(image: np.ndarray, label: np.ndarray, crop_size: tuple[int, int]):
    pad_after = [max(0, crop - size) for crop, size in zip(crop_size, image.shape)]
    if not any(pad_after):
        return image, label, pad_after
    pad_width = [(0, pad) for pad in pad_after]
    return (
        np.pad(image, pad_width, mode="constant", constant_values=0),
        np.pad(label, pad_width, mode="constant", constant_values=0),
        pad_after,
    )


def augment_crop(image: np.ndarray, label: np.ndarray, aug: dict, rng: np.random.RandomState):
    """In-plane flips, 90-degree XY rotation, and the 3D intensity perturbation.

    The 3D Z-axis flip has no pixels to flip inside one axial slice, so it is
    not applied. Spatial scale is not used. `scale_min` / `scale_max` multiply
    intensities, as in the 3D augmentation.
    """
    for axis in range(2):
        if rng.rand() < float(aug["flip_prob"]):
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis)
    if image.shape[0] == image.shape[1] and rng.rand() < float(aug["rot90_prob"]):
        k = int(rng.randint(1, 4))
        image = np.rot90(image, k=k, axes=(0, 1))
        label = np.rot90(label, k=k, axes=(0, 1))
    if image.shape != label.shape:
        raise RuntimeError("spatial augmentation desynchronized image and label")
    if rng.rand() < float(aug["intensity_prob"]):
        scale = rng.uniform(float(aug["scale_min"]), float(aug["scale_max"]))
        image = image * np.float32(scale)
        noise_std = float(aug["noise_std"])
        if noise_std > 0:
            image = image + rng.normal(0.0, noise_std, size=image.shape).astype(np.float32)
    return np.ascontiguousarray(image, dtype=np.float32), np.ascontiguousarray(label)


def collate_crops(samples: list[dict]) -> dict:
    return {
        "image": torch.stack([sample["image"] for sample in samples], dim=0),
        "label": torch.stack([sample["label"] for sample in samples], dim=0),
        "case_id": [sample["case_id"] for sample in samples],
        "slice_index": [sample["slice_index"] for sample in samples],
        "shape_xyz": [sample["shape_xyz"] for sample in samples],
        "used_foreground_sampler": [sample["used_foreground_sampler"] for sample in samples],
        "background_slice": [sample["background_slice"] for sample in samples],
    }
