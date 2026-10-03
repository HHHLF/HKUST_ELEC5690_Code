"""Case-level loading, audit, and random-patch sampling for the LA NRRD release."""

from __future__ import annotations

import json
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from la_seg3d.nrrd_io import read_nrrd

AXIS_NOTE = {
    "nrrd_sizes": "X Y Z, X fastest on disk",
    "array_and_model_axes": "axis 0 = X = D, axis 1 = Y = H, axis 2 = Z = W",
    "model_input": "[B, 1, D, H, W] = [B, 1, X, Y, Z]",
    "prediction_restore": "predictions stay in (X, Y, Z); no axis transpose is applied at save time",
}


def nonzero_zscore(image: np.ndarray) -> np.ndarray:
    """Z-score using voxels that are nonzero in the raw volume.

    An all-zero volume, or a nonzero region with zero variance, becomes zeros.
    """
    out = image.astype(np.float32, copy=True)
    mask = image != 0
    if not mask.any():
        return np.zeros_like(out)
    values = out[mask]
    std = float(values.std())
    if std < 1e-6:
        return np.zeros_like(out)
    mean = float(values.mean())
    out -= mean
    out /= std
    return out


CACHE_VERSION = 2


def cache_directory(cfg: dict) -> Path:
    configured = cfg["data"].get("cache_dir")
    if configured:
        return Path(configured)
    return Path("/dev/shm/elec5690_la_seg3d")


def _cache_one(task: tuple) -> dict:
    """Read one NRRD case and write the arrays that training reloads."""
    case_id, split_dir, folder, image_name, label_name, destination = task
    folder = Path(folder)
    destination = Path(destination)
    image, spacing, image_meta = read_nrrd(folder / image_name)
    label, label_spacing, _label_meta = read_nrrd(folder / label_name)
    if image.shape != label.shape:
        raise ValueError(f"{case_id} image shape {image.shape} != label shape {label.shape}")
    if spacing != label_spacing:
        raise ValueError(f"{case_id} image spacing {spacing} != label spacing {label_spacing}")
    label01 = map_label(label, case_id)
    destination.mkdir(parents=True, exist_ok=True)
    np.save(destination / "image_u8.npy", image)
    np.save(destination / "image_f32.npy", nonzero_zscore(image))
    np.save(destination / "label.npy", label01)
    np.save(destination / "fg.npy", np.argwhere(label01 > 0).astype(np.int16))
    return {
        "case_id": case_id,
        "split_dir": split_dir,
        "shape_xyz": list(image.shape),
        "spacing_xyz": list(spacing),
        "spacing_source": "NRRD space directions vector norms; header has no space units",
        "space": image_meta.get("space"),
        "space_units": image_meta.get("space units"),
        "image_min": int(image.min()),
        "image_max": int(image.max()),
        "label_values_raw": [int(v) for v in np.unique(label).tolist()],
        "foreground_voxels": int(label01.sum()),
    }


def _write_normalized(case_dir: str) -> str:
    folder = Path(case_dir)
    image = np.load(folder / "image_u8.npy")
    np.save(folder / "image_f32.npy", nonzero_zscore(image))
    return folder.name


def map_label(label: np.ndarray, case_id: str) -> np.ndarray:
    values = set(np.unique(label).tolist())
    if values <= {0, 1}:
        mapped = label.astype(np.uint8)
    elif values <= {0, 255}:
        mapped = (label == 255).astype(np.uint8)
    else:
        raise ValueError(f"{case_id} label values {sorted(values)} are not {{0,1}} or {{0,255}}")
    if int(mapped.sum()) == 0:
        raise ValueError(f"{case_id} left-atrium label is empty")
    return mapped


class VolumeStore:
    def __init__(self, cfg: dict):
        self.cfg = cfg
        self.root = Path(cfg["data"]["root"])
        self.image_name = cfg["data"]["image_name"]
        self.label_name = cfg["data"]["label_name"]
        self.cases: dict[str, dict] = {}
        self.train_ids: list[str] = []
        self.val_ids: list[str] = []
        self.test_ids: list[str] = []
        self.audit_cases: list[dict] = []

    def discover(self) -> None:
        h5_files = sorted(self.root.rglob("*.h5")) + sorted(self.root.rglob("*.hdf5"))
        if h5_files:
            raise RuntimeError(
                "HDF5 files were found, but this loader was built from the NRRD release. "
                f"First files: {[str(p) for p in h5_files[:5]]}"
            )
        train_ids = self._case_ids(self.cfg["data"]["train_dirname"])
        test_ids = self._case_ids(self.cfg["data"]["test_dirname"])
        overlap = sorted(set(train_ids) & set(test_ids))
        if overlap:
            raise RuntimeError(f"case ids appear in both official splits: {overlap[:10]}")
        self.test_ids = test_ids
        self.train_ids, self.val_ids = split_train_val(
            train_ids,
            val_ratio=float(self.cfg["data"]["val_ratio"]),
            seed=int(self.cfg["data"]["split_seed"]),
        )

    def _case_ids(self, dirname: str) -> list[str]:
        folder = self.root / dirname
        if not folder.is_dir():
            raise FileNotFoundError(folder)
        ids = []
        for path in sorted(folder.iterdir()):
            if path.is_dir() and (path / self.image_name).is_file() and (path / self.label_name).is_file():
                ids.append(path.name)
        if not ids:
            raise RuntimeError(f"no cases with {self.image_name} and {self.label_name} in {folder}")
        return ids

    def case_dir(self, case_id: str) -> Path:
        if case_id in self.test_ids:
            return self.root / self.cfg["data"]["test_dirname"] / case_id
        return self.root / self.cfg["data"]["train_dirname"] / case_id

    def prepare(self, retain: str, keep_raw: bool) -> None:
        """Load the prepared cache. Build it first if this machine has none."""
        self.discover()
        if retain == "train_val":
            keep = set(self.train_ids + self.val_ids)
        elif retain == "test":
            keep = set(self.test_ids)
        elif retain == "all":
            keep = set(self.train_ids + self.val_ids + self.test_ids)
        else:
            raise ValueError(f"unknown retain={retain}")
        cache_dir = cache_directory(self.cfg)
        if not self.cache_is_ready(cache_dir):
            if self._can_upgrade_cache(cache_dir):
                print(f"adding normalized volumes to {cache_dir}", flush=True)
                self.upgrade_cache(cache_dir)
            else:
                print(f"no prepared cache at {cache_dir}; building it now", flush=True)
                self.build_cache(cache_dir)
        self.load_cache(cache_dir, keep, keep_raw)

    def cache_is_ready(self, cache_dir: Path) -> bool:
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.is_file():
            return False
        manifest = json.loads(manifest_path.read_text())
        expected = set(self.train_ids + self.val_ids + self.test_ids)
        cached = {row["case_id"] for row in manifest.get("cases", [])}
        if manifest.get("version") != CACHE_VERSION or cached != expected:
            return False
        if manifest.get("dataset_root") != str(self.root):
            return False
        return all((cache_dir / case_id / "image_f32.npy").is_file() for case_id in expected)

    def _can_upgrade_cache(self, cache_dir: Path) -> bool:
        manifest_path = cache_dir / "manifest.json"
        if not manifest_path.is_file():
            return False
        manifest = json.loads(manifest_path.read_text())
        expected = set(self.train_ids + self.val_ids + self.test_ids)
        cached = {row["case_id"] for row in manifest.get("cases", [])}
        if cached != expected or manifest.get("dataset_root") != str(self.root):
            return False
        return all((cache_dir / case_id / "image_u8.npy").is_file() for case_id in expected)

    def upgrade_cache(self, cache_dir: Path) -> None:
        case_dirs = [str(cache_dir / case_id) for case_id in self.train_ids + self.val_ids + self.test_ids]
        workers = int(self.cfg["data"].get("cache_workers", min(8, os.cpu_count() or 1)))
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_write_normalized, case_dir) for case_dir in case_dirs]
            for future in as_completed(futures):
                future.result()
                done += 1
                if done % 20 == 0 or done == len(case_dirs):
                    print(f"normalized {done}/{len(case_dirs)} volumes", flush=True)
        manifest_path = cache_dir / "manifest.json"
        manifest = json.loads(manifest_path.read_text())
        manifest["version"] = CACHE_VERSION
        temporary = cache_dir / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest) + "\n")
        temporary.replace(manifest_path)

    def build_cache(self, cache_dir: Path | None = None) -> None:
        """Read every NRRD once and write arrays onto the cache directory."""
        if not self.train_ids:
            self.discover()
        cache_dir = cache_dir or cache_directory(self.cfg)
        cache_dir.mkdir(parents=True, exist_ok=True)
        tasks = []
        for case_id in self.train_ids + self.val_ids + self.test_ids:
            split_dir = "Testing Set" if case_id in self.test_ids else "Training Set"
            tasks.append(
                (
                    case_id,
                    split_dir,
                    str(self.case_dir(case_id)),
                    self.image_name,
                    self.label_name,
                    str(cache_dir / case_id),
                )
            )
        workers = int(self.cfg["data"].get("cache_workers", min(8, os.cpu_count() or 1)))
        metas = []
        done = 0
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = [pool.submit(_cache_one, task) for task in tasks]
            for future in as_completed(futures):
                metas.append(future.result())
                done += 1
                if done % 20 == 0 or done == len(tasks):
                    print(f"cached {done}/{len(tasks)} volumes", flush=True)
        order = {case_id: index for index, case_id in enumerate(self.train_ids + self.val_ids + self.test_ids)}
        metas.sort(key=lambda row: order[row["case_id"]])
        manifest = {
            "version": CACHE_VERSION,
            "dataset_root": str(self.root),
            "image_name": self.image_name,
            "label_name": self.label_name,
            "cases": metas,
        }
        temporary = cache_dir / "manifest.json.tmp"
        temporary.write_text(json.dumps(manifest) + "\n")
        temporary.replace(cache_dir / "manifest.json")
        print(f"cache ready at {cache_dir}", flush=True)

    def load_cache(self, cache_dir: Path, keep: set[str], keep_raw: bool) -> None:
        manifest = json.loads((cache_dir / "manifest.json").read_text())
        self.audit_cases = list(manifest["cases"])
        self.cases = {}
        kept = [row for row in self.audit_cases if row["case_id"] in keep]
        for index, row in enumerate(kept, start=1):
            case_dir = cache_dir / row["case_id"]
            record = dict(row)
            record["image"] = np.load(case_dir / "image_f32.npy")
            record["label"] = np.load(case_dir / "label.npy")
            record["fg_coords"] = np.load(case_dir / "fg.npy")
            if keep_raw:
                record["image_raw"] = np.load(case_dir / "image_u8.npy")
            self.cases[row["case_id"]] = record
            if index % 20 == 0 or index == len(kept):
                print(f"loaded cache {index}/{len(kept)} volumes from {cache_dir}", flush=True)

    def _load_case(self, case_id: str) -> dict:
        folder = self.case_dir(case_id)
        image, spacing, image_meta = read_nrrd(folder / self.image_name)
        label, label_spacing, label_meta = read_nrrd(folder / self.label_name)
        if image.shape != label.shape:
            raise ValueError(f"{case_id} image shape {image.shape} != label shape {label.shape}")
        if spacing != label_spacing:
            raise ValueError(f"{case_id} image spacing {spacing} != label spacing {label_spacing}")
        label01 = map_label(label, case_id)
        fg_coords = np.argwhere(label01 > 0).astype(np.int16)
        record = {
            "case_id": case_id,
            "split_dir": "Testing Set" if case_id in self.test_ids else "Training Set",
            "shape_xyz": list(image.shape),
            "spacing_xyz": list(spacing),
            "spacing_source": "NRRD space directions vector norms; header has no space units",
            "space": image_meta.get("space"),
            "space_units": image_meta.get("space units"),
            "image_min": int(image.min()),
            "image_max": int(image.max()),
            "label_values_raw": [int(v) for v in np.unique(label).tolist()],
            "foreground_voxels": int(label01.sum()),
            "image": nonzero_zscore(image),
            "image_raw": image,
            "label": label01,
            "fg_coords": fg_coords,
        }
        self.audit_cases.append(
            {k: v for k, v in record.items() if k not in {"image", "image_raw", "label", "fg_coords"}}
        )
        return record

    def audit_payload(self) -> dict:
        shapes = {}
        spacings = {}
        for row in self.audit_cases:
            shapes[str(row["shape_xyz"])] = shapes.get(str(row["shape_xyz"]), 0) + 1
            spacings[str(row["spacing_xyz"])] = spacings.get(str(row["spacing_xyz"]), 0) + 1
        return {
            "dataset_root": str(self.root),
            "h5_file_count": 0,
            "h5_note": (
                "A recursive search found no .h5 or .hdf5 volumes. "
                "The local release is NRRD (lgemri.nrrd image, laendo.nrrd cavity label). "
                "lawall.nrrd is the wall and is not the binary cavity target."
            ),
            "official_split": {
                "train_dirname": self.cfg["data"]["train_dirname"],
                "test_dirname": self.cfg["data"]["test_dirname"],
            },
            "axis_order": AXIS_NOTE,
            "patch_size_dhw": self.cfg["data"]["patch_size"],
            "patch_axis_order": "patch_size is [D, H, W] = [X, Y, Z]",
            "normalization": "per-case z-score of nonzero raw voxels; all-zero or zero-variance nonzero region -> zeros",
            "resample": "none; geometry is the original voxel grid",
            "distance_unit": "voxel",
            "spacing_note": (
                "Every inspected header has space directions (1,0,0) (0,1,0) (0,0,1) and no space units. "
                "ASD and HD95 are reported in voxels. The challenge paper's 0.625 mm figure is not encoded "
                "in this copy and is not applied."
            ),
            "shape_counts": shapes,
            "spacing_counts": spacings,
            "n_cases_loaded_for_audit": len(self.audit_cases),
            "cases": self.audit_cases,
        }

    def split_payload(self) -> dict:
        train, val, test = set(self.train_ids), set(self.val_ids), set(self.test_ids)
        return {
            "seed": int(self.cfg["data"]["split_seed"]),
            "val_ratio": float(self.cfg["data"]["val_ratio"]),
            "policy": (
                "Official Training Set and Testing Set directories are kept. "
                "The validation set is a case-level holdout from the Training Set only. "
                "Patches from one case never cross splits. The test set is not used to "
                "choose checkpoints, hyperparameters, or preprocessing."
            ),
            "counts": {"train": len(self.train_ids), "val": len(self.val_ids), "test": len(self.test_ids)},
            "train_ids": self.train_ids,
            "val_ids": self.val_ids,
            "test_ids": self.test_ids,
            "overlap": {
                "train_val": sorted(train & val),
                "train_test": sorted(train & test),
                "val_test": sorted(val & test),
                "has_overlap": bool(train & val or train & test or val & test),
            },
        }


def split_train_val(case_ids: list[str], val_ratio: float, seed: int) -> tuple[list[str], list[str]]:
    ids = sorted(case_ids)
    rng = np.random.RandomState(seed)
    order = rng.permutation(len(ids))
    n_val = int(round(len(ids) * val_ratio))
    if n_val <= 0 or n_val >= len(ids):
        raise ValueError(f"val split size {n_val} is invalid for {len(ids)} cases")
    val_index = set(order[:n_val].tolist())
    val_ids = [ids[i] for i in range(len(ids)) if i in val_index]
    train_ids = [ids[i] for i in range(len(ids)) if i not in val_index]
    return train_ids, val_ids


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2) + "\n")


class PatchDataset(Dataset):
    def __init__(self, store: VolumeStore, case_ids: list[str], training: bool):
        self.store = store
        self.case_ids = list(case_ids)
        self.training = training
        data_cfg = store.cfg["data"]
        self.patch_size = tuple(int(v) for v in data_cfg["patch_size"])
        self.fg_prob = float(data_cfg["fg_sample_prob"])
        self.patches_per_case = int(data_cfg["patches_per_case_per_epoch"])
        self.jitter = tuple(int(v) for v in data_cfg["fg_center_jitter"])
        self.aug = store.cfg["augment"]
        self.seed = int(store.cfg["seed"])
        self.epoch = 0
        if len(self.patch_size) != 3 or len(self.jitter) != 3:
            raise ValueError("patch_size and fg_center_jitter must have length 3")

    def set_epoch(self, epoch: int) -> None:
        self.epoch = int(epoch)

    def __len__(self) -> int:
        return len(self.case_ids) * self.patches_per_case

    def __getitem__(self, index: int):
        case = self.store.cases[self.case_ids[index % len(self.case_ids)]]
        rng = np.random.RandomState(self.seed + self.epoch * 100_003 + index)
        image, label = sample_patch(
            case["image"],
            case["label"],
            self.patch_size,
            self.fg_prob,
            self.jitter,
            rng,
            fg_coords=case["fg_coords"],
        )
        if self.training:
            image, label = augment_patch(image, label, self.aug, rng)
        if image.shape != self.patch_size or label.shape != self.patch_size:
            raise RuntimeError(f"patch shape {image.shape} != {self.patch_size}")
        image_t = torch.from_numpy(np.ascontiguousarray(image[None], dtype=np.float32))
        label_t = torch.from_numpy(np.ascontiguousarray(label, dtype=np.int64))
        return image_t, label_t


def sample_patch(
    image: np.ndarray,
    label: np.ndarray,
    patch_size: tuple[int, int, int],
    fg_prob: float,
    jitter: tuple[int, int, int],
    rng: np.random.RandomState,
    fg_coords: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    image, label, _pads = pad_volume(image, label, patch_size)
    shape = image.shape
    use_fg = rng.rand() < fg_prob
    if use_fg:
        fg = fg_coords if fg_coords is not None else np.argwhere(label > 0)
    else:
        fg = np.empty((0, 3), dtype=np.int64)
    if len(fg):
        center = fg[rng.randint(len(fg))]
        starts = []
        for axis, (center_i, patch, limit, jit) in enumerate(
            zip(center, patch_size, shape, jitter)
        ):
            offset = int(rng.randint(-jit, jit + 1)) if jit else 0
            start = int(center_i) - patch // 2 + offset
            start = max(0, min(start, limit - patch))
            starts.append(start)
    else:
        starts = [int(rng.randint(0, limit - patch + 1)) for limit, patch in zip(shape, patch_size)]
    slices = tuple(slice(start, start + patch) for start, patch in zip(starts, patch_size))
    return image[slices].copy(), label[slices].copy()


def pad_volume(image: np.ndarray, label: np.ndarray, patch_size: tuple[int, int, int]):
    pad_after = [max(0, patch - size) for patch, size in zip(patch_size, image.shape)]
    if not any(pad_after):
        return image, label, pad_after
    pad_width = [(0, pad) for pad in pad_after]
    return (
        np.pad(image, pad_width, mode="constant", constant_values=0),
        np.pad(label, pad_width, mode="constant", constant_values=0),
        pad_after,
    )


def augment_patch(image: np.ndarray, label: np.ndarray, aug: dict, rng: np.random.RandomState):
    for axis in range(3):
        if rng.rand() < float(aug["flip_prob"]):
            image = np.flip(image, axis=axis)
            label = np.flip(label, axis=axis)
    # 90-degree rotation only in the X-Y plane, and only when those two axes
    # have the same length. The default patch is 112 x 112 x 80, so a rotation
    # that involves Z would swap 112 and 80.
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
