"""ISIC 2018 Task 3 reader.

Supports an image directory plus a CSV. Column names are detected from the file.
The official release used here is one-hot columns MEL, NV, BCC, AKIEC, BKL, DF, VASC
and an ``image`` id column. Images are ``{id}.jpg``.
"""

from __future__ import annotations

import json
import random
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from torchvision.transforms import functional as TF

from isic_baseline import CLASS_NAMES, EXPECTED_SPLIT_SIZES, IMAGENET_MEAN, IMAGENET_STD

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}
ID_COLUMN_CANDIDATES = ("image", "image_id", "img", "id", "filename", "file")
LABEL_COLUMN_CANDIDATES = ("label", "diagnosis", "class", "dx", "category", "target")


def normalize_id(value) -> str:
    text = str(value).strip()
    return Path(text).stem


class ISICDataset(Dataset):
    def __init__(self, records: list[dict], image_size: int, train: bool) -> None:
        self.records = records
        self.image_size = image_size
        self.train = train

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, index: int):
        record = self.records[index]
        with Image.open(record["path"]) as image:
            image = image.convert("RGB")
            image = TF.resize(image, [self.image_size, self.image_size], interpolation=TF.InterpolationMode.BILINEAR)
            if self.train and random.random() < 0.5:
                image = TF.hflip(image)
            if self.train:
                image = TF.adjust_brightness(image, 1.0 + (random.random() - 0.5) * 0.2)
                image = TF.adjust_contrast(image, 1.0 + (random.random() - 0.5) * 0.2)
            tensor = TF.to_tensor(image)
            tensor = TF.normalize(tensor, IMAGENET_MEAN, IMAGENET_STD)
        return tensor, int(record["label"]), record["image_id"]


def _index_images(image_dir: Path) -> tuple[dict[str, Path], list[str]]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory does not exist: {image_dir}")
    by_id: dict[str, Path] = {}
    ignored = []
    duplicates = []
    for path in sorted(image_dir.iterdir()):
        if not path.is_file():
            continue
        if path.suffix.lower() not in IMAGE_EXTENSIONS:
            ignored.append(path.name)
            continue
        image_id = path.stem
        if image_id in by_id:
            duplicates.append(image_id)
        by_id[image_id] = path
    if duplicates:
        raise RuntimeError(f"Duplicate image stems in {image_dir}: {duplicates[:10]}")
    return by_id, ignored


def _labels_from_frame(frame: pd.DataFrame) -> tuple[pd.Series, np.ndarray, str]:
    columns = {column.lower(): column for column in frame.columns}
    class_columns = []
    for name in CLASS_NAMES:
        if name.lower() in columns:
            class_columns.append(columns[name.lower()])
    if len(class_columns) == len(CLASS_NAMES):
        values = frame[class_columns].apply(pd.to_numeric, errors="coerce").to_numpy(dtype=np.float64)
        if np.isnan(values).any():
            raise RuntimeError("One-hot label columns contain non-numeric values.")
        row_sum = values.sum(axis=1)
        bad = np.where(~np.isclose(row_sum, 1.0, atol=1e-3))[0]
        if len(bad):
            sample = frame.iloc[bad[:5], 0].astype(str).tolist()
            raise RuntimeError(
                f"{len(bad)} CSV rows are not a single one-hot label among {CLASS_NAMES}. "
                f"Example ids: {sample}. These rows were not dropped silently."
            )
        labels = values.argmax(axis=1).astype(np.int64)
        return frame.index.to_series(), labels, "one-hot:" + ",".join(class_columns)

    label_column = next((columns[name] for name in LABEL_COLUMN_CANDIDATES if name in columns), None)
    if label_column is None:
        raise RuntimeError(
            f"Cannot find 7 one-hot columns {CLASS_NAMES} or a label column in {list(frame.columns)}."
        )
    mapping = {name.lower(): index for index, name in enumerate(CLASS_NAMES)}
    raw = frame[label_column].astype(str).str.strip()
    unknown = sorted({value for value in raw.unique() if value.lower() not in mapping and not value.isdigit()})
    if unknown:
        raise RuntimeError(f"Labels outside {CLASS_NAMES}: {unknown}. These rows were not dropped silently.")
    labels = []
    for value in raw.tolist():
        if value.lower() in mapping:
            labels.append(mapping[value.lower()])
        else:
            index = int(float(value))
            if index < 0 or index >= len(CLASS_NAMES):
                raise RuntimeError(f"Label index {index} is outside 0..{len(CLASS_NAMES) - 1}.")
            labels.append(index)
    return frame.index.to_series(), np.asarray(labels, dtype=np.int64), f"column:{label_column}"


def load_split(split: str, image_dir, csv_path) -> dict:
    image_dir = Path(image_dir)
    csv_path = Path(csv_path)
    if not csv_path.is_file():
        raise FileNotFoundError(f"Label CSV does not exist: {csv_path}")
    frame = pd.read_csv(csv_path, encoding="utf-8-sig")
    if frame.empty:
        raise RuntimeError(f"{csv_path} has no rows.")
    columns = {column.lower(): column for column in frame.columns}
    id_column = next((columns[name] for name in ID_COLUMN_CANDIDATES if name in columns), frame.columns[0])
    ids = frame[id_column].map(normalize_id)
    duplicate_ids = ids[ids.duplicated()].tolist()
    if duplicate_ids:
        raise RuntimeError(f"Duplicate image ids in {csv_path}: {duplicate_ids[:10]}")
    _, labels, label_mode = _labels_from_frame(frame)
    images, ignored = _index_images(image_dir)
    missing = [image_id for image_id in ids.tolist() if image_id not in images]
    used = set(ids.tolist())
    extra = sorted(set(images).difference(used))
    if missing:
        raise RuntimeError(
            f"{split}: {len(missing)} labeled ids have no image in {image_dir}. "
            f"Examples: {missing[:10]}. These samples were not dropped silently."
        )
    records = [
        {"image_id": image_id, "label": int(label), "path": str(images[image_id])}
        for image_id, label in zip(ids.tolist(), labels.tolist())
    ]
    counts = Counter(CLASS_NAMES[record["label"]] for record in records)
    expected = EXPECTED_SPLIT_SIZES.get(split)
    report = {
        "split": split,
        "csv": str(csv_path),
        "image_dir": str(image_dir),
        "id_column": id_column,
        "label_mode": label_mode,
        "num_csv_rows": int(len(frame)),
        "num_samples": int(len(records)),
        "expected_samples": expected,
        "count_matches_assignment": None if expected is None else len(records) == expected,
        "class_counts": {name: int(counts.get(name, 0)) for name in CLASS_NAMES},
        "ignored_non_images": ignored,
        "extra_images_without_labels": extra,
        "missing_images": missing,
    }
    return {"records": records, "report": report}


def audit_splits(splits: dict[str, dict]) -> dict:
    """Print sample counts, class counts, and cross-split id overlap. Does not drop rows."""
    reports = {name: payload["report"] for name, payload in splits.items()}
    id_sets = {name: {record["image_id"] for record in payload["records"]} for name, payload in splits.items()}
    overlap = {}
    names = list(id_sets)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            shared = sorted(id_sets[left] & id_sets[right])
            overlap[f"{left}&{right}"] = {"count": len(shared), "examples": shared[:10]}
    reports["cross_split_overlap"] = overlap
    print("=== ISIC split audit ===")
    for name, report in reports.items():
        if name == "cross_split_overlap":
            continue
        expected = report["expected_samples"]
        relation = "n/a" if expected is None else ("OK" if report["count_matches_assignment"] else f"DIFF expected {expected}")
        print(
            f"{name}: samples={report['num_samples']} ({relation}) "
            f"id_column={report['id_column']} labels={report['label_mode']}"
        )
        print(f"  class counts: {json.dumps(report['class_counts'], ensure_ascii=False)}")
        if report["ignored_non_images"]:
            print(f"  ignored non-image files: {report['ignored_non_images']}")
        if report["extra_images_without_labels"]:
            print(
                f"  extra images without labels: {len(report['extra_images_without_labels'])} "
                f"examples={report['extra_images_without_labels'][:5]}"
            )
    for key, value in overlap.items():
        status = "none" if value["count"] == 0 else f"{value['count']} examples={value['examples']}"
        print(f"overlap {key}: {status}")
    if any(value["count"] for value in overlap.values()):
        raise RuntimeError(f"Image ids overlap across splits: {overlap}")
    return reports


def make_loader(records, image_size: int, batch_size: int, train: bool, num_workers: int, seed: int) -> DataLoader:
    from isic_baseline.utils import seed_worker

    dataset = ISICDataset(records, image_size=image_size, train=train)
    generator = torch.Generator()
    generator.manual_seed(seed)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=train,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
        drop_last=False,
        worker_init_fn=seed_worker(seed) if num_workers else None,
        generator=generator,
    )
