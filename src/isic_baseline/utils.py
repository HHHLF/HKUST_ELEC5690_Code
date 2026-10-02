"""Shared helpers: config, seeding, device, JSON."""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path

import numpy as np
import torch
import yaml

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def resolve_path(path, base: Path | None = None) -> Path:
    path = Path(path)
    if path.is_absolute():
        return path
    return (base or PROJECT_ROOT) / path


def load_config(path: str | Path, overrides: dict | None = None) -> dict:
    path = Path(path)
    with path.open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    if overrides:
        for key, value in overrides.items():
            if value is not None:
                cfg[key] = value
    cfg["_config_path"] = str(path.resolve())
    for key in (
        "weights",
        "train_image_dir",
        "train_csv",
        "val_image_dir",
        "val_csv",
        "test_image_dir",
        "test_csv",
        "output_dir",
        "resume",
    ):
        if cfg.get(key):
            cfg[key] = str(resolve_path(cfg[key]))
    return cfg


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def seed_worker(seed: int):
    def _init(worker_id: int) -> None:
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return _init


def get_device(name: str) -> torch.device:
    if name.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is False.")
    return torch.device(name)


class _Tee:
    def __init__(self, *streams) -> None:
        self.streams = streams

    def write(self, data: str) -> int:
        for stream in self.streams:
            stream.write(data)
            stream.flush()
        return len(data)

    def flush(self) -> None:
        for stream in self.streams:
            stream.flush()


def start_log(path: Path):
    """Overwrite ``path`` and copy later stdout and stderr into it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("w", encoding="utf-8", buffering=1)
    sys.stdout = _Tee(sys.__stdout__, handle)
    sys.stderr = _Tee(sys.__stderr__, handle)
    print(f"Log file: {path}")
    return handle


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, ensure_ascii=False)


def load_json(path: Path):
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)
