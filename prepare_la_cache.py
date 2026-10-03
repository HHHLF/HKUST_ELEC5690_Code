#!/usr/bin/env python
"""Read the NRRD volumes once and store them on the cache directory."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from la_seg3d.data import VolumeStore, cache_directory
from la_seg3d.utils import load_config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    store = VolumeStore(cfg)
    store.discover()
    store.build_cache(cache_directory(cfg))


if __name__ == "__main__":
    main()
