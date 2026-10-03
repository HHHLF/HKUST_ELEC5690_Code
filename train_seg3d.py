#!/usr/bin/env python
"""Train the 3D U-Net. The test set is audited and then left untouched."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from la_seg3d.engine import train
from la_seg3d.utils import load_config, pick_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--resume", default=None, help="checkpoint to resume, usually last.pt")
    args = parser.parse_args()
    cfg = load_config(args.config)
    train(cfg, pick_device(cfg), resume=args.resume)


if __name__ == "__main__":
    main()
