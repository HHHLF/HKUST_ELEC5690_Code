#!/usr/bin/env python
"""Train the 2D U-Net. The test set is not used to choose the checkpoint."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT / "src"))

from la_seg2d.engine import train
from la_seg3d.utils import load_config, pick_device


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/segmentation2d.yaml")
    parser.add_argument("--resume", default=None, help="checkpoint to resume, usually last.pt")
    args = parser.parse_args()
    cfg = load_config(ROOT / args.config)
    train(cfg, pick_device(cfg), resume=args.resume, root=ROOT)


if __name__ == "__main__":
    main()
