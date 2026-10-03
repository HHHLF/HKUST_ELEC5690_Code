"""Minimal NRRD reader for the local left-atrium release.

The files are NRRD0004, type unsigned char, encoding raw, dimension 3.
`sizes` is X Y Z with X fastest on disk. There is no `space units` field.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np


def read_nrrd(path: str | Path) -> tuple[np.ndarray, tuple[float, float, float], dict]:
    path = Path(path)
    with path.open("rb") as handle:
        lines: list[str] = []
        while True:
            raw = handle.readline()
            if raw == b"":
                raise ValueError(f"{path} ended before the NRRD header terminator")
            if raw in (b"\n", b"\r\n"):
                break
            lines.append(raw.decode("ascii").strip())
        meta = _parse_header(lines, path)
        sizes = [int(v) for v in meta["sizes"].split()]
        if len(sizes) != 3:
            raise ValueError(f"{path} expected 3 sizes, got {sizes}")
        sx, sy, sz = sizes
        count = sx * sy * sz
        payload = handle.read(count)
        if len(payload) != count:
            raise ValueError(f"{path} payload length {len(payload)} != {count}")
        extra = handle.read(1)
        if extra:
            raise ValueError(f"{path} has trailing bytes after the raw volume")

    # Disk order is X-fastest. Reshape as (Z, Y, X), then transpose to (X, Y, Z).
    array_zyx = np.frombuffer(payload, dtype=np.uint8).reshape((sz, sy, sx))
    array_xyz = np.transpose(array_zyx, (2, 1, 0)).copy()
    spacing = _spacing_from_directions(meta.get("space directions", ""), path)
    meta["data_offset_note"] = "array axis 0=X, 1=Y, 2=Z"
    return array_xyz, spacing, meta


def _parse_header(lines: list[str], path: Path) -> dict:
    if not lines or not lines[0].startswith("NRRD"):
        raise ValueError(f"{path} is not an NRRD file")
    meta: dict[str, str] = {}
    for line in lines[1:]:
        if not line or line.startswith("#") or ":" not in line:
            continue
        key, value = line.split(":", 1)
        meta[key.strip()] = value.strip()
    if meta.get("type") != "unsigned char":
        raise ValueError(f"{path} type is {meta.get('type')}, expected unsigned char")
    if meta.get("encoding") != "raw":
        raise ValueError(f"{path} encoding is {meta.get('encoding')}, expected raw")
    if meta.get("dimension") != "3":
        raise ValueError(f"{path} dimension is {meta.get('dimension')}, expected 3")
    if "sizes" not in meta:
        raise ValueError(f"{path} header has no sizes")
    return meta


def _spacing_from_directions(text: str, path: Path) -> tuple[float, float, float]:
    vectors = re.findall(r"\(([^)]*)\)", text)
    if len(vectors) != 3:
        raise ValueError(f"{path} space directions {text!r} does not contain 3 vectors")
    norms = []
    for vector in vectors:
        comps = np.array([float(part) for part in vector.split(",")], dtype=np.float64)
        if comps.shape != (3,):
            raise ValueError(f"{path} bad space direction {vector}")
        norm = float(np.linalg.norm(comps))
        if norm <= 0:
            raise ValueError(f"{path} non-positive spacing from {vector}")
        norms.append(norm)
    return (norms[0], norms[1], norms[2])
