"""Portable immutable snapshot helpers shared by data staging and RM9 research."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def file_hash(path):
    with open(path, "rb") as fp:
        return hashlib.file_digest(fp, "sha256").hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write(path, obj):
    """Replace a complete JSON document atomically, on Windows as well as Linux."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    tmp.write_text(json.dumps(obj, indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def pin(path, obj):
    if Path(path).exists():
        if read(path) != obj:
            raise ValueError(f"{path}: inputs changed. Keep this run and use a new directory.")
        return
    write(path, obj)
