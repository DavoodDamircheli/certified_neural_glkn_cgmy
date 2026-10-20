"""HDF5 read/write helpers used throughout the pipeline.

Centralizes the on-disk layout for reference solutions, datasets, and
experiment results so every stage (``data.dataset``, ``experiments``,
``scripts``) reads and writes HDF5 files the same way: one file, per-record
groups, real datasets for arrays, h5py attrs for scalar metadata.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import h5py
import numpy as np


def _write_dataset(group: h5py.Group, name: str, value: Any) -> None:
    if name in group:
        del group[name]
    arr = np.asarray(value)
    if arr.dtype.kind == "U":  # h5py has no native unicode dtype; use variable-length strings
        arr = arr.astype(h5py.string_dtype(encoding="utf-8"))
    chunks = True if arr.ndim > 0 and arr.size > 0 else None
    group.create_dataset(name, data=arr, chunks=chunks)


def _write_attrs(group_or_file, attrs: dict[str, Any] | None) -> None:
    if not attrs:
        return
    for key, value in attrs.items():
        group_or_file.attrs[key] = value


def write_arrays(
    path: str | Path,
    arrays: dict[str, np.ndarray],
    attrs: dict[str, Any] | None = None,
    mode: str = "w",
) -> None:
    """Write a dict of named arrays (and optional scalar attributes) to an HDF5 file.

    Args:
        path: Destination HDF5 file path.
        arrays: Mapping of dataset name -> array to store.
        attrs: Optional mapping of scalar metadata stored as file-level attributes.
        mode: h5py file mode ('w' to overwrite, 'a' to append).
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, mode) as f:
        for name, arr in arrays.items():
            _write_dataset(f, name, arr)
        _write_attrs(f, attrs)


def read_arrays(path: str | Path, keys: list[str] | None = None) -> dict[str, np.ndarray]:
    """Read named arrays from an HDF5 file (root level only; see also read_group).

    Args:
        path: Source HDF5 file path.
        keys: Optional subset of dataset names to read; reads all if None.
    """
    with h5py.File(path, "r") as f:
        names = keys if keys is not None else [k for k in f.keys() if isinstance(f[k], h5py.Dataset)]
        return {name: f[name][()] for name in names}


def read_attrs(path: str | Path, group: str | None = None) -> dict[str, Any]:
    """Read file-level (or, if ``group`` is given, group-level) attributes."""
    with h5py.File(path, "r") as f:
        target = f[group] if group is not None else f
        return dict(target.attrs)


def read_group(path: str | Path, group: str) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Read every dataset and every attribute under one group.

    Returns:
        (arrays, attrs) for the given group, e.g. one sample's
        ``samples/000123`` group written by ``append_record``.
    """
    with h5py.File(path, "r") as f:
        grp = f[group]
        arrays = {k: v[()] for k, v in grp.items() if isinstance(v, h5py.Dataset)}
        attrs = dict(grp.attrs)
        return arrays, attrs


def list_groups(path: str | Path, parent: str = "/") -> list[str]:
    """List immediate child group names under ``parent`` (e.g. all sample ids)."""
    with h5py.File(path, "r") as f:
        node = f[parent]
        return [k for k in node.keys() if isinstance(node[k], h5py.Group)]


def append_record(
    path: str | Path,
    group: str,
    arrays: dict[str, np.ndarray],
    attrs: dict[str, Any] | None = None,
) -> None:
    """Write (or overwrite) one named group in an HDF5 file, creating the file if needed.

    Intended for incrementally growing a dataset file one sample at a time
    (e.g. during ``data.dataset.generate_dataset``): each call opens the
    file in append mode, writes exactly one group, and closes it, so a
    long-running generation loop never holds the file open across samples
    and a crash partway through only loses the in-flight sample.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with h5py.File(path, "a") as f:
        if group in f:
            del f[group]
        grp = f.create_group(group)
        for name, arr in arrays.items():
            _write_dataset(grp, name, arr)
        _write_attrs(grp, attrs)
