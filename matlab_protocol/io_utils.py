from __future__ import annotations

from dataclasses import asdict, is_dataclass
from pathlib import Path
import csv
import tempfile

import h5py
import numpy as np


def _serialize_value(value):
    if is_dataclass(value):
        return {k: _serialize_value(v) for k, v in asdict(value).items()}
    if isinstance(value, dict):
        return {k: _serialize_value(v) for k, v in value.items()}
    if isinstance(value, np.ndarray):
        return value
    if isinstance(value, (list, tuple)):
        return np.asarray(value)
    return value


def save_simulation_h5(
    save_dir: str | Path,
    qx: np.ndarray,
    qy: np.ndarray,
    I_noisy: np.ndarray,
    V: float,
    instrument,
    ensemble,
) -> Path:
    save_path = Path(save_dir).expanduser()
    save_path.mkdir(parents=True, exist_ok=True)
    existing = sorted(save_path.glob("sim_*_*.h5"))
    if not existing:
        next_idx = 1
    else:
        indices = []
        for path in existing:
            try:
                indices.append(int(path.name.split("_")[1]))
            except Exception:
                continue
        next_idx = max(indices, default=0) + 1
    fname_base = f"sim_{next_idx:04d}_V{V:0.3f}"
    final_path = save_path / f"{fname_base}.h5"
    temp_path = Path(tempfile.gettempdir()) / final_path.name
    chunk = tuple(min(100, dim) for dim in I_noisy.shape)
    if temp_path.exists():
        temp_path.unlink()
    with h5py.File(temp_path, "w") as h5:
        h5.create_dataset("/qx", data=qx, chunks=chunk, compression="gzip", compression_opts=5)
        h5.create_dataset("/qy", data=qy, chunks=chunk, compression="gzip", compression_opts=5)
        h5.create_dataset("/I_noisy", data=I_noisy, chunks=chunk, compression="gzip", compression_opts=5)
    temp_path.replace(final_path)

    csv_path = save_path / "sim_metadata_log.csv"
    write_header = not csv_path.exists()
    with csv_path.open("a", newline="") as fh:
        writer = csv.writer(fh)
        if write_header:
            writer.writerow(["Date", "Index", "Filename", "V", "Rg", "DistType", "Lambda", "SD_cm"])
        from datetime import datetime

        writer.writerow(
            [
                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                next_idx,
                final_path.name,
                f"{V:.4f}",
                f"{ensemble.rg:.2f}",
                ensemble.d_nam,
                f"{instrument.lambda_:.3f}",
                f"{instrument.SD_dist:.1f}",
            ]
        )
    return final_path

