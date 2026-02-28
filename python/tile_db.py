from __future__ import annotations
import os, json, math, struct, random
from dataclasses import dataclass
from typing import Tuple

@dataclass(frozen=True)
class GeoGrid:
    lat_min: float
    lon_min: float
    lat_step: float
    lon_step: float
    n_lat: int
    n_lon: int

    def tile_index(self, lat: float, lon: float) -> int:
        """Map (lat,lon) -> [0, n_lat*n_lon). Clamps to grid."""
        i = int(math.floor((lat - self.lat_min) / self.lat_step))
        j = int(math.floor((lon - self.lon_min) / self.lon_step))
        i = max(0, min(self.n_lat - 1, i))
        j = max(0, min(self.n_lon - 1, j))
        return i * self.n_lon + j

    @property
    def n_tiles(self) -> int:
        return self.n_lat * self.n_lon


def _pad_record(b: bytes, record_size: int) -> bytes:
    if len(b) > record_size:
        return b[:record_size]
    return b + b"\x00" * (record_size - len(b))


def write_tile_db_bin(
    path: str,
    grid: GeoGrid,
    record_size: int,
    seed: int = 0,
) -> None:
    """
    Writes a binary file of N fixed-size records.
    Record i contains JSON metadata + padding to `record_size`.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    rng = random.Random(seed)

    with open(path, "wb") as f:
        # header: magic + record_size + n_tiles
        f.write(b"TILEDB1")
        f.write(struct.pack("<II", record_size, grid.n_tiles))

        for idx in range(grid.n_tiles):
            #Fields to show in dmeo
            payload = {
                "tile": idx,
                "temperature_c": round(rng.uniform(-10.0, 40.0), 1),
                "air_quality_index": int(rng.uniform(0, 301)),      # 0..300
                "precipitation_mm": round(rng.uniform(0.0, 25.0), 1),
                "proximity_alert_level": int(rng.uniform(1, 5)),    # 1..4
            }
            raw = json.dumps(payload, separators=(",", ":")).encode("utf-8")
            f.write(_pad_record(raw, record_size))


def read_tile_db_bin_header(path: str) -> Tuple[int, int, int]:
    with open(path, "rb") as f:
        magic = f.read(7)
        if magic != b"TILEDB1":
            raise ValueError("Bad magic header")
        record_size, n_tiles = struct.unpack("<II", f.read(8))
        header_size = 7 + 8
    return record_size, n_tiles, header_size


def direct_fetch(path: str, idx: int) -> bytes:
    """Baseline (non-PIR): server returns record idx."""
    record_size, n_tiles, header_size = read_tile_db_bin_header(path)
    if not (0 <= idx < n_tiles):
        raise IndexError("idx out of range")
    with open(path, "rb") as f:
        f.seek(header_size + idx * record_size)
        return f.read(record_size)
