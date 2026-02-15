from __future__ import annotations
import os, time
from dataclasses import dataclass
from typing import List

from tile_db import GeoGrid, write_tile_db_bin, read_tile_db_bin_header, direct_fetch
from ypir_adapter import ypir_setup, ypir_make_query, ypir_answer, ypir_extract


@dataclass
class Result:
    label: str
    db_path: str
    n_tiles: int
    record_size: int
    idx: int
    upload_bytes: int
    download_bytes: int
    t_client_ms: float
    t_server_ms: float
    t_total_ms: float


def _ms(dt: float) -> float:
    return 1000.0 * dt


def ensure_db(path: str, record_size: int, grid: GeoGrid) -> None:
    if os.path.exists(path):
        return
    write_tile_db_bin(path, grid, record_size, seed=42)


def run_baseline(db_path: str, idx: int) -> Result:
    record_size, n_tiles, _ = read_tile_db_bin_header(db_path)

    # Client sends idx (pretend 4 bytes)
    upload = 4

    t0 = time.perf_counter()
    s0 = time.perf_counter()
    rec = direct_fetch(db_path, idx)
    s1 = time.perf_counter()
    c1 = time.perf_counter()
    _ = rec.rstrip(b"\x00")
    t1 = time.perf_counter()

    return Result(
        label="baseline_direct",
        db_path=db_path,
        n_tiles=n_tiles,
        record_size=record_size,
        idx=idx,
        upload_bytes=upload,
        download_bytes=len(rec),
        t_client_ms=_ms((t1 - t0) - (s1 - s0)),
        t_server_ms=_ms(s1 - s0),
        t_total_ms=_ms(t1 - t0),
    )


def run_ypir(db_path: str, idx: int) -> Result:
    record_size, n_tiles, _ = read_tile_db_bin_header(db_path)

    # IMPORTANT: n_items & item_size_bytes define the logical DB for params_for(...)
    # Here we align them with your tile DB.
    ctx = ypir_setup(db_path, n_tiles, record_size, is_simplepir=False)

    t0 = time.perf_counter()

    # Client query generation
    c0 = time.perf_counter()
    query_bytes = ypir_make_query(ctx, idx)
    c1 = time.perf_counter()

    # Server answer
    s0 = time.perf_counter()
    resp_bytes = ypir_answer(ctx, query_bytes)
    s1 = time.perf_counter()

    # Client extract
    c2 = time.perf_counter()
    raw = ypir_extract(ctx, resp_bytes)
    # Demo-friendly: interpret extracted bytes as “record”, truncate to record_size
    rec = bytes(raw[:record_size])
    c3 = time.perf_counter()

    t1 = time.perf_counter()

    return Result(
        label="ypir",
        db_path=db_path,
        n_tiles=n_tiles,
        record_size=record_size,
        idx=idx,
        upload_bytes=len(query_bytes),
        download_bytes=len(resp_bytes),
        t_client_ms=_ms((c1 - c0) + (c3 - c2)),
        t_server_ms=_ms(s1 - s0),
        t_total_ms=_ms(t1 - t0),
    )


def print_results(results: List[Result]) -> None:
    print("\n=== RESULTS ===")
    print("label,db,tiles,recB,idx,upB,downB,client_ms,server_ms,total_ms")
    for r in results:
        print(
            f"{r.label},{os.path.basename(r.db_path)},{r.n_tiles},{r.record_size},{r.idx},"
            f"{r.upload_bytes},{r.download_bytes},"
            f"{r.t_client_ms:.2f},{r.t_server_ms:.2f},{r.t_total_ms:.2f}"
        )


def main() -> None:
    grid = GeoGrid(
        lat_min=48.0, lon_min=8.0,
        lat_step=0.01, lon_step=0.01,
        n_lat=400, n_lon=400
    )

    record_size = 256  # bytes per tile record

    os.makedirs("data", exist_ok=True)
    db_path = os.path.join("data", "tiles.bin")
    ensure_db(db_path, record_size, grid)

    # Example “user location”
    lat, lon = 48.137, 11.575
    idx = grid.tile_index(lat, lon)

    results: List[Result] = []
    results.append(run_baseline(db_path, idx))
    results.append(run_ypir(db_path, idx))

    print_results(results)


if __name__ == "__main__":
    main()
