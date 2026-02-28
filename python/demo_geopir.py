from __future__ import annotations
import os, time, json
from dataclasses import dataclass
from typing import List, Tuple

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


def parse_tile_record(rec: bytes) -> dict:
    """
    Records are JSON padded with NUL bytes to fixed record_size.
    """
    
    txt = rec.rstrip(b"\x00").decode("utf-8", errors="replace")
    try:
        return json.loads(txt)
    except json.JSONDecodeError:
        return {"_raw": txt}


def pretty_print_tile(d: dict) -> None:
    if "_raw" in d:
        print(d["_raw"])
        return

    print(f"tile: {d.get('tile')}")
    print(f"temperature: {d.get('temperature_c')} °C")
    print(f"AQI: {d.get('air_quality_index')}")
    print(f"precipitation: {d.get('precipitation_mm')} mm")
    lvl = d.get("proximity_alert_level")
    print(f"proximity alert: level {lvl} (4 = highest)")


def run_baseline(db_path: str, idx: int) -> Tuple[Result, bytes]:
    record_size, n_tiles, _ = read_tile_db_bin_header(db_path)

    #4 bytes
    upload = 4

    t0 = time.perf_counter()
    s0 = time.perf_counter()
    rec = direct_fetch(db_path, idx)
    s1 = time.perf_counter()

    # client decode by removing padding
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
    ), rec


def run_ypir(db_path: str, idx: int) -> Tuple[Result, bytes]:
    record_size, n_tiles, _ = read_tile_db_bin_header(db_path)

    # align them with tile DB
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
    ), rec


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

    baseline_res, baseline_rec = run_baseline(db_path, idx)
    ypir_res, ypir_rec = run_ypir(db_path, idx)

    print("\n=== TILE DATA (decoded from returned record) ===")
    print("Baseline direct fetch:")
    pretty_print_tile(parse_tile_record(baseline_rec))


    results: List[Result] = [baseline_res, ypir_res]
    print_results(results)


if __name__ == "__main__":
    main()
