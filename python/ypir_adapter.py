from __future__ import annotations
from dataclasses import dataclass
from typing import Any, Tuple, Optional

import ypir_rs

from tile_db import read_tile_db_bin_header


@dataclass
class YpirContext:
    params: Any
    client: Any
    server: Any
    dim_log2: int
    required_db_bytes: int
    n_items: int
    item_size_bytes: int


def _read_records_region(db_path: str) -> bytes:
    """Reads only the records region (skips TILEDB header)."""
    record_size, n_tiles, header_size = read_tile_db_bin_header(db_path)
    with open(db_path, "rb") as f:
        f.seek(header_size)
        return f.read()


def _build_ypir_db_bytes(db_path: str, required: int) -> bytes:
    """
    YPIR expects a big flattened u8 matrix of a specific length.
    Your TILEDB file is N fixed-size records; here we map it into the required
    length by repeating/truncating the records region.
    """
    records = _read_records_region(db_path)
    if len(records) == 0:
        raise ValueError("DB records region is empty")

    if len(records) >= required:
        return records[:required]

    # Repeat records until we have enough bytes
    reps = (required + len(records) - 1) // len(records)
    big = (records * reps)[:required]
    return big


def ypir_setup(db_path: str, n_items: int, item_size_bytes: int, *, is_simplepir: bool = False) -> YpirContext:
    """
    Builds params, client, and server.
    NOTE: is_simplepir=False corresponds to your "YPIR" path.
    """
    params = ypir_rs.params_for(n_items, item_size_bytes, is_simplepir)
    dim_log2 = int(ypir_rs.params_db_dim_1(params))
    required = int(ypir_rs.required_db_bytes(params))

    # Build DB bytes in the format/size the Rust server expects
    db_bytes = _build_ypir_db_bytes(db_path, required)

    client = ypir_rs.client_new(params)

    # inp_transposed=False, pad_rows=True are safe defaults for demos
    server = ypir_rs.server_new(params, db_bytes, False, True)

    return YpirContext(
        params=params,
        client=client,
        server=server,
        dim_log2=dim_log2,
        required_db_bytes=required,
        n_items=n_items,
        item_size_bytes=item_size_bytes,
    )


def ypir_make_query(ctx: YpirContext, idx: int) -> bytes:
    """
    Generates a packed query bytes blob (ready for ypir_rs.answer).
    """
    # Query index must be within [0, 2^dim_log2). We map tile idx into that range.
    row = idx % (1 << ctx.dim_log2)

    # These defaults work with the packing step we added in Rust:
    public_seed_idx = 0
    packing = True
    pack = True  # IMPORTANT: server expects packed query words

    return ypir_rs.query(ctx.client, public_seed_idx, ctx.dim_log2, packing, row, pack)


def ypir_answer(ctx: YpirContext, query_bytes: bytes) -> bytes:
    return ypir_rs.answer(ctx.server, query_bytes)


def ypir_extract(ctx: YpirContext, response_bytes: bytes) -> bytes:
    """
    Returns decoded response words as bytes (little-endian u64s).
    For “record bytes”, you’ll typically slice to item_size_bytes (demo-friendly).
    """
    out = ypir_rs.extract(ctx.client, response_bytes)
    return out
