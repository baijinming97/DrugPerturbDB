from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


ROOT = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/tahoe")
OBS = ROOT / "tmp/obs_metadata.parquet"
COLUMNS = ["plate", "drugname_drugconc", "drug", "cell_line", "pass_filter"]


def main(rank: int, world: int) -> None:
    out = ROOT / "tmp" / f"obs_full_qc_counts_{rank:02d}-of-{world:02d}.parquet"
    receipt = ROOT / "tmp" / f"obs_full_qc_counts_{rank:02d}-of-{world:02d}.json"
    if out.exists() or receipt.exists():
        raise FileExistsError(f"Refuse overwrite: {out} or {receipt}")
    pf = pq.ParquetFile(OBS)
    tables = []
    full_cells = 0
    row_groups = list(range(rank, pf.metadata.num_row_groups, world))
    for pos, row_group in enumerate(row_groups):
        table = pf.read_row_group(row_group, columns=COLUMNS, use_threads=False)
        for column in ["cell_line", "pass_filter"]:
            field_index = table.schema.get_field_index(column)
            if pa.types.is_dictionary(table.schema.field(column).type):
                table = table.set_column(field_index, column, table[column].cast(pa.string()))
        full = table.filter(pc.equal(table["pass_filter"], pa.scalar("full")))
        full_cells += full.num_rows
        if full.num_rows:
            grouped = full.group_by(["plate", "cell_line", "drugname_drugconc", "drug"]).aggregate(
                [("pass_filter", "count")]
            )
            grouped = grouped.rename_columns(
                ["plate", "cell_line", "target", "raw_drug", "full_qc_cells"]
            )
            grouped = grouped.append_column(
                "source_obs_row_group",
                pa.array([row_group] * grouped.num_rows, type=pa.int32()),
            )
            tables.append(grouped)
        print(
            f"obs_full_qc rank={rank}/{world} row_groups={pos + 1}/{len(row_groups)} full_cells={full_cells}",
            flush=True,
        )
    combined = pa.concat_tables(tables) if tables else pa.table({})
    tmp = out.with_name(out.name + ".part")
    pq.write_table(combined, tmp, compression="zstd")
    os.replace(tmp, out)
    payload = {
        "status": "PASS",
        "rank": rank,
        "world": world,
        "obs_rows": pf.metadata.num_rows,
        "obs_row_groups": pf.metadata.num_row_groups,
        "row_groups_scanned": len(row_groups),
        "full_qc_cells": full_cells,
        "group_rows": combined.num_rows,
        "output": str(out),
    }
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    args = parser.parse_args()
    main(args.rank, args.world)
