from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import fsspec
import pyarrow as pa
import pyarrow.parquet as pq


ROOT = Path("@WORKSPACE@/baicai/canonical_perturbation_data_v1/tahoe")
URL = (
    "https://huggingface.co/datasets/tahoebio/tahoe-de-rhaister/resolve/"
    "c7963cf334bec0683225d41c9586d900ca6303a2/pdex/all_plates_pdex.parquet?download=true"
)
COLUMNS = ["target", "target_membership", "ref_membership", "plate", "cell_line"]


def main(rank: int, world: int, tag: str, limit_row_groups: int) -> None:
    out = ROOT / "tmp" / f"pdex_profile_scan_{tag}_{rank:02d}-of-{world:02d}.parquet"
    receipt = ROOT / "tmp" / f"pdex_profile_scan_{tag}_{rank:02d}-of-{world:02d}.json"
    if out.exists() or receipt.exists():
        raise FileExistsError(f"Refuse overwrite: {out} or {receipt}")

    records = []
    with fsspec.open(
        URL,
        "rb",
        block_size=8 * 1024 * 1024,
        cache_type="readahead",
        client_kwargs={"trust_env": True},
    ) as handle:
        pf = pq.ParquetFile(handle)
        row_groups = list(range(rank, pf.metadata.num_row_groups, world))
        if limit_row_groups:
            row_groups = row_groups[:limit_row_groups]
        for pos, row_group in enumerate(row_groups):
            table = pf.read_row_group(row_group, columns=COLUMNS, use_threads=False)
            if pa.types.is_large_string(table.schema.field("target").type):
                table = table.set_column(
                    table.schema.get_field_index("target"),
                    "target",
                    table["target"].cast(pa.string()),
                )
            grouped = table.group_by(["plate", "cell_line", "target"]).aggregate(
                [
                    ("target_membership", "min"),
                    ("target_membership", "max"),
                    ("target_membership", "count"),
                    ("ref_membership", "min"),
                    ("ref_membership", "max"),
                ]
            )
            for row in grouped.to_pylist():
                records.append(
                    {
                        "row_group": row_group,
                        "plate": int(row["plate"]),
                        "cell_line": str(row["cell_line"]),
                        "target": str(row["target"]),
                        "target_membership_min": int(row["target_membership_min"]),
                        "target_membership_max": int(row["target_membership_max"]),
                        "ref_membership_min": int(row["ref_membership_min"]),
                        "ref_membership_max": int(row["ref_membership_max"]),
                        "feature_rows": int(row["target_membership_count"]),
                    }
                )
            if (pos + 1) % 100 == 0 or pos + 1 == len(row_groups):
                print(
                    f"rank={rank}/{world} row_groups={pos + 1}/{len(row_groups)} profile_chunks={len(records)}",
                    flush=True,
                )

        total_row_groups = pf.metadata.num_row_groups
        total_rows = pf.metadata.num_rows

    table_out = pa.Table.from_pylist(records)
    tmp = out.with_name(out.name + ".part")
    pq.write_table(table_out, tmp, compression="zstd")
    os.replace(tmp, out)
    payload = {
        "status": "PASS",
        "rank": rank,
        "world": world,
        "tag": tag,
        "pdex_row_groups": total_row_groups,
        "pdex_rows": total_rows,
        "row_groups_scanned": len(row_groups),
        "profile_chunks": len(records),
        "output": str(out),
    }
    receipt.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(payload, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--rank", type=int, required=True)
    parser.add_argument("--world", type=int, required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--limit-row-groups", type=int, default=0)
    args = parser.parse_args()
    main(args.rank, args.world, args.tag, args.limit_row_groups)
