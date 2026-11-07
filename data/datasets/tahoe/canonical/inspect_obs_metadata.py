from __future__ import annotations

import json

import pyarrow.parquet as pq


PATH = "@WORKSPACE@/baicai/canonical_perturbation_data_v1/tahoe/tmp/obs_metadata.parquet"
with open(PATH, "rb") as handle:
    pf = pq.ParquetFile(handle)
    print(
        json.dumps(
            {
                "rows": pf.metadata.num_rows,
                "row_groups": pf.metadata.num_row_groups,
                "columns": pf.schema_arrow.names,
                "schema": str(pf.schema_arrow),
                "row_group_0_rows": pf.metadata.row_group(0).num_rows,
            },
            indent=2,
        ),
        flush=True,
    )
    table = pf.read_row_group(0, use_threads=False)
    print(json.dumps({"head": table.slice(0, 10).to_pylist()}, indent=2, default=str))
