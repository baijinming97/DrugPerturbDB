from __future__ import annotations

import json
import io
import urllib.request

import pyarrow.parquet as pq


BASE = "https://huggingface.co/datasets/tahoebio/Tahoe-100M/resolve/main/metadata"
for name in ["cell_line_metadata.parquet", "drug_metadata.parquet", "gene_metadata.parquet"]:
    url = f"{BASE}/{name}?download=true"
    with urllib.request.urlopen(url) as response:
        table = pq.read_table(io.BytesIO(response.read()))
    payload = {
        "file": name,
        "rows": table.num_rows,
        "columns": table.column_names,
        "schema": str(table.schema),
        "head": table.slice(0, 5).to_pylist(),
    }
    if name == "cell_line_metadata.parquet":
        payload["all_rows"] = table.to_pylist()
    print(json.dumps(payload, indent=2, sort_keys=True, default=str))
