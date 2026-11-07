import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, "@WORKSPACE@/baicai/canonical_perturbation_data_v1/l1000/scripts")
from canonical_l1000_loader import CanonicalL1000


reader = CanonicalL1000()
checks = {}
for dataset, rows in {
    "l3_treated": [1_312_169, 0, 42, 42],
    "l3_control": [76_741, 0, 42, 42],
    "l5": [510_269, 0, 42, 42],
}.items():
    values = reader.read(dataset, rows, genes=[977, 0, 100])
    assert values.shape == (4, 3)
    assert np.array_equal(values[2], values[3])
    checks[dataset] = {"shape": list(values.shape), "finite": bool(np.isfinite(values).all())}
controls = reader.control_rows_for_treatments([0, 1_312_169])
checks["control_mapping_lengths"] = list(map(len, controls))
print(json.dumps({"status": "PASS", "checks": checks}, sort_keys=True))
