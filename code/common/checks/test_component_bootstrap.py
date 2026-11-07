"""Component-specific intervals exclude observations outside the eligible support."""
import pandas as pd
from tahoe_repro.evaluate import summarize_drugs


def test_zero_weight_groups_do_not_change_component_interval():
    data = pd.DataFrame({"group": ["a", "b", "c"],
                         "weight": [0.2, 0.3, 0.5], "score": [0.1, 0.4, 0.8]})
    outside = pd.DataFrame({"group": ["ineligible"], "weight": [0.0], "score": [999.0]})
    expected = summarize_drugs(data, ["score"])
    actual = summarize_drugs(pd.concat([data, outside], ignore_index=True), ["score"])
    assert actual == expected
