"""Structural test for Scenario_Weight handling (added 2026-09-11). No solve is run.

Until 2026-09-11 DesignInputs hard-coded the scenario weights to 1/S instead of reading the .dat's
Scenario_Weight as Pyomo does; this only went unnoticed because every earlier design had uniform
weights. Checks:
  1. The probabilistic merged-demand design (scenario count read from
     merged_scenarios_gili_ketapang.json) loads its non-uniform weights exactly as the JSON
     records them.
  2. A legacy uniform design (9sc, 0.111111 each) still loads its own .dat values.
  3. The weighted-expectation semantics every cost term uses ((... * scenario_weight).sum("s")):
     equal per-scenario costs give that cost, a cost only in scenario 1 gives cost * w_1.
  4. A .dat whose weights do not sum to 1 is refused.

    cd Code/linopy_port && conda activate mgpy
    python _test_scenario_weights.py        # exit code 0 = all passed
"""
import json
import os
import sys
import tempfile

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from linopy_core_model_builder import DesignInputs  # noqa: E402

INPUTS = os.path.join(HERE, "..", "Inputs")
with open(os.path.join(INPUTS, "merged_scenarios_gili_ketapang.json"), encoding="utf-8") as f:
    summary = json.load(f)
expected = np.array(summary["probabilistic"]["scenario_weights"])
S_PROB = len(expected)
MERGED = dict(
    params_dat=os.path.join(INPUTS, f"Parameters_{S_PROB}sc_20y_gili_ketapang_1step_MILP_dieselops_dea2025.dat"),
    demand_csv=os.path.join(INPUTS, summary["outputs"]["demand_prob"]),
    res_csv=os.path.join(INPUTS, summary["outputs"]["res_prob"]),
    max_battery_kwh=50000.0, max_generator_kw=9400.0)
LEGACY = dict(
    params_dat=os.path.join(INPUTS, "Parameters_9sc_20y_gili_ketapang_capexp5_MILP_dieselops.dat"),
    demand_csv=os.path.join(INPUTS, "Demand_9sc_20y_gili_ketapang.csv"),
    res_csv=os.path.join(INPUTS, "RES_Time_Series_9sc_gili_ketapang.csv"),
    max_battery_kwh=62031.0, max_generator_kw=9400.0)

fails = 0


def check(cond, msg):
    global fails
    print(("[OK]   " if cond else "[FAIL] ") + msg)
    if not cond:
        fails += 1


# 1. merged design
inp = DesignInputs(**MERGED)
w = inp.scenario_weight.values
check(inp.S == S_PROB and len(w) == S_PROB, f"merged design: {S_PROB} scenarios, {len(w)} weights")
check(np.allclose(w, expected, atol=1e-9), "merged design: weights equal merged_scenarios_gili_ketapang.json")
check(abs(w.sum() - 1.0) < 1e-9, f"merged design: weights sum to 1 ({w.sum():.9f})")
check(not np.allclose(w, 1.0 / inp.S), "merged design: weights are not the old hard-coded 1/S")

# 2. legacy uniform design
inp_legacy = DesignInputs(**LEGACY)
check(np.allclose(inp_legacy.scenario_weight.values, 0.111111, atol=1e-12),
      "legacy 9sc design: weights are the .dat's 0.111111 values")

# 3. weighted expectation semantics used by every cost term
per_scenario_cost = np.full(inp.S, 7.0)
check(abs((per_scenario_cost * w).sum() - 7.0) < 1e-9, "equal per-scenario costs -> weighted sum equals that cost")
only_first = np.zeros(inp.S)
only_first[0] = 100.0
check(abs((only_first * w).sum() - 100.0 * w[0]) < 1e-9, "cost only in scenario 1 -> weighted sum = cost * w_1")

# 4. weights that do not sum to 1 are refused
with open(MERGED["params_dat"], encoding="utf-8") as f:
    text = f.read()
bad = text.replace(f"1      {expected[0]:.6f}", "1      0.500000", 1)
assert bad != text, "test setup: weight line for scenario 1 not found"
with tempfile.NamedTemporaryFile("w", suffix=".dat", delete=False, encoding="utf-8") as f:
    f.write(bad)
    bad_path = f.name
try:
    DesignInputs(**{**MERGED, "params_dat": bad_path})
    check(False, "weights summing to != 1 should raise")
except ValueError:
    check(True, "weights summing to != 1 raise ValueError")
finally:
    os.remove(bad_path)

print("ALL PASSED" if fails == 0 else f"{fails} FAILURE(S)")
sys.exit(fails)
