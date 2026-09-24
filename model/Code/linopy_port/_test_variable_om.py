"""Structural test for the optional variable O&M cost (added 2026-09-18). No solve is run.

Builds the 6sc/2y partial-load fixture of _test_uc_integral.py twice -- once as its .dat stands
(no Generator_Variable_OM_Cost / Battery_Variable_OM_Cost, so both default to 0) and once with
both costs set on the DesignInputs object -- and checks:
  1. the defaults are 0 when the .dat does not set them;
  2. the objective difference between the two builds is exactly
     weight(s) * cost / (1+r)^yt on every Generator_Energy_Total and Battery_Outflow element, and
     zero on every other variable and on the constant.

    cd Code/linopy_port && conda activate mgpy
    python _test_variable_om.py        # exit code 0 = all passed
"""
import os
import sys

import numpy as np
import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from linopy_core_model_builder import DesignInputs, build_core_model  # noqa: E402

INPUTS = os.path.join(HERE, "..", "Inputs")
PARTIAL = dict(
    params_dat=os.path.join(INPUTS, "Parameters_6sc_2y_gili_ketapang_MILP_dieselops.dat"),
    demand_csv=os.path.join(INPUTS, "Demand_6sc_2y_gili_ketapang_dieselops.csv"),
    res_csv=os.path.join(INPUTS, "RES_Time_Series_6sc_gili_ketapang.csv"),
    max_battery_kwh=74675.0, max_generator_kw=9400.0)
GEN_VOM, BATT_VOM = 0.00717, 0.001943

fails = 0


def check(cond, msg):
    global fails
    print(("[OK]   " if cond else "[FAIL] ") + msg)
    if not cond:
        fails += 1


def objective_by_label(m):
    """Objective coefficients summed per variable label, plus the constant."""
    expr = m.objective.expression
    coeffs = np.ravel(expr.coeffs.values)
    labels = np.ravel(expr.vars.values)
    keep = labels >= 0
    out = {}
    for lab, c in zip(labels[keep], coeffs[keep]):
        out[int(lab)] = out.get(int(lab), 0.0) + float(c)
    const = float(np.asarray(expr.const).sum())
    return out, const


inp_base = DesignInputs(**PARTIAL)
check(inp_base.GEN_VAR_OM == 0.0 and inp_base.BATT_VAR_OM == 0.0,
      f"defaults when the .dat does not set them: GEN_VAR_OM={inp_base.GEN_VAR_OM}, BATT_VAR_OM={inp_base.BATT_VAR_OM}")
base = build_core_model(inp_base, bess_mode="relaxed")

inp_vom = DesignInputs(**PARTIAL)
inp_vom.GEN_VAR_OM, inp_vom.BATT_VAR_OM = GEN_VOM, BATT_VOM
vom = build_core_model(inp_vom, bess_mode="relaxed")

b_obj, b_const = objective_by_label(base["m"])
v_obj, v_const = objective_by_label(vom["m"])
check(abs(v_const - b_const) < 1e-9, f"objective constant unchanged ({b_const:.6g})")

disc = (1 + inp_base.r) ** xr.DataArray(inp_base.yt_idx.values, coords={"yt": inp_base.yt_idx})
w = inp_base.scenario_weight
expected = {}
for name, cost in (("Generator_Energy_Total", GEN_VOM), ("Battery_Outflow", BATT_VOM)):
    labels = base["m"].variables[name].labels
    check(bool((labels == vom["m"].variables[name].labels).all()), f"{name}: same labels in both builds")
    coef = (cost * w / disc).broadcast_like(labels)
    for lab, c in zip(np.ravel(labels.values), np.ravel(coef.transpose(*labels.dims).values)):
        expected[int(lab)] = float(c)

all_labels = set(b_obj) | set(v_obj)
worst_expected = max(abs((v_obj.get(l, 0.0) - b_obj.get(l, 0.0)) - expected[l]) for l in expected)
worst_other = max((abs(v_obj.get(l, 0.0) - b_obj.get(l, 0.0)) for l in all_labels - set(expected)), default=0.0)
check(worst_expected < 1e-12, f"added coefficient = weight*cost/(1+r)^yt on {len(expected)} elements "
      f"(worst deviation {worst_expected:.2e})")
check(worst_other < 1e-12, f"no other objective coefficient changes (worst {worst_other:.2e} over "
      f"{len(all_labels - set(expected))} labels)")

print("ALL PASSED" if fails == 0 else f"{fails} FAILURE(S)")
sys.exit(fails)
