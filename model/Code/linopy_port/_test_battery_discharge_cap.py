"""Test for the battery discharge power cap, Max_Bat_flow_out (added 2026-09-14).

Checks:
  1. Structural (no solve): the constraint is built by default in every bess_mode and in
     lp_mode, and is absent with battery_discharge_cap=False.
  2. Functional (two small Gurobi solves, about a minute): a pinned design on the 1-year merged
     smoke-test inputs with a deliberately small battery (2,000 kWh, so the cap is 500 kW at the
     .dat's 4 h discharge time). Without the cap its dispatch discharges above 500 kW; with the
     cap it never does, and the cost cannot fall.

    cd Code/linopy_port && conda activate mgpy
    python _test_battery_discharge_cap.py        # exit code 0 = all passed
"""
import os
import sys

import xarray as xr

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from linopy_core_model_builder import DesignInputs, build_core_model  # noqa: E402

INPUTS = os.path.join(HERE, "..", "Inputs")
SMOKE = dict(  # sites/smoketest_1sc_1y_merged.conf
    params_dat=os.path.join(INPUTS, "Parameters_1sc_1y_gili_ketapang_merged_smoketest.dat"),
    demand_csv=os.path.join(INPUTS, "Demand_1sc_1y_gili_ketapang_merged_smoketest.csv"),
    res_csv=os.path.join(INPUTS, "RES_Time_Series_1sc_gili_ketapang_merged.csv"),
    max_battery_kwh=20550.0, max_generator_kw=9400.0)
TOL = 1e-3  # kWh, same threshold as check_simultaneous_flows()

fails = 0


def check(cond, msg):
    global fails
    print(("[OK]   " if cond else "[FAIL] ") + msg)
    if not cond:
        fails += 1


inp = DesignInputs(**SMOKE)
check(inp.BATT_MIN_DISCHARGE_TIME > 0, f"Maximum_Battery_Discharge_Time loaded: {inp.BATT_MIN_DISCHARGE_TIME} h")

# 1. structural
for label, kwargs in (("bess_mode=binary", dict(bess_mode="binary")),
                      ("bess_mode=relaxed", dict(bess_mode="relaxed")),
                      ("bess_mode=nobinary", dict(bess_mode="nobinary")),
                      ("lp_mode", dict(bess_mode="nobinary", lp_mode=True))):
    m = build_core_model(inp, **kwargs)["m"]
    check("Max_Bat_flow_out" in m.constraints, f"{label}: Max_Bat_flow_out present by default")
m = build_core_model(inp, bess_mode="relaxed", battery_discharge_cap=False)["m"]
check("Max_Bat_flow_out" not in m.constraints, "battery_discharge_cap=False: constraint absent")

# 2. functional
BATTERY_KWH = 2000
cap_kw = BATTERY_KWH * inp.BATT_NOM_CAP_KWH / inp.BATT_MIN_DISCHARGE_TIME
pinned = {
    "RES": xr.DataArray([5055.0], coords={"ut": inp.ut_idx}),          # the smoke test's own PV
    "Battery": xr.DataArray([BATTERY_KWH / inp.BATT_NOM_CAP_KWH], coords={"ut": inp.ut_idx}),
    "Generator": xr.DataArray([2.0], coords={"ut": inp.ut_idx}),       # the two existing units
}
solve_kwargs = dict(OutputFlag=0, Threads=4, MIPGap=1e-6)
results = {}
for cap_on in (False, True):
    # battery_min_capacity=False: a 2,000 kWh battery is far below the one-day autonomy floor.
    built = build_core_model(inp, pinned=pinned, bess_mode="relaxed", battery_min_capacity=False,
                             battery_discharge_cap=cap_on)
    status, condition = built["m"].solve(solver_name="gurobi", **solve_kwargs)
    npc = built["m"].objective.value + built["total_const_offset"] if status == "ok" else None
    peak = float(built["Battery_Outflow"].solution.max()) if status == "ok" else None
    hours_over = int((built["Battery_Outflow"].solution > cap_kw + TOL).sum()) if status == "ok" else None
    results[cap_on] = (status, condition, npc, peak, hours_over)
    print(f"  cap={'on ' if cap_on else 'off'}: status {status} {condition}, NPC {npc:,.2f}, "
          f"peak discharge {peak:.1f} kW, hours above {cap_kw:.1f} kW: {hours_over}")

off, on = results[False], results[True]
check(off[0] == "ok" and on[0] == "ok", "both pinned solves return ok")
if off[0] == "ok" and on[0] == "ok":
    check(off[4] > 0, f"without the cap the dispatch exceeds {cap_kw:.1f} kW (so the test exercises the cap)")
    check(on[4] == 0, f"with the cap no hour exceeds {cap_kw:.1f} kW")
    check(on[2] >= off[2] - 1e-6 * abs(off[2]), "adding the cap does not lower the cost")

print("ALL PASSED" if fails == 0 else f"{fails} FAILURE(S)")
sys.exit(fails)
