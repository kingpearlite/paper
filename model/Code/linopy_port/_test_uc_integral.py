"""Structural test for the integral fixed-design path (added 2026-09-11). No solve is run.

Checks:
  1. resolve_plain_mode() follows the .dat's Generator_Partial_Load flag, still accepts an
     agreeing LINOPY_PLAIN, and raises on a contradicting one.
  2. build_core_model(uc_integral=True) declares Generator_Partial binary and Generator_Full
     integer; the default (uc_integral=False) keeps both continuous, as every earlier solve had.
  3. Under the plain formulation uc_integral changes nothing (both stay continuous, pinned to 0).
  4. (2026-09-18) gen_parallel=True with uc_integral=True makes Generator_Partial an integer unit
     count (not binary) and pins Generator_Full to 0; without uc_integral it changes nothing.

    cd Code/linopy_port && conda activate mgpy
    python _test_uc_integral.py        # exit code 0 = all passed
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from linopy_core_model_builder import DesignInputs, build_core_model, resolve_plain_mode  # noqa: E402

INPUTS = os.path.join(HERE, "..", "Inputs")
PARTIAL = dict(  # Generator_Partial_Load=1
    params_dat=os.path.join(INPUTS, "Parameters_6sc_2y_gili_ketapang_MILP_dieselops.dat"),
    demand_csv=os.path.join(INPUTS, "Demand_6sc_2y_gili_ketapang_dieselops.csv"),
    res_csv=os.path.join(INPUTS, "RES_Time_Series_6sc_gili_ketapang.csv"),
    max_battery_kwh=74675.0, max_generator_kw=9400.0)
PLAIN = dict(  # Generator_Partial_Load=0
    params_dat=os.path.join(INPUTS, "Parameters_1sc_2y_gili_ketapang_scpos2_MILP.dat"),
    demand_csv=os.path.join(INPUTS, "Demand_1sc_2y_gili_ketapang_scpos2.csv"),
    res_csv=os.path.join(INPUTS, "RES_Time_Series_1sc_gili_ketapang_scpos2.csv"),
    max_battery_kwh=25000.0, max_generator_kw=4000.0)

fails = 0


def check(cond, msg):
    global fails
    print(("[OK]   " if cond else "[FAIL] ") + msg)
    if not cond:
        fails += 1


def var_kind(m, name):
    attrs = m.variables[name].attrs
    return "binary" if attrs.get("binary") else "integer" if attrs.get("integer") else "continuous"


inp_partial = DesignInputs(**PARTIAL)
inp_plain = DesignInputs(**PLAIN)

# 1. resolve_plain_mode
check(resolve_plain_mode(inp_partial, env={}) is False, "Generator_Partial_Load=1, LINOPY_PLAIN unset -> partial-load")
check(resolve_plain_mode(inp_plain, env={}) is True, "Generator_Partial_Load=0, LINOPY_PLAIN unset -> plain")
check(resolve_plain_mode(inp_plain, env={"LINOPY_PLAIN": "1"}) is True, "agreeing LINOPY_PLAIN=1 accepted")
check(resolve_plain_mode(inp_partial, env={"LINOPY_PLAIN": "0"}) is False, "agreeing LINOPY_PLAIN=0 accepted")
for inp, raw in ((inp_partial, "1"), (inp_plain, "0")):
    try:
        resolve_plain_mode(inp, env={"LINOPY_PLAIN": raw})
        check(False, f"contradicting LINOPY_PLAIN={raw} should raise")
    except ValueError:
        check(True, f"contradicting LINOPY_PLAIN={raw} raises ValueError")

# 2./3. variable domains (build only)
for label, inp, plain, uc in (("partial-load, default", inp_partial, False, False),
                              ("partial-load, uc_integral", inp_partial, False, True),
                              ("plain, uc_integral", inp_plain, True, True)):
    built = build_core_model(inp, bess_mode="binary" if uc else "relaxed", plain_mode=plain, uc_integral=uc)
    kinds = (var_kind(built["m"], "Generator_Partial"), var_kind(built["m"], "Generator_Full"))
    expected = ("binary", "integer") if (uc and not plain) else ("continuous", "continuous")
    check(kinds == expected, f"{label}: Generator_Partial/Full = {kinds}, expected {expected}")
    check("Generator_Full_unused_parallel" not in built["m"].constraints,
          f"{label}: no parallel-mode constraint by default")

# 4. parallel load sharing
for label, uc, expected, pinned in (("partial-load, uc_integral, gen_parallel", True, ("integer", "integer"), True),
                                    ("partial-load, relaxed, gen_parallel", False, ("continuous", "continuous"), False)):
    built = build_core_model(inp_partial, bess_mode="binary" if uc else "relaxed", uc_integral=uc, gen_parallel=True)
    kinds = (var_kind(built["m"], "Generator_Partial"), var_kind(built["m"], "Generator_Full"))
    check(kinds == expected, f"{label}: Generator_Partial/Full = {kinds}, expected {expected}")
    check(("Generator_Full_unused_parallel" in built["m"].constraints) == pinned,
          f"{label}: Generator_Full pinned to 0 = {pinned}")

print("ALL PASSED" if fails == 0 else f"{fails} FAILURE(S)")
sys.exit(fails)
