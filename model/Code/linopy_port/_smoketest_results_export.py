"""Smoke test for linopy_results_export.py's NEW code (pyomo_instance_adapter.py +
build_adapter()) -- the part with zero test coverage, since the free-sizing solve machinery
itself is already extensively validated elsewhere. Deliberately bypasses Gurobi/linopy solving
entirely: hand-builds a tiny fake "solved module" (S=2, YEARS=2, PERIODS=24, STEP_DURATION=1,
STEPS_NUMBER=2) with the exact attribute surface build_adapter() reads, then runs the real
build_adapter() + Results.py's ResultsSummary/TimeSeries/PrintResults against it. If this
succeeds, the adapter mechanism and Results.py wiring are proven correct; if it fails, the
failure is in this new code, not in Gurobi/the real solve (which this test never touches).

Run: (mgpy conda env) python _smoketest_results_export.py
"""
import os
import sys
import types

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "Code", "Model")))

from pyomo_instance_adapter import PyomoInstanceAdapter, da_to_pyomo_dict  # noqa: E402
import linopy_results_export as export_mod  # noqa: E402 -- NOT triggering its __main__ solve (guarded), just reusing build_adapter/loaders


def load_dat_scalar(path, name):
    lines = open(path, encoding="utf-8").readlines()
    for i, l in enumerate(lines):
        if f"param: {name} " in l or f"param: {name}:" in l:
            if ":=" in l:
                head = l.split(":=", 1)[1].split("#", 1)[0].strip()
                if head.rstrip(";").strip():
                    return float(head.rstrip(";").strip())
            j = i + 1
            while j < len(lines):
                stripped = lines[j].strip()
                if stripped and stripped != ";" and "\t" in stripped:
                    return float(stripped.split("\t")[1])
                if ";" in stripped and stripped != ";":
                    break
                j += 1
    raise ValueError(f"param {name} not found in {path}")


class _FakeVar:
    """Mimics a linopy Variable enough for build_adapter(): just needs `.solution`."""
    def __init__(self, da):
        self.solution = da


def build_fake_module():
    S, YEARS, PERIODS, STEP_DURATION, STEPS_NUMBER = 2, 2, 24, 1, 2
    s_idx = pd.Index(range(1, S + 1), name="s")
    yt_idx = pd.Index(range(1, YEARS + 1), name="yt")
    t_idx = pd.Index(range(1, PERIODS + 1), name="t")
    ut_idx = pd.Index(range(1, STEPS_NUMBER + 1), name="ut")
    ut_of = xr.DataArray([1, 2], coords={"yt": yt_idx})  # yt=1->ut=1, yt=2->ut=2 (STEP_DURATION=1)
    step_start_year = {1: 1, 2: 2}

    def sizing_da(vals):
        return xr.DataArray(np.array(vals, dtype=float), coords={"ut": ut_idx})

    def dispatch_da(fill):
        return xr.DataArray(np.full((S, YEARS, PERIODS), fill, dtype=float), coords={"s": s_idx, "yt": yt_idx, "t": t_idx})

    RES_Units = sizing_da([10.0, 15.0])          # cumulative units per step
    Battery_Units = sizing_da([5.0, 8.0])
    Generator_Units = sizing_da([2.0, 3.0])

    demand = dispatch_da(10.0)
    RES_Energy_Production = dispatch_da(4.0)
    Generator_Energy_Total = dispatch_da(6.0)
    Generator_Energy_Partial = dispatch_da(3.0)
    Generator_Partial = dispatch_da(0.5)   # fractional -- fine, relaxed dispatch, matches real solves
    Generator_Full = dispatch_da(1.0)
    Battery_Inflow = dispatch_da(1.0)
    Battery_Outflow = dispatch_da(1.0)
    Battery_SOC = dispatch_da(5.0)
    Energy_Curtailment = dispatch_da(0.0)
    Lost_Load = dispatch_da(0.0)

    real_dat = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "Code", "Inputs", "Parameters_6sc_20y_gili_ketapang_capexp5_MILP_dieselops.dat"))

    RES_NOM_CAP_KW = load_dat_scalar(real_dat, "RES_Nominal_Capacity")
    RES_SPECIFIC_COST = load_dat_scalar(real_dat, "RES_Specific_Investment_Cost")
    RES_OM_FRAC = load_dat_scalar(real_dat, "RES_Specific_OM_Cost")
    RES_LIFETIME = load_dat_scalar(real_dat, "RES_Lifetime")
    RES_EXISTING_KW = load_dat_scalar(real_dat, "RES_capacity")
    RES_EXISTING_YEARS = load_dat_scalar(real_dat, "RES_years")
    GEN_NOM_CAP_KW = load_dat_scalar(real_dat, "Generator_Nominal_Capacity_milp")
    GEN_SPECIFIC_COST = load_dat_scalar(real_dat, "Generator_Specific_Investment_Cost")
    GEN_OM_FRAC = load_dat_scalar(real_dat, "Generator_Specific_OM_Cost")
    GEN_LIFETIME = load_dat_scalar(real_dat, "Generator_Lifetime")
    GEN_EXISTING_KW = load_dat_scalar(real_dat, "Generator_capacity")
    GEN_EXISTING_YEARS = load_dat_scalar(real_dat, "GEN_years")
    GEN_EFFICIENCY = load_dat_scalar(real_dat, "Generator_Efficiency")
    FUEL_LHV = load_dat_scalar(real_dat, "Fuel_LHV")
    FUEL_SPECIFIC_COST_1 = load_dat_scalar(real_dat, "Fuel_Specific_Start_Cost")
    GEN_PGEN = load_dat_scalar(real_dat, "Generator_pgen")
    GEN_MARGINAL_COST = FUEL_SPECIFIC_COST_1 / (FUEL_LHV * GEN_EFFICIENCY)
    GEN_START_COST_1 = GEN_MARGINAL_COST * GEN_NOM_CAP_KW * GEN_PGEN
    GEN_MARGINAL_COST_MILP_1 = (GEN_MARGINAL_COST * GEN_NOM_CAP_KW - GEN_START_COST_1) / GEN_NOM_CAP_KW
    BATT_NOM_CAP_KWH = load_dat_scalar(real_dat, "Battery_Nominal_Capacity_milp")
    BATT_SPECIFIC_COST = load_dat_scalar(real_dat, "Battery_Specific_Investment_Cost")
    BATT_ELECTRONIC_COST = load_dat_scalar(real_dat, "Battery_Specific_Electronic_Investment_Cost")
    BATT_OM_FRAC = load_dat_scalar(real_dat, "Battery_Specific_OM_Cost")
    BATT_EXISTING_KWH = load_dat_scalar(real_dat, "Battery_capacity")
    BATT_CYCLES = load_dat_scalar(real_dat, "Battery_Cycles")
    BATT_DOD = load_dat_scalar(real_dat, "Battery_Depth_of_Discharge")
    BATT_REPL_UNIT_COST = (BATT_SPECIFIC_COST - BATT_ELECTRONIC_COST) / (BATT_CYCLES * 2 * BATT_DOD)
    LOST_LOAD_COST = load_dat_scalar(real_dat, "Lost_Load_Specific_Cost")
    r = load_dat_scalar(real_dat, "Real_Discount_Rate")

    # Stages 2-5 (full-parity roadmap) added these to DesignInputs -- build_adapter() now reads
    # them unconditionally (even off-grid/Land_Use=0/etc., to build the params dict correctly),
    # so the fake namespace needs them too or every one of those attribute reads throws.
    # LAND_USE/WACC_CALCULATION/FUEL_SPECIFIC_COST_CALCULATION/GRID_CONNECTION all 0/off here --
    # this smoke test covers the everyday (all features off) adapter path; each stage's own real
    # functional test (throwaway .dat copies, see linopy_free_sizing_crosscheck.py's comments)
    # covers the feature-on path against a real solve instead of this synthetic fixture.
    LAND_USE = int(load_dat_scalar(real_dat, "Land_Use"))
    RENEWABLES_TOTAL_AREA = load_dat_scalar(real_dat, "Renewables_Total_Area")
    RES_SPECIFIC_AREA = load_dat_scalar(real_dat, "RES_Specific_Area")
    RES_EXISTING_AREA = load_dat_scalar(real_dat, "RES_existing_area")
    WACC_CALCULATION = int(load_dat_scalar(real_dat, "WACC_Calculation"))
    FUEL_SPECIFIC_COST_CALCULATION = int(load_dat_scalar(real_dat, "Fuel_Specific_Cost_Calculation"))
    RES_UNIT_CO2 = load_dat_scalar(real_dat, "RES_unit_CO2_emission")
    BESS_UNIT_CO2 = load_dat_scalar(real_dat, "BESS_unit_CO2_emission")
    GEN_UNIT_CO2 = load_dat_scalar(real_dat, "GEN_unit_CO2_emission")
    FUEL_UNIT_CO2 = load_dat_scalar(real_dat, "FUEL_unit_CO2_emission")
    GRID_CONNECTION = int(load_dat_scalar(real_dat, "Grid_Connection"))
    GRID_CONNECTION_TYPE = int(load_dat_scalar(real_dat, "Grid_Connection_Type"))
    YEAR_GRID_CONNECTION = load_dat_scalar(real_dat, "Year_Grid_Connection")
    GRID_SOLD_EL_PRICE = load_dat_scalar(real_dat, "Grid_Sold_El_Price")
    GRID_PURCHASED_EL_PRICE = load_dat_scalar(real_dat, "Grid_Purchased_El_Price")
    GRID_DISTANCE = load_dat_scalar(real_dat, "Grid_Distance")
    GRID_CONNECTION_COST = load_dat_scalar(real_dat, "Grid_Connection_Cost")
    GRID_MAINTENANCE_COST = load_dat_scalar(real_dat, "Grid_Maintenance_Cost")
    MAXIMUM_GRID_POWER = load_dat_scalar(real_dat, "Maximum_Grid_Power")
    NATIONAL_GRID_CO2 = load_dat_scalar(real_dat, "National_Grid_Specific_CO2_emissions")
    grid_availability = xr.DataArray(np.ones((S, YEARS, PERIODS)), coords={"s": s_idx, "yt": yt_idx, "t": t_idx})
    GREENFIELD_INVESTMENT = int(load_dat_scalar(real_dat, "Greenfield_Investment"))  # Stage 7

    m = types.SimpleNamespace(
        GREENFIELD_INVESTMENT=GREENFIELD_INVESTMENT,
        LAND_USE=LAND_USE, RENEWABLES_TOTAL_AREA=RENEWABLES_TOTAL_AREA, RES_SPECIFIC_AREA=RES_SPECIFIC_AREA,
        RES_EXISTING_AREA=RES_EXISTING_AREA, WACC_CALCULATION=WACC_CALCULATION,
        FUEL_SPECIFIC_COST_CALCULATION=FUEL_SPECIFIC_COST_CALCULATION,
        RES_UNIT_CO2=RES_UNIT_CO2, BESS_UNIT_CO2=BESS_UNIT_CO2, GEN_UNIT_CO2=GEN_UNIT_CO2, FUEL_UNIT_CO2=FUEL_UNIT_CO2,
        GRID_CONNECTION=GRID_CONNECTION, GRID_CONNECTION_TYPE=GRID_CONNECTION_TYPE,
        YEAR_GRID_CONNECTION=YEAR_GRID_CONNECTION, GRID_SOLD_EL_PRICE=GRID_SOLD_EL_PRICE,
        GRID_PURCHASED_EL_PRICE=GRID_PURCHASED_EL_PRICE, GRID_DISTANCE=GRID_DISTANCE,
        GRID_CONNECTION_COST=GRID_CONNECTION_COST, GRID_MAINTENANCE_COST=GRID_MAINTENANCE_COST,
        MAXIMUM_GRID_POWER=MAXIMUM_GRID_POWER, NATIONAL_GRID_CO2=NATIONAL_GRID_CO2,
        grid_availability=grid_availability,
        S=S, YEARS=YEARS, PERIODS=PERIODS, STEP_DURATION=STEP_DURATION, STEPS_NUMBER=STEPS_NUMBER,
        s_idx=s_idx, yt_idx=yt_idx, t_idx=t_idx, ut_idx=ut_idx, ut_of=ut_of, step_start_year=step_start_year,
        r=r, RES_NOM_CAP_KW=RES_NOM_CAP_KW, GEN_NOM_CAP_KW=GEN_NOM_CAP_KW, BATT_NOM_CAP_KWH=BATT_NOM_CAP_KWH,
        RES_SPECIFIC_COST=RES_SPECIFIC_COST, GEN_SPECIFIC_COST=GEN_SPECIFIC_COST, BATT_SPECIFIC_COST=BATT_SPECIFIC_COST,
        RES_OM_FRAC=RES_OM_FRAC, GEN_OM_FRAC=GEN_OM_FRAC, BATT_OM_FRAC=BATT_OM_FRAC,
        RES_EXISTING_KW=RES_EXISTING_KW, GEN_EXISTING_KW=GEN_EXISTING_KW, BATT_EXISTING_KWH=BATT_EXISTING_KWH,
        RES_EXISTING_YEARS=RES_EXISTING_YEARS, GEN_EXISTING_YEARS=GEN_EXISTING_YEARS,
        RES_LIFETIME=RES_LIFETIME, GEN_LIFETIME=GEN_LIFETIME,
        GEN_MARGINAL_COST=GEN_MARGINAL_COST, GEN_MARGINAL_COST_MILP_1=GEN_MARGINAL_COST_MILP_1, GEN_START_COST_1=GEN_START_COST_1,
        GEN_EFFICIENCY=GEN_EFFICIENCY, FUEL_LHV=FUEL_LHV,
        BATT_REPL_UNIT_COST=BATT_REPL_UNIT_COST, LOST_LOAD_COST=LOST_LOAD_COST,
        DELTA_TIME=1.0,  # 2026-09-04, Plots.py wiring: build_adapter() now reads inp.DELTA_TIME unconditionally
        PARAMS_DAT=real_dat, load_dat_scalar=load_dat_scalar,
        result=dict(
            # Placeholder -- this fake dataset isn't economically meaningful, just needs to be a
            # real number so PyomoInstanceAdapter populates ObjectiveFuntion (None means "no
            # objective", which build_adapter() would otherwise silently skip wiring up).
            status="ok", condition="optimal", npc_true=250000.0,
            RES_Units=_FakeVar(RES_Units), Battery_Units=_FakeVar(Battery_Units), Generator_Units=_FakeVar(Generator_Units),
            RES_Energy_Production=_FakeVar(RES_Energy_Production), Generator_Energy_Total=_FakeVar(Generator_Energy_Total),
            Generator_Energy_Partial=_FakeVar(Generator_Energy_Partial), Generator_Partial=_FakeVar(Generator_Partial),
            Generator_Full=_FakeVar(Generator_Full), Battery_Inflow=_FakeVar(Battery_Inflow), Battery_Outflow=_FakeVar(Battery_Outflow),
            Battery_SOC=_FakeVar(Battery_SOC), Energy_Curtailment=_FakeVar(Energy_Curtailment), Lost_Load=_FakeVar(Lost_Load),
            demand=demand,
            # Stage 5 (full-parity roadmap): None here matches build_core_model()'s own return
            # when GRID_CONNECTION=0 (this fake fixture's case) -- build_adapter() falls back to
            # a zero DataArray for both when it sees None.
            Energy_From_Grid=None, Energy_To_Grid=None,
        ),
    )
    # Stage 0 (full-parity roadmap) moved build_adapter()'s scalar/index reads from `m.<ATTR>`
    # to `m.inputs.<ATTR>` (DesignInputs) -- this fake namespace keeps its flat layout above and
    # just points `.inputs` back at itself so `inp.RES_NOM_CAP_KW` etc. still resolve.
    m.inputs = m
    return m


if __name__ == "__main__":
    print("Building fake solved module (S=2, YEARS=2, PERIODS=24, STEPS_NUMBER=2)...")
    fake = build_fake_module()

    # run_id.py reads sys.argv[1] at IMPORT time (inside Results.py's own `from run_id import
    # RUN_ID`) -- must be set BEFORE the first `from Results import ...`, not after, or the tag
    # is silently ignored and RUN_ID falls back to a bare timestamp.
    sys.argv = [sys.argv[0], "SMOKETEST"]  # tags Results/<RUN_ID> so it's obvious to delete

    print("Calling build_adapter()...")
    adapter = export_mod.build_adapter(fake)
    print(f"  -> adapter built OK. npc_true used for objective = {fake.result['npc_true']}")

    print("Calling Results.py's TimeSeries()...")
    from Results import ResultsSummary, TimeSeries as BuildTimeSeries, PrintResults  # noqa: E402
    ts = BuildTimeSeries(adapter)
    print(f"  -> TimeSeries OK, {len(ts)} scenario(s), {len(ts[1])} year(s) each")

    print("Calling Results.py's ResultsSummary() (writes Code/Results/<RUN_ID>/Results_Summary.xlsx)...")
    results = ResultsSummary(adapter, adapter.Optimization_Goal.value, ts)
    print("  -> ResultsSummary OK. Sheets:", list(results.keys()))

    print("\nCalling PrintResults()...")
    PrintResults(adapter, results)

    print("\nSMOKE TEST PASSED -- adapter + Results.py wiring works end-to-end.")
