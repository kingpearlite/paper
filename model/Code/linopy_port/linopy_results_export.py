"""Runs a linopy free-sizing solve (via linopy_free_sizing_crosscheck.py) and feeds the solved
values into the REAL project's own Results.py (Code/Model/Results.py), UNMODIFIED, so a linopy
run produces the exact same Code/Results/<RUN_ID>/Results_Summary.xlsx +
Time_Series_SC_<s>.xlsx output a Pyomo run would -- instead of only ever printing NPC + sizing
to stdout, which is all the crosscheck scripts did before this file existed.

HOW: `pyomo_instance_adapter.py`'s PyomoInstanceAdapter mimics Pyomo's `instance.X.value` /
`.get_values()` / `.extract_values()` API, backed by dicts built from this solve's values.
Results.py's own ResultsSummary/TimeSeries/PrintResults functions run against that adapter
exactly as they would against a real Pyomo ConcreteModel instance.

SCOPE (2026-08-29, updated through Stage 5): populated for Model_Components=0 (RES+Battery+
Generator) only -- Model_Components 1/2 (drop generator or battery entirely) remain unported,
a separate structural variant deferred beyond the current roadmap. Grid_Connection, Land_Use,
Fuel_Specific_Cost_Calculation, and WACC_Calculation are now read from the site's own .dat
(Stages 1-5) instead of hardcoded off; Multiobjective_Optimization=0 (single-objective NPC
minimization) remains hardcoded (Stage 6, not yet ported). Results.py's own
`if instance.Grid_Connection...` / `if instance.Land_Use...` /
`if instance.Multiobjective_Optimization...` guards skip those code paths when the
corresponding flag is 0, so nothing beyond a design's own real parameter set needs supplying.

CO2 EMISSIONS (Stage 1 off-grid, Stage 5 added GRID_emission): computed via
linopy_core_model_builder.compute_co2_emissions(), covering RES/BESS/GEN lifecycle emissions,
fuel-burn emissions, and (when grid-connected) grid-import emissions. NOT YET CROSS-VALIDATED
against Pyomo's own CO2 figure for any real design (local-laptop smoke-tested only) -- do this
before calling a CO2 number here "certified", same bar as every other figure in this pipeline.

GRID CONNECTION (Stage 5, full-parity roadmap): Energy_From_Grid/Energy_To_Grid, grid investment/
O&M cost, electricity purchase cost/export revenue, and grid salvage value are now real when
GRID_CONNECTION=1 (see linopy_core_model_builder.py's build_core_model() docstring). Grid
availability is a DELIBERATE SIMPLIFICATION: always 1.0 (grid never down) -- Pyomo's own
Grid_Availability always reads an external CSV (deterministic or Weibull-simulated); no such
CSV convention exists in this project since no Gili Ketapang design has ever used a real grid
connection, so it isn't ported. Model_Components 1/2's grid variants are correspondingly also
not covered (see SCOPE above). NOT CROSS-VALIDATED against Pyomo for any grid-connected design.

USAGE: LINOPY_CONFIRM_REAL_SCALE=1 LINOPY_DESIGN=capexp5_6sc python linopy_results_export.py [tag]
(same env vars as linopy_free_sizing_crosscheck.py -- importing that module IS how this script
triggers the actual solve, see the `import linopy_free_sizing_crosscheck as solved` line below;
[tag] is an optional Results/<RUN_ID> folder-name suffix, same convention as Code/Model/run_id.py).
"""
import os
import sys

import numpy as np
import xarray as xr

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "Code", "Model")))

from pyomo_instance_adapter import PyomoInstanceAdapter, da_to_pyomo_dict  # noqa: E402
from linopy_core_model_builder import compute_co2_emissions  # noqa: E402 -- Stage 1, full-parity roadmap

# NOTE: the actual solve is triggered by importing linopy_free_sizing_crosscheck, which is
# deliberately deferred to the __main__ block below (not done at this module's import time) --
# so build_adapter() can be imported and reused (e.g. by _smoketest_results_export.py) without
# forcing a real Gurobi solve or tripping that module's own LINOPY_CONFIRM_REAL_SCALE gate.


def load_dat_indexed_string(path, name):
    lines = open(path, encoding="utf-8").readlines()
    for i, l in enumerate(lines):
        if f"param: {name} :=" in l:
            j, out = i + 1, {}
            while j < len(lines):
                stripped = lines[j].strip()
                if stripped == ";":
                    break
                parts = stripped.split(None, 1)
                if len(parts) == 2:
                    out[int(parts[0])] = parts[1].strip().strip("'").strip('"').rstrip(";").strip()
                j += 1
            return out
    raise ValueError(f"indexed string param {name} not found in {path}")


def load_dat_startdate(path):
    for l in open(path, encoding="utf-8"):
        if "param: StartDate" in l:
            return l.split(":=", 1)[1].split("#", 1)[0].strip().strip(";").strip().strip("'").strip('"')
    raise ValueError(f"StartDate not found in {path}")


def load_dat_scalar_string(path, name):
    # 2026-09-04, Plots.py wiring: same quoted-scalar-param parsing load_dat_startdate already
    # does for StartDate, generalized to any name (Battery_Color, Lost_Load_Color, etc.).
    for l in open(path, encoding="utf-8"):
        if f"param: {name} " in l or f"param: {name}:" in l:
            return l.split(":=", 1)[1].split("#", 1)[0].strip().strip(";").strip().strip("'").strip('"')
    raise ValueError(f"scalar string param {name} not found in {path}")


def build_adapter(solved_module):
    m = solved_module
    inp = m.inputs  # Stage 0 (full-parity roadmap): all .dat scalars/indices now live on
    # DesignInputs (linopy_core_model_builder.py) rather than as module-level globals on the
    # solved script -- every m.<PARAM> reference below that used to read a free_sizing global
    # now reads inp.<PARAM> instead. m.result/m.PARAMS_DAT/m.load_dat_scalar are unchanged.
    r_ = m.result
    S, Y, P, ST = inp.S, inp.YEARS, inp.PERIODS, inp.STEPS_NUMBER

    # lp_mode (Stage 8, full-parity roadmap): build_core_model() returns this in its result dict
    # (added 2026-09-04). Mirrors build_core_model()'s own scale trick exactly -- in LP mode,
    # Battery_Units_da/Generator_Units_da ARE Battery_Nominal_Capacity/Generator_Nominal_Capacity
    # directly (kWh/kW), so the per-unit "nominal size" used everywhere below to recover capacity
    # from a unit COUNT must be 1.0 for those two. plain_mode (True whenever lp_mode is True, but
    # can also be True on its own for a MILP plain-generator design) selects which fuel-cost
    # formula matches what the solve actually used.
    lp_mode = bool(r_.get("lp_mode", False))
    plain_mode = bool(r_.get("plain_mode", False))
    batt_unit_kwh = 1.0 if lp_mode else inp.BATT_NOM_CAP_KWH
    gen_unit_kw = 1.0 if lp_mode else inp.GEN_NOM_CAP_KW

    RES_Units_da = r_["RES_Units"].solution
    Battery_Units_da = r_["Battery_Units"].solution
    Generator_Units_da = r_["Generator_Units"].solution
    RES_Energy_Production_da = r_["RES_Energy_Production"].solution
    Generator_Energy_Total_da = r_["Generator_Energy_Total"].solution
    Generator_Energy_Partial_da = r_["Generator_Energy_Partial"].solution
    Generator_Partial_da = r_["Generator_Partial"].solution
    Generator_Full_da = r_["Generator_Full"].solution
    Battery_Inflow_da = r_["Battery_Inflow"].solution
    Battery_Outflow_da = r_["Battery_Outflow"].solution
    Battery_SOC_da = r_["Battery_SOC"].solution
    Energy_Curtailment_da = r_["Energy_Curtailment"].solution
    Lost_Load_da = r_["Lost_Load"].solution
    demand_da = r_["demand"]
    # Grid connection (Stage 5, full-parity roadmap) -- Energy_From_Grid/To_Grid are None in the
    # solved dict when GRID_CONNECTION=0 (build_core_model() never declared them); fall back to
    # zero DataArrays so every downstream computation below works unchanged in that case.
    Energy_From_Grid_da = r_["Energy_From_Grid"].solution if r_["Energy_From_Grid"] is not None else xr.zeros_like(Lost_Load_da)
    Energy_To_Grid_da = r_["Energy_To_Grid"].solution if r_["Energy_To_Grid"] is not None else xr.zeros_like(Lost_Load_da)

    disc_yt = (1 + inp.r) ** xr.DataArray(inp.yt_idx.values, coords={"yt": inp.yt_idx})

    # --- scalar/derived cost components, recomputed in plain python from solved sizing --------
    # (mirrors linopy_free_sizing_crosscheck.py's own formulas exactly -- kept independent of
    # relying on `.solution` on arbitrary LinearExpressions, only real Variables' `.solution`.)
    res_units_1 = float(RES_Units_da.sel(ut=1))
    bat_units_1 = float(Battery_Units_da.sel(ut=1))
    gen_units_1 = float(Generator_Units_da.sel(ut=1))
    inv_res = res_units_1 * inp.RES_NOM_CAP_KW * inp.RES_SPECIFIC_COST
    inv_gen = gen_units_1 * gen_unit_kw * inp.GEN_SPECIFIC_COST
    inv_bat = bat_units_1 * batt_unit_kwh * inp.BATT_SPECIFIC_COST
    investment_cost = inv_res + inv_gen + inv_bat
    # Net out the existing-capacity (brownfield) credit -- mirrors build_core_model()'s own
    # investment_const_offset exactly. Bug found 2026-09-04 via the Stage 1-7 Pyomo
    # cross-validation: this recompute previously left step-1 investment_cost UN-netted (raw
    # Units[1]*NomCap*SpecificCost, no credit subtracted), while the solve itself nets this via
    # a separate post-solve constant (linopy's objective can't hold a bare constant term, so
    # build_core_model() applies the credit AFTER solving, not inside the investment_cost
    # expression). The final NPC (`objective_value=r_["npc_true"]` below, "Weighted Net present
    # cost" in the exported Cost sheet) was ALWAYS correct, since it comes straight from
    # `m.objective.value + total_const_offset` -- but this un-netted `investment_cost` fed BOTH
    # the "Total Investment cost" cost-sheet row AND `npc_per_s` (the "Net present cost,
    # Scenario N" row) directly, silently overstating both by exactly the brownfield credit
    # magnitude for every design with nonzero RES/GEN/BATT existing capacity (i.e. every
    # certified Gili Ketapang design, which all credit the existing diesel plant). Caught via a
    # real $75,472 (470kW x $160.58/kW existing-generator credit) mismatch against a real Pyomo
    # export on the smoketest_1sc_2y cross-validation design.
    investment_cost -= (inp.RES_EXISTING_KW * inp.RES_SPECIFIC_COST
                         + inp.GEN_EXISTING_KW * inp.GEN_SPECIFIC_COST
                         + inp.BATT_EXISTING_KWH * inp.BATT_SPECIFIC_COST)
    for ut in inp.ut_idx[1:]:
        yt0 = inp.step_start_year[ut]
        disc = (1 + inp.r) ** (yt0 - 1)
        d_res = float(RES_Units_da.sel(ut=ut) - RES_Units_da.sel(ut=ut - 1))
        d_gen = float(Generator_Units_da.sel(ut=ut) - Generator_Units_da.sel(ut=ut - 1))
        d_bat = float(Battery_Units_da.sel(ut=ut) - Battery_Units_da.sel(ut=ut - 1))
        investment_cost += (d_res * inp.RES_NOM_CAP_KW * inp.RES_SPECIFIC_COST + d_gen * gen_unit_kw * inp.GEN_SPECIFIC_COST + d_bat * batt_unit_kwh * inp.BATT_SPECIFIC_COST) / disc

    om_cost = 0.0
    for yt in inp.yt_idx:
        disc = (1 + inp.r) ** yt
        ut = int(inp.ut_of.sel(yt=yt))
        res_u = float(RES_Units_da.sel(ut=ut))
        gen_u = float(Generator_Units_da.sel(ut=ut))
        bat_u = float(Battery_Units_da.sel(ut=ut))
        om_cost += (res_u * inp.RES_NOM_CAP_KW * inp.RES_SPECIFIC_COST * inp.RES_OM_FRAC
                    + gen_u * gen_unit_kw * inp.GEN_SPECIFIC_COST * inp.GEN_OM_FRAC
                    + bat_u * batt_unit_kwh * inp.BATT_SPECIFIC_COST * inp.BATT_OM_FRAC) / disc

    # Grid connection cost/O&M (Stage 5, full-parity roadmap) -- mirrors build_core_model()'s
    # own Inv_Grid/OyM_Grid accumulators exactly; zero when GRID_CONNECTION=0.
    if inp.GRID_CONNECTION == 1:
        for yt in inp.yt_idx:
            yt_val = int(yt)
            if yt_val >= inp.YEAR_GRID_CONNECTION:
                investment_cost += (inp.GRID_CONNECTION_COST * inp.GRID_DISTANCE) / (1 + inp.r) ** (yt_val - 1)
                om_cost += (inp.GRID_CONNECTION_COST * inp.GRID_DISTANCE * inp.GRID_MAINTENANCE_COST) / (1 + inp.r) ** yt_val

    def salvage_term(units_da, existing_kw, existing_years, nom_cap, specific_cost, lifetime):
        step1_factor = specific_cost * (lifetime - Y) / lifetime / (1 + inp.r) ** Y
        sv = float(units_da.sel(ut=1)) * nom_cap * step1_factor
        const = -existing_kw * step1_factor
        const += existing_kw * specific_cost * (lifetime - existing_years - Y) / lifetime / (1 + inp.r) ** Y
        for ut in inp.ut_idx[1:]:
            yt0 = inp.step_start_year[ut]
            increment = float(units_da.sel(ut=ut) - units_da.sel(ut=ut - 1)) * nom_cap
            sv += increment * specific_cost * (lifetime + (yt0 - 1) - Y) / lifetime / (1 + inp.r) ** Y
        return sv + const

    raw_salvage = salvage_term(RES_Units_da, inp.RES_EXISTING_KW, inp.RES_EXISTING_YEARS, inp.RES_NOM_CAP_KW, inp.RES_SPECIFIC_COST, inp.RES_LIFETIME) \
        + salvage_term(Generator_Units_da, inp.GEN_EXISTING_KW, inp.GEN_EXISTING_YEARS, gen_unit_kw, inp.GEN_SPECIFIC_COST, inp.GEN_LIFETIME)
    if inp.GRID_CONNECTION == 1:
        # SV_Grid (Stage 5) -- mirrors build_core_model()'s own grid salvage term exactly.
        raw_salvage += (inp.GRID_DISTANCE * inp.GRID_CONNECTION_COST) / (1 + inp.r) ** (Y - inp.YEAR_GRID_CONNECTION)
    salvage_value = max(0.0, raw_salvage)  # same floor-at-zero fix as the solve itself

    # --- per-scenario breakdowns (Results.py needs these un-collapsed across "s") -------------
    # Stage 3 (full-parity roadmap): mirror build_core_model()'s own fuel-cost branch exactly --
    # this reporting recompute must use the SAME per-year costs the objective was actually
    # solved against, or the exported Total_Fuel_Cost_Act would silently disagree with the NPC.
    if inp.FUEL_SPECIFIC_COST_CALCULATION == 1:
        gen_marginal_cost, gen_marginal_cost_milp_1, gen_start_cost_1 = (
            inp.gen_marginal_cost_yearly, inp.gen_marginal_cost_milp_yearly, inp.gen_start_cost_yearly)
    else:
        gen_marginal_cost, gen_marginal_cost_milp_1, gen_start_cost_1 = (
            inp.GEN_MARGINAL_COST, inp.GEN_MARGINAL_COST_MILP_1, inp.GEN_START_COST_1)
    # plain_mode (True whenever lp_mode is True, per build_core_model()'s own forcing) has no
    # partial-load/start-cost split -- Generator_Full/Partial/Energy_Partial are all pinned to 0
    # by build_core_model() in that branch, so the partial-load formula below would silently
    # recompute a zero fuel cost. Mirrors build_core_model()'s own plain_mode fuel_cost branch
    # exactly (2026-09-04 fix, found while wiring lp_mode through to this exporter).
    if plain_mode:
        fuel_per_s_da = ((Generator_Energy_Total_da * gen_marginal_cost) / disc_yt).sum(["yt", "t"])
    else:
        fuel_per_s_da = ((Generator_Full_da * gen_unit_kw * gen_marginal_cost
                          + Generator_Energy_Partial_da * gen_marginal_cost_milp_1
                          + Generator_Partial_da * gen_start_cost_1) / disc_yt).sum(["yt", "t"])
    lost_load_per_s_da = (Lost_Load_da * inp.LOST_LOAD_COST / disc_yt).sum(["yt", "t"])
    batt_repl_per_s_da = ((Battery_Inflow_da + Battery_Outflow_da) * inp.BATT_REPL_UNIT_COST / disc_yt).sum(["yt", "t"])

    fuel_per_s = {int(s): float(fuel_per_s_da.sel(s=s)) for s in inp.s_idx}
    lost_load_per_s = {int(s): float(lost_load_per_s_da.sel(s=s)) for s in inp.s_idx}
    batt_repl_per_s = {int(s): float(batt_repl_per_s_da.sel(s=s)) for s in inp.s_idx}
    # Variable O&M (2026-09-18): mirrors build_core_model()'s var_om_cost; zero unless the .dat
    # sets Generator_Variable_OM_Cost / Battery_Variable_OM_Cost. Reported inside the scenario's
    # variable cost so the exported NPC still equals the solver's.
    var_om_per_s_da = ((Generator_Energy_Total_da * inp.GEN_VAR_OM + Battery_Outflow_da * inp.BATT_VAR_OM) / disc_yt).sum(["yt", "t"])
    var_om_per_s = {int(s): float(var_om_per_s_da.sel(s=s)) for s in inp.s_idx}

    # Grid electricity purchase cost / export revenue per scenario (Stage 5, full-parity
    # roadmap) -- mirrors build_core_model()'s own grid_electricity_cost/revenue exactly; zero
    # when GRID_CONNECTION=0.
    grid_cost_per_s = {int(s): 0.0 for s in inp.s_idx}
    grid_revenue_per_s = {int(s): 0.0 for s in inp.s_idx}
    if inp.GRID_CONNECTION == 1:
        grid_cost_da = (Energy_From_Grid_da * inp.grid_availability * inp.GRID_PURCHASED_EL_PRICE / disc_yt).sum(["yt", "t"])
        grid_cost_per_s = {int(s): float(grid_cost_da.sel(s=s)) for s in inp.s_idx}
        if inp.GRID_CONNECTION_TYPE == 0:
            grid_revenue_da = (Energy_To_Grid_da * inp.grid_availability * inp.GRID_SOLD_EL_PRICE / disc_yt).sum(["yt", "t"])
            grid_revenue_per_s = {int(s): float(grid_revenue_da.sel(s=s)) for s in inp.s_idx}

    variable_cost_per_s = {s: om_cost + fuel_per_s[s] + lost_load_per_s[s] + batt_repl_per_s[s] + var_om_per_s[s] + grid_cost_per_s[s] - grid_revenue_per_s[s] for s in fuel_per_s}
    npc_per_s = {s: investment_cost + variable_cost_per_s[s] - salvage_value for s in fuel_per_s}
    fuel_cost_act_sg = {(s, 1): fuel_per_s[s] for s in fuel_per_s}

    PARAMS_DAT = m.PARAMS_DAT
    res_names = load_dat_indexed_string(PARAMS_DAT, "RES_Names")
    gen_names = load_dat_indexed_string(PARAMS_DAT, "Generator_Names")
    fuel_names = load_dat_indexed_string(PARAMS_DAT, "Fuel_Names")
    start_date = load_dat_startdate(PARAMS_DAT)
    # Plot colors (2026-09-04, Plots.py wiring) -- DispatchPlot/CashFlowPlot/SizePlot all read
    # these hex color codes directly off `instance`; never needed before since this exporter
    # previously only called ResultsSummary/TimeSeries/PrintResults, not the Plots.py functions.
    res_colors = load_dat_indexed_string(PARAMS_DAT, "RES_Colors")
    gen_colors = load_dat_indexed_string(PARAMS_DAT, "Generator_Colors")
    battery_color = load_dat_scalar_string(PARAMS_DAT, "Battery_Color")
    lost_load_color = load_dat_scalar_string(PARAMS_DAT, "Lost_Load_Color")
    curtailment_color = load_dat_scalar_string(PARAMS_DAT, "Curtailment_Color")
    energy_to_grid_color = load_dat_scalar_string(PARAMS_DAT, "Energy_To_Grid_Color")
    energy_from_grid_color = load_dat_scalar_string(PARAMS_DAT, "Energy_From_Grid_Color")

    params = dict(
        Scenarios=S, Periods=P, Years=Y, Steps_Number=ST, RES_Sources=1, Generator_Types=1,
        Step_Duration=inp.STEP_DURATION, Discount_Rate=inp.r,
        # Model_Components stays hardcoded 0 (RES+Battery+Generator) -- Model_Components 1/2 are
        # a separate, unported structural variant (see module docstring). Grid_Connection/_Type
        # (Stage 5, full-parity roadmap): now the site's real flags instead of hardcoded 0.
        Model_Components=0, Grid_Connection=inp.GRID_CONNECTION, Grid_Connection_Type=inp.GRID_CONNECTION_TYPE,
        # MILP_Formulation (Stage 8, full-parity roadmap): Results.py branches its OWN internal
        # cost-table recompute on this flag throughout (RES_Units_milp/Battery_Units/
        # Generator_Units + "_milp"-suffixed nominal capacities when 1, vs. RES_Units/
        # Battery_Nominal_Capacity/Generator_Nominal_Capacity read directly when 0) -- must match
        # what was actually solved, or Results.py reads the wrong attribute names/formula.
        MILP_Formulation=0 if lp_mode else 1,
        # Land_Use (Stage 2, full-parity roadmap): now read from the site's own .dat flag
        # instead of hardcoded 0, matching Renewables_Max_Land_Use's Pyomo-side gating.
        Land_Use=inp.LAND_USE, Renewables_Total_Area=inp.RENEWABLES_TOTAL_AREA,
        RES_Specific_Area={1: inp.RES_SPECIFIC_AREA},
        # WACC_Calculation (Stage 4, full-parity roadmap): real flag instead of hardcoded 0 --
        # Results.py's PrintResults() prints the computed WACC when this is 1 (L1906-1908).
        # inp.r already holds the WACC-derived rate when active (DesignInputs computed it),
        # so Discount_Rate below needs no separate branch.
        # Generator_Partial_Load (2026-09-11): the formulation actually solved, no longer
        # hardcoded 1 -- plain_mode is now derived from the .dat's own flag.
        Multiobjective_Optimization=0, Optimization_Goal=1, Generator_Partial_Load=0 if plain_mode else 1,
        WACC_Calculation=inp.WACC_CALCULATION,
        # Fuel_Specific_Cost_Calculation (Stage 3, full-parity roadmap): now the site's real
        # flag instead of hardcoded 0 -- Results.py branches on this (YearlyCosts(), L1350-1357)
        # to pick the flat "_1" params or the (g,y) year-varying ones below.
        # Greenfield_Investment (Stage 7, full-parity roadmap): real flag instead of hardcoded 0.
        Fuel_Specific_Cost_Calculation=inp.FUEL_SPECIFIC_COST_CALCULATION, Greenfield_Investment=inp.GREENFIELD_INVESTMENT,
        RES_Names=res_names, Generator_Names=gen_names, Fuel_Names=fuel_names, StartDate=start_date,
        # Delta_Time + plot colors (2026-09-04, Plots.py wiring) -- see the loading comment above.
        Delta_Time=inp.DELTA_TIME,
        RES_Colors=res_colors, Generator_Colors=gen_colors, Battery_Color=battery_color,
        Lost_Load_Color=lost_load_color, Curtailment_Color=curtailment_color,
        Energy_To_Grid_Color=energy_to_grid_color, Energy_From_Grid_Color=energy_from_grid_color,
        RES_Nominal_Capacity={1: inp.RES_NOM_CAP_KW}, RES_capacity={1: inp.RES_EXISTING_KW},
        RES_years={1: inp.RES_EXISTING_YEARS}, RES_Specific_Investment_Cost={1: inp.RES_SPECIFIC_COST},
        RES_Specific_OM_Cost={1: inp.RES_OM_FRAC}, RES_Lifetime={1: inp.RES_LIFETIME},
        Generator_Nominal_Capacity_milp={1: gen_unit_kw}, Generator_capacity={1: inp.GEN_EXISTING_KW},
        GEN_years={1: inp.GEN_EXISTING_YEARS}, Generator_Specific_Investment_Cost={1: inp.GEN_SPECIFIC_COST},
        Generator_Specific_OM_Cost={1: inp.GEN_OM_FRAC}, Generator_Lifetime={1: inp.GEN_LIFETIME},
        Generator_Efficiency={1: inp.GEN_EFFICIENCY}, Fuel_LHV={1: inp.FUEL_LHV},
        Generator_Marginal_Cost_1={1: inp.GEN_MARGINAL_COST}, Generator_Marginal_Cost_milp_1={1: inp.GEN_MARGINAL_COST_MILP_1},
        Generator_Start_Cost_1={1: inp.GEN_START_COST_1},
        # YearlyCosts() fetches ALL SIX Generator_Marginal/Start_Cost variants unconditionally
        # via .extract_values() but only actually INDEXES INTO the (g,y) ones when
        # Fuel_Specific_Cost_Calculation==1 (Results.py L1350/1354/1373). Stage 3 (full-parity
        # roadmap): when ==1, these are now DesignInputs' real escalation-computed per-year
        # values instead of the flat "_1" value mirrored across every year; when ==0 (every
        # certified design so far) the mirrored-flat form is harmless since it's never read.
        Generator_Marginal_Cost=(
            {(1, y): float(inp.gen_marginal_cost_yearly.sel(yt=y)) for y in range(1, Y + 1)}
            if inp.FUEL_SPECIFIC_COST_CALCULATION == 1
            else {(1, y): inp.GEN_MARGINAL_COST for y in range(1, Y + 1)}
        ),
        Generator_Marginal_Cost_milp=(
            {(1, y): float(inp.gen_marginal_cost_milp_yearly.sel(yt=y)) for y in range(1, Y + 1)}
            if inp.FUEL_SPECIFIC_COST_CALCULATION == 1
            else {(1, y): inp.GEN_MARGINAL_COST_MILP_1 for y in range(1, Y + 1)}
        ),
        Generator_Start_Cost=(
            {(1, y): float(inp.gen_start_cost_yearly.sel(yt=y)) for y in range(1, Y + 1)}
            if inp.FUEL_SPECIFIC_COST_CALCULATION == 1
            else {(1, y): inp.GEN_START_COST_1 for y in range(1, Y + 1)}
        ),
        Battery_Nominal_Capacity_milp=batt_unit_kwh, Battery_capacity=inp.BATT_EXISTING_KWH,
        Battery_Specific_Investment_Cost=inp.BATT_SPECIFIC_COST, Battery_Specific_OM_Cost=inp.BATT_OM_FRAC,
        Unitary_Battery_Replacement_Cost=inp.BATT_REPL_UNIT_COST,
        Lost_Load_Specific_Cost=inp.LOST_LOAD_COST,
        Scenario_Weight={int(s): float(inp.scenario_weight.sel(s=s)) for s in inp.s_idx},
        Energy_Demand=da_to_pyomo_dict(demand_da, dim_order=("s", "yt", "t")),
        # Grid_Connection=0 (every certified design so far) makes all of these inert
        # (EnergySystemCost's Grid_Investment line multiplies by Grid_Connection
        # unconditionally, so the params must still exist) -- real values from the .dat either
        # way, not placeholders. Maximum_Grid_Power/National_Grid_Specific_CO2_emissions/
        # Grid_Sold_El_Price/Grid_Purchased_El_Price added here (Stage 5) -- previously omitted
        # entirely since they were structurally unreachable while Grid_Connection was hardcoded 0.
        Grid_Connection_Cost=inp.GRID_CONNECTION_COST, Grid_Distance=inp.GRID_DISTANCE,
        Year_Grid_Connection=inp.YEAR_GRID_CONNECTION, Grid_Maintenance_Cost=inp.GRID_MAINTENANCE_COST,
        Maximum_Grid_Power=inp.MAXIMUM_GRID_POWER, National_Grid_Specific_CO2_emissions=inp.NATIONAL_GRID_CO2,
        Grid_Sold_El_Price=inp.GRID_SOLD_EL_PRICE, Grid_Purchased_El_Price=inp.GRID_PURCHASED_EL_PRICE,
    )

    # CO2 emissions (Stage 1 off-grid + Stage 5 grid-import term) -- real values instead of the
    # original zero-fill. energy_from_grid_da is the zero-filled fallback when GRID_CONNECTION=0,
    # which correctly makes Scenario_GRID_emission all-zero in that case (every certified design).
    co2 = compute_co2_emissions(inp, RES_Units_da, Battery_Units_da, Generator_Units_da, Generator_Energy_Total_da,
                                 energy_from_grid_da=Energy_From_Grid_da if inp.GRID_CONNECTION == 1 else None,
                                 lp_mode=lp_mode)

    variables = dict(
        Investment_Cost=investment_cost,
        Operation_Maintenance_Cost_Act=om_cost,
        Salvage_Value=salvage_value,
        Scenario_Net_Present_Cost=npc_per_s,
        Total_Scenario_Variable_Cost_Act=variable_cost_per_s,
        Total_Fuel_Cost_Act=fuel_cost_act_sg,
        Scenario_Lost_Load_Cost_Act=lost_load_per_s,
        Battery_Replacement_Cost_Act=batt_repl_per_s,
        Scenario_CO2_emission=co2["Scenario_CO2_emission"],
        Scenario_FUEL_emission=co2["Scenario_FUEL_emission"],
        Scenario_GRID_emission=co2["Scenario_GRID_emission"],
        RES_emission=co2["RES_emission"], GEN_emission=co2["GEN_emission"], BESS_emission=co2["BESS_emission"],
        FUEL_emission=da_to_pyomo_dict(co2["FUEL_emission"], dim_order=("s", "yt", "g", "t"), inject={"g": 1}),
        # Grid connection (Stage 5, full-parity roadmap): real values when GRID_CONNECTION=1
        # instead of always-zero -- TimeSeries() calls .get_values() on these UNCONDITIONALLY
        # (only their later USE is gated behind `if Grid_Connection==1`), so they must exist
        # either way; zero is still correct/inert for every certified (off-grid) design.
        Energy_From_Grid=da_to_pyomo_dict(Energy_From_Grid_da, dim_order=("s", "yt", "t")),
        Energy_To_Grid=da_to_pyomo_dict(Energy_To_Grid_da, dim_order=("s", "yt", "t")),
        Total_Electricity_Cost_Act=grid_cost_per_s, Total_Revenues_Act=grid_revenue_per_s,
        RES_Units_milp=da_to_pyomo_dict(RES_Units_da, dim_order=("ut", "r"), inject={"r": 1}),
        Battery_Units=da_to_pyomo_dict(Battery_Units_da, dim_order=("ut",)),
        Generator_Units=da_to_pyomo_dict(Generator_Units_da, dim_order=("ut", "g"), inject={"g": 1}),
        RES_Energy_Production=da_to_pyomo_dict(RES_Energy_Production_da, dim_order=("s", "yt", "r", "t"), inject={"r": 1}),
        Generator_Energy_Total=da_to_pyomo_dict(Generator_Energy_Total_da, dim_order=("s", "yt", "g", "t"), inject={"g": 1}),
        Generator_Energy_Partial=da_to_pyomo_dict(Generator_Energy_Partial_da, dim_order=("s", "yt", "g", "t"), inject={"g": 1}),
        Generator_Partial=da_to_pyomo_dict(Generator_Partial_da, dim_order=("s", "yt", "g", "t"), inject={"g": 1}),
        Generator_Full=da_to_pyomo_dict(Generator_Full_da, dim_order=("s", "yt", "g", "t"), inject={"g": 1}),
        Battery_Inflow=da_to_pyomo_dict(Battery_Inflow_da, dim_order=("s", "yt", "t")),
        Battery_Outflow=da_to_pyomo_dict(Battery_Outflow_da, dim_order=("s", "yt", "t")),
        Battery_SOC=da_to_pyomo_dict(Battery_SOC_da, dim_order=("s", "yt", "t")),
        Energy_Curtailment=da_to_pyomo_dict(Energy_Curtailment_da, dim_order=("s", "yt", "t")),
        Lost_Load=da_to_pyomo_dict(Lost_Load_da, dim_order=("s", "yt", "t")),
    )

    if lp_mode:
        # LP-native attribute names (Stage 8, full-parity roadmap) -- Results.py's
        # `if instance.MILP_Formulation.value:` branches (EnergySystemCost, TimeSeries,
        # SystemSize, SystemLand, renewable-penetration/generator-share tables -- audited across
        # every occurrence in Results.py, 2026-09-04) read a DIFFERENT set of attribute names in
        # their `else` (LP) branch than the MILP one already populated above:
        #   - RES_Units (not RES_Units_milp) -- same values, Results.py's own alternate name.
        #   - Battery_Nominal_Capacity / Generator_Nominal_Capacity, read directly (no Units *
        #     per-unit-size multiplication in Results.py's own LP code) -- exactly
        #     Battery_Units_da/Generator_Units_da's values, since build_core_model()'s lp_mode
        #     already makes those variables BE the capacity (see its own docstring).
        #   - Generator_Energy_Production (not Generator_Energy_Total) -- the single LP dispatch
        #     variable; identical values to Generator_Energy_Total_da since lp_mode forces
        #     plain_mode=True (no partial-load split to begin with).
        variables["RES_Units"] = variables["RES_Units_milp"]
        variables["Battery_Nominal_Capacity"] = da_to_pyomo_dict(Battery_Units_da, dim_order=("ut",))
        variables["Generator_Nominal_Capacity"] = da_to_pyomo_dict(Generator_Units_da, dim_order=("ut", "g"), inject={"g": 1})
        variables["Generator_Energy_Production"] = variables["Generator_Energy_Total"]

    # r_["npc_true"] is the ALREADY-VALIDATED weighted NPC from the solve itself (matches
    # Pyomo's ObjectiveFuntion.expr() semantics: investment + scenario-weighted variable costs
    # - salvage). Reusing it here rather than recomputing avoids a second, easy-to-get-subtly-
    # wrong derivation of the same number (variable costs are scenario-dependent and must be
    # weight-summed, not just added as the fixed O&M part alone).
    return PyomoInstanceAdapter(params, variables, objective_value=r_["npc_true"])


def export_adapter_to_results(adapter):
    """Shared tail for every linopy exporter (free-sizing here, fixed-design in
    linopy_fixed_design_export.py): given an adapter already built by build_adapter(), write
    Results_Summary.xlsx/Time_Series_SC_*.xlsx + the three Plots.py PNGs via the REAL project's
    own Results.py/Plots.py, unmodified. Callers control the Code/Results/<RUN_ID> tag the usual
    way (sys.argv[1], read once by Code/Model/run_id.py at Results.py's import time) -- this
    function does not touch sys.argv itself."""
    from Results import ResultsSummary, TimeSeries as BuildTimeSeries, PrintResults  # noqa: E402
    from Plots import DispatchPlot, CashFlowPlot, SizePlot  # noqa: E402 -- 2026-09-04, plot wiring

    print("\nExporting linopy results through Results.py (same format Pyomo runs produce)...")
    ts = BuildTimeSeries(adapter)
    results = ResultsSummary(adapter, adapter.Optimization_Goal.value, ts)

    # Plots (2026-09-04): same three calls, same default arguments, same order as MicroGrids.py's
    # own driver -- this exporter previously only wired up ResultsSummary/TimeSeries/PrintResults,
    # never Plots.py, so no PNGs were ever produced for a linopy-solved design. PlotDate matches
    # MicroGrids.py's own hardcoded default ('01/01/2023'), which lines up with every site's
    # StartDate convention (see that file's own comment on the same value).
    PlotScenario = 1
    PlotDate = "01/01/2023 00:00:00"
    PlotTime = 3
    PlotFormat = "png"
    PlotResolution = 400
    DispatchPlot(adapter, ts, PlotScenario, PlotDate, PlotTime, PlotResolution, PlotFormat)
    CashFlowPlot(adapter, results, PlotResolution, PlotFormat)
    SizePlot(adapter, results, PlotResolution, PlotFormat)

    PrintResults(adapter, results)
    print("\nDone -- see Code/Results/<RUN_ID>/ (RUN_ID printed by run_id.py's own import above if not visible, check Code/Results/ for the newest folder).")


if __name__ == "__main__":
    # Importing this SOLVES the model (same LINOPY_CONFIRM_REAL_SCALE/LINOPY_DESIGN/
    # LINOPY_WARMSTART_MST env-var gate as running it directly) -- deferred to here (not done at
    # this module's own import time) so build_adapter() stays importable/reusable without
    # forcing a real Gurobi solve as a side effect (see _smoketest_results_export.py).
    import linopy_free_sizing_crosscheck as solved

    if solved.result["status"] != "ok":
        # A time- or memory-limited solve keeps a feasible incumbent; exporting it is useful (the
        # 12sc 5-step run lost its design this way), but it is NOT a proven optimum, so it needs
        # LINOPY_EXPORT_SUBOPTIMAL=1 and says so loudly (2026-09-12).
        if os.environ.get("LINOPY_EXPORT_SUBOPTIMAL") == "1" and solved.result.get("npc_true") is not None:
            print("=" * 78)
            print(f"WARNING: exporting an UNPROVEN incumbent -- status {solved.result['status']}/"
                  f"{solved.result['condition']}, NPC {solved.result['npc_true']:,.2f}.")
            print("This design is feasible but not proven optimal; read the solver's own gap.")
            print("=" * 78)
        else:
            raise SystemExit(f"Solve did not reach status 'ok' (got {solved.result['status']}/{solved.result['condition']}) -- not exporting Results.")

    adapter = build_adapter(solved)
    export_adapter_to_results(adapter)
