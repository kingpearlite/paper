"""Shared core-model builder for the linopy port -- Stage 0 of the full-parity roadmap
(see the approved plan, "Bring linopy to full feature parity with the Pyomo model (staged)").

WHY THIS EXISTS: linopy_free_sizing_crosscheck.py's build_and_solve_free_sizing() and
linopy_fixed_design_crosscheck.py's build_and_solve_relaxed() were two separately maintained,
near-identical implementations of the same RES/Battery/Generator/partial-load-generator
constraint set -- confirmed by diffing their function lists (only free_sizing/fixed_design's own
DESIGNS-dict plumbing and CLI-level printing differed). Every future feature (CO2, land use,
grid connection, ...) would otherwise need to be written and validated TWICE. This module is the
one place that logic now lives; both scripts import build_core_model() instead of building it
themselves.

TWO REAL DISCREPANCIES FOUND WHILE CONSOLIDATING (both fixed here, not silently):
1. Battery_Min_Capacity (the Battery_Independence autonomy-day floor) existed in
   linopy_free_sizing_crosscheck.py (added 2026-08-28, itself found missing via a Pyomo IIS) but
   was NEVER ported to linopy_fixed_design_crosscheck.py -- confirmed by grep, this constraint is
   now added unconditionally to the shared builder (battery_min_capacity=True default).
2. The Salvage_Value floor-at-zero Big-M fix (2026-08-28, also found via a Pyomo IIS on
   capexp5_6sc) existed in free-sizing but not in fixed-design, which used the raw, unfloored
   salvage formula. Now shared (salvage_floor=True default).
Both are exposed as constructor arguments defaulting to the CORRECT (post-fix) behavior, so a
caller can still reproduce the historical, pre-fix numerics exactly for a diagnostic comparison
-- see linopy_free_sizing_crosscheck.py's own LINOPY_HISTORICAL_REVERT flag, which maps to
salvage_floor=False here. Per the project's own established practice (Stage 0's plan entry), this
must be validated (not assumed a no-op) against every already-certified design's own recorded
NPC before either script is trusted post-refactor.

FOURTH DISCREPANCY (2026-09-14): the battery discharge power cap. Pyomo caps
Battery_Outflow at capacity / Maximum_Battery_Discharge_Time in every formulation -- inside the
mode gate in the bilinear form, as BatteryFlowDischargeCap (Constraints.py Max_Bat_flow_out_milp,
registered in Model_Resolution.py) in the linear and nobinary forms, and as Max_Bat_flow_out in
the LP classes. This builder only ever had the charge-side cap (Max_Bat_in). It is now added in
every mode (battery_discharge_cap=True default). It cannot bind where the battery holds at least
Maximum_Battery_Discharge_Time hours of peak demand, because Max_Bat_out already caps discharge
at demand; it did bind on the 12sc zero-autonomy design (3,479 kWh, 869.75 kW cap), whose
certified dispatch exceeded it in 51 of 2,102,400 scenario-hours.

THIRD DISCREPANCY: linopy_fixed_design_crosscheck.py never had a CPU/RAM resource monitor
(despite gili_ketapang_checklist_dlensemble.md's 31 Aug entry claiming one was "ported ... into
linopy_fixed_design_crosscheck.py, which hadn't had one" -- confirmed by grep, no such code was
present in the file this session found). start_resource_monitor/stop_resource_monitor below (moved
here verbatim from linopy_free_sizing_crosscheck.py) now give both scripts the same monitor.
"""
import os
import threading
import time

import numpy as np
import pandas as pd
import xarray as xr
import psutil
import linopy


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


def load_dat_indexed(path, name):
    """Reads an indexed .dat param block ('param: name := # comment' followed by 'k  value' lines,
    the last ending in ';') into {int(k): float(value)}."""
    lines = open(path, encoding="utf-8").readlines()
    for i, l in enumerate(lines):
        if l.startswith(f"param: {name} ") or l.startswith(f"param: {name}:"):
            vals = {}
            for raw in lines[i + 1:]:
                s = raw.split("#", 1)[0].strip()
                if not s:
                    continue
                end = s.endswith(";")
                s = s.rstrip(";").strip()
                if s:
                    k, v = s.split()
                    vals[int(k)] = float(v)
                if end:
                    return vals
            break
    raise ValueError(f"indexed param {name} not found (or not terminated by ';') in {path}")


def load_res_cf(path, periods_real, cf_floor=0.005, apply_floor=True):
    df = pd.read_csv(path, delimiter=",", decimal=".", header=0, index_col=0)
    df.columns = df.columns.astype(int)
    if apply_floor:
        df = df.where(df >= cf_floor, 0.0)
    return xr.DataArray(df.values.T, coords={"s": pd.Index(df.columns, name="s"), "t": pd.Index(range(1, periods_real + 1), name="t")})


def load_demand(path, s_real, years_real, periods_real):
    df = pd.read_csv(path, delimiter=";", decimal=",", header=0, index_col=0)
    data = np.empty((s_real, years_real, periods_real))
    for s in range(1, s_real + 1):
        for y in range(1, years_real + 1):
            data[s - 1, y - 1, :] = df[str((s - 1) * years_real + y)].to_numpy()
    return xr.DataArray(data, coords={"s": pd.Index(range(1, s_real + 1), name="s"), "yt": pd.Index(range(1, years_real + 1), name="yt"), "t": pd.Index(range(1, periods_real + 1), name="t")})


class DesignInputs:
    """Everything one design needs, loaded once. RES_CF_FLOOR=0.005 matches both original
    scripts' own documented floor (physically negligible dawn/dusk irradiance, <0.5% of
    nameplate -- see linopy_free_sizing_crosscheck.py's original comment for the full
    derivation)."""

    RES_CF_FLOOR = 0.005

    # No-op ceiling when max_battery_kwh/max_generator_kw is left unset -- mirrors Pyomo's own
    # opt-in convention for MGPY_MAX_BATTERY_KWH/MGPY_MAX_GENERATOR_KW exactly (Model_Creation.py
    # defaults both to 10**9 when the env var is unset, i.e. a ceiling large enough to never
    # actually bind). Unlike Pyomo, this port's Generator_Max_Units constraint and (in the
    # default "binary" bess_mode) Battery_Single_Flow_Charge constraint are always added, so a
    # real numeric ceiling is mandatory here -- this constant is what makes "unset" behave as
    # "opt-in, no-op" instead of a crash.
    NO_OP_CEILING = 1e9

    def __init__(self, params_dat, demand_csv, res_csv, max_battery_kwh=None, max_generator_kw=None,
                 apply_res_cf_floor=True):
        self.params_dat = params_dat
        self.S = int(load_dat_scalar(params_dat, "Scenarios"))
        self.YEARS = int(load_dat_scalar(params_dat, "Years"))
        self.PERIODS = int(load_dat_scalar(params_dat, "Periods"))
        self.STEP_DURATION = int(load_dat_scalar(params_dat, "Step_Duration"))
        self.STEPS_NUMBER = int(np.ceil(self.YEARS / self.STEP_DURATION))

        self.s_idx = pd.Index(range(1, self.S + 1), name="s")
        self.yt_idx = pd.Index(range(1, self.YEARS + 1), name="yt")
        self.t_idx = pd.Index(range(1, self.PERIODS + 1), name="t")
        self.ut_idx = pd.Index(range(1, self.STEPS_NUMBER + 1), name="ut")
        self.ut_of = xr.DataArray([int(np.ceil(yt / self.STEP_DURATION)) for yt in self.yt_idx], coords={"yt": self.yt_idx})
        self.step_start_year = {ut: (ut - 1) * self.STEP_DURATION + 1 for ut in self.ut_idx}
        # Scenario_Weight is read from the .dat, as Pyomo does (Model_Creation.py). Until
        # 2026-09-11 this was hard-coded to 1/S, which only matched because every design had
        # uniform weights (the .dat's 6-decimal 1/S values differ from exact 1/S by <= 2e-6); the
        # merged-demand designs have non-uniform TAKSR weights, which 1/S would have silently
        # replaced.
        w = load_dat_indexed(params_dat, "Scenario_Weight")
        if sorted(w) != list(range(1, self.S + 1)):
            raise ValueError(f"Scenario_Weight indices {sorted(w)} do not match Scenarios={self.S} in {params_dat}")
        w_arr = np.array([w[s] for s in self.s_idx])
        if (w_arr <= 0).any() or abs(w_arr.sum() - 1.0) > 1e-5:
            raise ValueError(f"Scenario_Weight must be positive and sum to 1, got sum {w_arr.sum():.8f} in {params_dat}")
        self.scenario_weight = xr.DataArray(w_arr, coords={"s": self.s_idx})

        # Unported-feature guards (2026-09-11, prerequisite for making linopy the primary solve
        # path). Before this, none of these three flags was ever read here, so a .dat setting any
        # of them non-zero would have solved SILENTLY as if it were 0 -- a wrong answer with no
        # warning, unlike Fuel_Specific_Cost_Import / Grid_Availability_Simulation below, which
        # already refuse. All certified Gili Ketapang .dat files set all three to 0 (checked 94/94
        # in Code/Inputs; the only non-zero ones are two legacy baseline/backup files no linopy
        # DESIGNS entry or sites/*.conf references).
        self.MODEL_COMPONENTS = int(load_dat_scalar(params_dat, "Model_Components"))
        if self.MODEL_COMPONENTS != 0:
            raise NotImplementedError(
                f"Model_Components={self.MODEL_COMPONENTS} (1=RES+Battery only, 2=RES+Generator only) "
                "is not ported -- this port always builds the full RES+Battery+Generator "
                "(Model_Components=0) formulation. Use the Pyomo path for this site.")
        # Pyomo's Initialize.py replaces the CSV with endogenously generated data when either flag
        # is 1 (NASA POWER fetch, L240; archetype demand generation, L126). This port only ever
        # reads the CSVs, so it would silently solve against DIFFERENT input data than Pyomo.
        if int(load_dat_scalar(params_dat, "RE_Supply_Calculation")) != 0:
            raise NotImplementedError(
                "RE_Supply_Calculation=1 (live NASA POWER fetch) is not ported -- this port always "
                "reads the pre-generated RES_Time_Series CSV. Generate the CSV first and set the "
                "flag to 0, or use the Pyomo path.")
        if int(load_dat_scalar(params_dat, "Demand_Profile_Generation")) != 0:
            raise NotImplementedError(
                "Demand_Profile_Generation=1 (archetype-based demand generation) is not ported -- "
                "this port always reads the pre-generated Demand CSV. Generate the CSV first and "
                "set the flag to 0, or use the Pyomo path.")
        # Generator_Partial_Load (2026-09-11): before this, the formulation was picked only by the
        # LINOPY_PLAIN env var (default: partial-load), never by the .dat -- 76/94 .dat files set
        # 0. Harmless while Generator_Partial/Full were always continuous (the relaxed unit
        # commitment then costs exactly E*marginal_cost, same as plain dispatch), but wrong as soon
        # as they are made integral. Callers derive plain_mode from this via resolve_plain_mode().
        self.GENERATOR_PARTIAL_LOAD = int(load_dat_scalar(params_dat, "Generator_Partial_Load"))

        self.cf = load_res_cf(res_csv, self.PERIODS, cf_floor=self.RES_CF_FLOOR, apply_floor=apply_res_cf_floor)
        self.demand = load_demand(demand_csv, self.S, self.YEARS, self.PERIODS)

        # WACC (Stage 4, full-parity roadmap) -- mirrors Initialize.py's Initialize_Discount_Rate
        # exactly: WACC_Calculation==0 uses the flat Real_Discount_Rate (every certified Gili
        # Ketapang design so far); ==1 derives the discount rate from financing structure
        # instead. All six params are declared in every .dat (confirmed: 85/85 files), even when
        # inert under WACC_Calculation=0.
        self.REAL_DISCOUNT_RATE_DEFAULT = load_dat_scalar(params_dat, "Real_Discount_Rate")
        self.WACC_CALCULATION = int(load_dat_scalar(params_dat, "WACC_Calculation"))
        self.COST_OF_EQUITY = load_dat_scalar(params_dat, "cost_of_equity")
        self.COST_OF_DEBT = load_dat_scalar(params_dat, "cost_of_debt")
        self.TAX = load_dat_scalar(params_dat, "tax")
        self.EQUITY_SHARE = load_dat_scalar(params_dat, "equity_share")
        self.DEBT_SHARE = load_dat_scalar(params_dat, "debt_share")
        if self.WACC_CALCULATION == 1:
            if self.EQUITY_SHARE == 0:  # pure debt financing -- after-tax cost of debt alone
                self.r = self.COST_OF_DEBT * (1 - self.TAX)
            else:
                leverage = self.DEBT_SHARE / self.EQUITY_SHARE
                self.r = (self.COST_OF_DEBT * (1 - self.TAX) * leverage / (1 + leverage)
                          + self.COST_OF_EQUITY * 1 / (1 + leverage))
        else:
            self.r = self.REAL_DISCOUNT_RATE_DEFAULT
        self.DELTA_TIME = load_dat_scalar(params_dat, "Delta_Time")
        self.RES_NOM_CAP_KW = load_dat_scalar(params_dat, "RES_Nominal_Capacity")
        self.RES_INVERTER_EFF = load_dat_scalar(params_dat, "RES_Inverter_Efficiency")
        self.RES_SPECIFIC_COST = load_dat_scalar(params_dat, "RES_Specific_Investment_Cost")
        self.RES_OM_FRAC = load_dat_scalar(params_dat, "RES_Specific_OM_Cost")
        self.RES_LIFETIME = load_dat_scalar(params_dat, "RES_Lifetime")
        self.RES_EXISTING_KW = load_dat_scalar(params_dat, "RES_capacity")
        self.RES_EXISTING_YEARS = load_dat_scalar(params_dat, "RES_years")
        self.GEN_NOM_CAP_KW = load_dat_scalar(params_dat, "Generator_Nominal_Capacity_milp")
        self.GEN_SPECIFIC_COST = load_dat_scalar(params_dat, "Generator_Specific_Investment_Cost")
        self.GEN_OM_FRAC = load_dat_scalar(params_dat, "Generator_Specific_OM_Cost")
        self.GEN_LIFETIME = load_dat_scalar(params_dat, "Generator_Lifetime")
        self.GEN_EXISTING_KW = load_dat_scalar(params_dat, "Generator_capacity")
        self.GEN_EXISTING_YEARS = load_dat_scalar(params_dat, "GEN_years")
        self.GEN_MIN_OUTPUT = load_dat_scalar(params_dat, "Generator_Min_output")
        self.GEN_PGEN = load_dat_scalar(params_dat, "Generator_pgen")
        self.GEN_EFFICIENCY = load_dat_scalar(params_dat, "Generator_Efficiency")
        self.FUEL_LHV = load_dat_scalar(params_dat, "Fuel_LHV")
        FUEL_SPECIFIC_COST_1 = load_dat_scalar(params_dat, "Fuel_Specific_Start_Cost")
        self.GEN_MARGINAL_COST = FUEL_SPECIFIC_COST_1 / (self.FUEL_LHV * self.GEN_EFFICIENCY)
        self.GEN_START_COST_1 = self.GEN_MARGINAL_COST * self.GEN_NOM_CAP_KW * self.GEN_PGEN
        self.GEN_MARGINAL_COST_MILP_1 = (self.GEN_MARGINAL_COST * self.GEN_NOM_CAP_KW - self.GEN_START_COST_1) / self.GEN_NOM_CAP_KW

        # Time-varying fuel cost (Stage 3, full-parity roadmap) -- Fuel_Specific_Cost_Calculation
        # (0=flat/constant, matching GEN_MARGINAL_COST above; 1=year-varying) and
        # Fuel_Specific_Cost_Import (only meaningful when ==1: 0=compute from an annual
        # escalation rate, 1=read a per-year series from an external Fuel_Costs.csv). Every .dat
        # in this project declares both flags and Fuel_Specific_Cost_Rate (confirmed: 85/85
        # files), even when Fuel_Specific_Cost_Calculation=0 makes them inert.
        self.FUEL_SPECIFIC_COST_CALCULATION = int(load_dat_scalar(params_dat, "Fuel_Specific_Cost_Calculation"))
        self.FUEL_SPECIFIC_COST_IMPORT = int(load_dat_scalar(params_dat, "Fuel_Specific_Cost_Import"))
        self.FUEL_SPECIFIC_COST_RATE = load_dat_scalar(params_dat, "Fuel_Specific_Cost_Rate")
        self.gen_marginal_cost_yearly = None       # xr.DataArray(yt) or None -- set below
        self.gen_marginal_cost_milp_yearly = None
        self.gen_start_cost_yearly = None
        if self.FUEL_SPECIFIC_COST_CALCULATION == 1:
            if self.FUEL_SPECIFIC_COST_IMPORT == 1:
                raise NotImplementedError(
                    "Fuel_Specific_Cost_Import=1 (year-varying fuel cost read from an external "
                    "Fuel_Costs.csv) is not yet ported -- Stage 3 of the full-parity roadmap only "
                    "covers the escalation-rate sub-mode (Fuel_Specific_Cost_Import=0), since no "
                    "Gili Ketapang design uses the import mode and there is no real Fuel_Costs.csv "
                    "to validate against yet."
                )
            # Mirrors Initialize.py's Initialize_Fuel_Specific_Cost's Fuel_Specific_Cost_Import==0
            # branch: year 1 = the base start cost, each later year = previous year's cost x
            # (1 + escalation rate) -- compounding, not simple/flat escalation off the base year.
            fuel_cost_by_year = {}
            previous_cost = FUEL_SPECIFIC_COST_1
            for yt in range(1, self.YEARS + 1):
                cost = previous_cost if yt == 1 else previous_cost * (1 + self.FUEL_SPECIFIC_COST_RATE)
                fuel_cost_by_year[yt] = cost
                previous_cost = cost
            marginal_cost_vals = [fuel_cost_by_year[yt] / (self.FUEL_LHV * self.GEN_EFFICIENCY) for yt in range(1, self.YEARS + 1)]
            self.gen_marginal_cost_yearly = xr.DataArray(marginal_cost_vals, coords={"yt": self.yt_idx})
            # Generator_Start_Cost[g,y] = Generator_Marginal_Cost[g,y] * Nominal_Capacity * pgen
            # (Initialize.py's Initialize_Generator_Start_Cost, Fuel_Specific_Cost_Calculation==1
            # branch); Generator_Marginal_Cost_milp[g,y] backs the start cost back out of the
            # full-capacity marginal cost (Initialize_Generator_Marginal_Cost_milp's == 1 branch).
            self.gen_start_cost_yearly = self.gen_marginal_cost_yearly * self.GEN_NOM_CAP_KW * self.GEN_PGEN
            self.gen_marginal_cost_milp_yearly = (self.gen_marginal_cost_yearly * self.GEN_NOM_CAP_KW - self.gen_start_cost_yearly) / self.GEN_NOM_CAP_KW

        # Grid connection (Stage 5, full-parity roadmap) -- these nine params are present in
        # every .dat (confirmed: 85/85), even when Grid_Connection=0 makes them inert (true of
        # every certified Gili Ketapang design). Large_Constant (the single-flow-grid big-M) is
        # NOT present in ANY .dat (0/85) -- no site has ever set Grid_Connection=1, so it was
        # never needed; falls back to 10x Maximum_Grid_Power when missing (matching this file's
        # existing BIG_M_SALVAGE-anchored-to-a-real-ceiling pattern). A real Grid_Connection=1
        # site should add a real Large_Constant to its own .dat rather than rely on this fallback.
        self.GRID_CONNECTION = int(load_dat_scalar(params_dat, "Grid_Connection"))
        self.GRID_CONNECTION_TYPE = int(load_dat_scalar(params_dat, "Grid_Connection_Type"))
        self.YEAR_GRID_CONNECTION = load_dat_scalar(params_dat, "Year_Grid_Connection")
        self.GRID_SOLD_EL_PRICE = load_dat_scalar(params_dat, "Grid_Sold_El_Price")
        self.GRID_PURCHASED_EL_PRICE = load_dat_scalar(params_dat, "Grid_Purchased_El_Price")
        self.GRID_DISTANCE = load_dat_scalar(params_dat, "Grid_Distance")
        self.GRID_CONNECTION_COST = load_dat_scalar(params_dat, "Grid_Connection_Cost")
        self.GRID_MAINTENANCE_COST = load_dat_scalar(params_dat, "Grid_Maintenance_Cost")
        self.MAXIMUM_GRID_POWER = load_dat_scalar(params_dat, "Maximum_Grid_Power")
        self.NATIONAL_GRID_CO2 = load_dat_scalar(params_dat, "National_Grid_Specific_CO2_emissions")
        try:
            self.GRID_LARGE_CONSTANT = load_dat_scalar(params_dat, "Large_Constant")
        except ValueError:
            self.GRID_LARGE_CONSTANT = self.MAXIMUM_GRID_POWER * 10

        # Grid_Availability: Pyomo ALWAYS reads this from an external CSV (either a
        # pre-supplied deterministic file, or one freshly generated by a stochastic Weibull
        # outage simulator when Grid_Availability_Simulation=1) -- there is no "just assume
        # always available" branch in Initialize.py at all. Since no Gili Ketapang site has ever
        # used Grid_Connection=1, no such CSV exists to establish or validate a file convention
        # against. Stage 5 therefore supports ONLY a deterministic always-available grid
        # (availability=1.0 for every scenario/year/period) as an explicit, documented
        # simplification -- NOT a port of Pyomo's file-based/stochastic availability modeling.
        self.GRID_AVAILABILITY_SIMULATION = int(load_dat_scalar(params_dat, "Grid_Availability_Simulation"))
        # Grid_Average_Number_Outages/Duration (2026-09-04, Stage 5 cross-validation): loaded so
        # the guard below can recognize the DEGENERATE Grid_Availability_Simulation=1 case --
        # Grid_Availability.py's own grid_availability() function (Pyomo side) special-cases
        # average_n_outages==0 and average_outage_duration==0 to write an all-ones (always
        # available) CSV, mathematically identical to this port's own hardcoded
        # `self.grid_availability = ones(...)` below. That specific case is genuinely equivalent,
        # not a stub -- confirmed by reading Grid_Availability.py directly and by a real
        # cross-validation run matching Pyomo exactly under it. Any NONZERO outage average still
        # means genuine Weibull stochastic simulation, which remains unported (raises below).
        self.GRID_AVERAGE_NUMBER_OUTAGES = load_dat_scalar(params_dat, "Grid_Average_Number_Outages")
        self.GRID_AVERAGE_OUTAGE_DURATION = load_dat_scalar(params_dat, "Grid_Average_Outage_Duration")
        grid_sim_is_degenerate_constant = (self.GRID_AVERAGE_NUMBER_OUTAGES == 0 and self.GRID_AVERAGE_OUTAGE_DURATION == 0)
        if self.GRID_CONNECTION == 1 and self.GRID_AVAILABILITY_SIMULATION != 0 and not grid_sim_is_degenerate_constant:
            raise NotImplementedError(
                "Grid_Availability_Simulation=1 with a nonzero Grid_Average_Number_Outages/"
                "Grid_Average_Outage_Duration (stochastic Weibull-simulated grid-availability "
                "time series) is not ported -- Stage 5 of the full-parity roadmap only supports "
                "a deterministic always-available grid (Grid_Availability=1.0 everywhere), since "
                "no Gili Ketapang design has ever used a real grid connection and there is no "
                "real availability CSV to validate against. (Grid_Availability_Simulation=1 with "
                "BOTH outage params at exactly 0 is allowed -- Pyomo's own Grid_Availability.py "
                "treats that as the same always-available case this port already implements.)"
            )
        self.grid_availability = xr.DataArray(
            np.ones((self.S, self.YEARS, self.PERIODS)),
            coords={"s": self.s_idx, "yt": self.yt_idx, "t": self.t_idx},
        )

        # Multi-objective (Stage 6, full-parity roadmap) -- present in every .dat (confirmed:
        # 85/85), even when Multiobjective_Optimization=0 (every certified design) makes them
        # inert. Pareto_solution (which single point of the sweep to actually export/return) is
        # loaded here for completeness but is a driver-level choice, not read inside
        # build_core_model()/run_pareto_sweep() themselves.
        self.MULTIOBJECTIVE_OPTIMIZATION = int(load_dat_scalar(params_dat, "Multiobjective_Optimization"))
        self.PARETO_POINTS = int(load_dat_scalar(params_dat, "Pareto_points"))
        self.PARETO_SOLUTION = int(load_dat_scalar(params_dat, "Pareto_solution"))
        self.PLOT_MAX_COST = int(load_dat_scalar(params_dat, "Plot_Max_Cost"))

        self.BATT_NOM_CAP_KWH = load_dat_scalar(params_dat, "Battery_Nominal_Capacity_milp")
        self.BATT_SPECIFIC_COST = load_dat_scalar(params_dat, "Battery_Specific_Investment_Cost")
        self.BATT_ELECTRONIC_COST = load_dat_scalar(params_dat, "Battery_Specific_Electronic_Investment_Cost")
        self.BATT_OM_FRAC = load_dat_scalar(params_dat, "Battery_Specific_OM_Cost")
        self.BATT_EXISTING_KWH = load_dat_scalar(params_dat, "Battery_capacity")
        self.BATT_CHARGE_EFF = load_dat_scalar(params_dat, "Battery_Charge_Battery_Efficiency")
        self.BATT_DISCHARGE_EFF = load_dat_scalar(params_dat, "Battery_Discharge_Battery_Efficiency")
        self.BATT_DOD = load_dat_scalar(params_dat, "Battery_Depth_of_Discharge")
        self.BATT_MIN_CHARGE_TIME = load_dat_scalar(params_dat, "Maximum_Battery_Charge_Time")
        self.BATT_MIN_DISCHARGE_TIME = load_dat_scalar(params_dat, "Maximum_Battery_Discharge_Time")
        self.BATT_INITIAL_SOC = load_dat_scalar(params_dat, "Battery_Initial_SOC")
        self.BATT_CYCLES = load_dat_scalar(params_dat, "Battery_Cycles")
        self.BATT_REPL_UNIT_COST = (self.BATT_SPECIFIC_COST - self.BATT_ELECTRONIC_COST) / (self.BATT_CYCLES * 2 * self.BATT_DOD)
        # Variable O&M (2026-09-18, linopy-only extension: MicroGridsPy/Pyomo models O&M only as a
        # fixed yearly fraction of capex). USD per kWh of genset output and per kWh discharged
        # from the battery, e.g. the Indonesian technology catalogue's 7.17 and 1.94 USD/MWh for
        # 2025. Optional: absent from a .dat means 0, and build_core_model() then adds nothing,
        # so every design certified before this change solves the identical model.
        try:
            self.GEN_VAR_OM = load_dat_scalar(params_dat, "Generator_Variable_OM_Cost")
        except ValueError:
            self.GEN_VAR_OM = 0.0
        try:
            self.BATT_VAR_OM = load_dat_scalar(params_dat, "Battery_Variable_OM_Cost")
        except ValueError:
            self.BATT_VAR_OM = 0.0
        self.BATT_MAX_INFLOW_KWH = max_battery_kwh if max_battery_kwh is not None else self.NO_OP_CEILING
        self.GEN_MAX_CAPACITY_KW = max_generator_kw if max_generator_kw is not None else self.NO_OP_CEILING
        self.LOST_LOAD_COST = load_dat_scalar(params_dat, "Lost_Load_Specific_Cost")
        self.LOST_LOAD_MAX_FRACTION = load_dat_scalar(params_dat, "Lost_Load_Fraction")
        self.BATT_INDEPENDENCE_DAYS = load_dat_scalar(params_dat, "Battery_Independence")

        # CO2 parameters (Stage 1) -- loaded here so DesignInputs stays the single place every
        # .dat parameter is read from; harmless to load even before Stage 1 wires them into the
        # objective/export, since load_dat_scalar just reads a number.
        self.RES_UNIT_CO2 = load_dat_scalar(params_dat, "RES_unit_CO2_emission")
        self.BESS_UNIT_CO2 = load_dat_scalar(params_dat, "BESS_unit_CO2_emission")
        self.GEN_UNIT_CO2 = load_dat_scalar(params_dat, "GEN_unit_CO2_emission")
        self.FUEL_UNIT_CO2 = load_dat_scalar(params_dat, "FUEL_unit_CO2_emission")

        # Land use (Stage 2) -- read straight from the .dat's own Land_Use flag, same as Pyomo
        # (Model_Resolution.py only builds RenewablesMaxLandUse when Land_Use==1); every .dat in
        # this project declares all four of these (confirmed: 85/85 files), even when Land_Use=0
        # makes them inert, so no fallback/default is needed here.
        self.LAND_USE = int(load_dat_scalar(params_dat, "Land_Use"))
        self.RENEWABLES_TOTAL_AREA = load_dat_scalar(params_dat, "Renewables_Total_Area")
        self.RES_SPECIFIC_AREA = load_dat_scalar(params_dat, "RES_Specific_Area")
        self.RES_EXISTING_AREA = load_dat_scalar(params_dat, "RES_existing_area")

        # Greenfield (Stage 7, full-parity roadmap). Pyomo's Constraints_Greenfield_Milp is a
        # SEPARATE class from Constraints_Brownfield_Milp, but line-by-line comparison
        # (Investment_Cost, Salvage_Value, RES/GEN/BESS_emission -- every formula that involves
        # RES_capacity/Generator_capacity/Battery_capacity at all) shows Greenfield's own
        # formulas are ALGEBRAICALLY IDENTICAL to Brownfield's with the existing-capacity credit
        # terms (RES_capacity/Generator_capacity/Battery_capacity/RES_years/GEN_years) all held
        # at exactly zero -- Energy_balance/Renewable_Energy/dispatch/O&M-cost never reference
        # existing capacity at all, so those are already identical between the two regardless.
        # This project's OWN `.dat` files already independently confirm this: every Brownfield
        # `.dat`'s `Greenfield_Investment` comment states it was "Validated ... against the
        # certified greenfield fixed-design result with all existing-capacity fields at 0
        # (bit-for-bit match)" on the Pyomo side. So Stage 7 needs no new constraint code at
        # all -- only forcing these five params to zero when Greenfield_Investment==1,
        # overriding whatever the .dat happens to say (mirroring Pyomo's own behavior of simply
        # never reading them in that class, rather than trusting every .dat author to have
        # zeroed them by hand).
        self.GREENFIELD_INVESTMENT = int(load_dat_scalar(params_dat, "Greenfield_Investment"))
        if self.GREENFIELD_INVESTMENT == 1:
            self.RES_EXISTING_KW = 0.0
            self.RES_EXISTING_YEARS = 0.0
            self.GEN_EXISTING_KW = 0.0
            self.GEN_EXISTING_YEARS = 0.0
            self.BATT_EXISTING_KWH = 0.0


def build_core_model(inp: DesignInputs, pinned=None, bess_mode="binary", plain_mode=False,
                      relax_gen_units=False, salvage_floor=True, battery_min_capacity=True,
                      big_m_salvage=2.0e6, objective="npc", co2_equals=None, lp_mode=False,
                      uc_integral=False, battery_discharge_cap=True, gen_parallel=False):
    """Builds the RES/Battery/Generator sizing + partial-load-generator + energy-balance model
    shared by free-sizing and fixed-design verification.

    pinned: None for free sizing (RES_Units/Battery_Units/Generator_Units left as free integer
        decision variables); or {"RES": DataArray, "Battery": DataArray, "Generator": DataArray}
        (indexed by inp.ut_idx) to add Fix_* equality constraints pinning an already-decided
        design (fixed-design verification).
    bess_mode: "binary" (true MILP mutual-exclusion Single_Flow_BESS, both gate constraints --
        the genuine formulation), "relaxed" (Single_Flow_BESS continuous in [0,1], both gate
        constraints still active -- the MGPY_FIX_RELAX_BINARIES=1 / fixed-design-verification
        equivalent), or "nobinary" (no Single_Flow_BESS variable or gate constraints at all --
        relies on round-trip losses making simultaneous charge/discharge uneconomic; ALWAYS
        check the returned Battery_Inflow/Battery_Outflow with a simultaneous-flow check before
        trusting a nobinary result).
    uc_integral: False (default -- Generator_Partial continuous in [0,1] and Generator_Full
        continuous >= 0, the MGPY_RELAX_UC=1 / MGPY_FIX_RELAX_BINARIES=1 equivalent that every
        solve before 2026-09-11 used) or True (Generator_Partial binary, Generator_Full integer:
        Pyomo's real Generator_Partial_Load=1 domains, for a true integral fixed-design re-solve).
        No effect under plain_mode, where both are pinned to 0.
    gen_parallel (2026-09-18): False (default -- Pyomo's own commitment: N_full integer units at
        full output plus AT MOST ONE unit (Generator_Partial binary) between min and full output)
        or True (parallel load sharing: Generator_Partial becomes an integer n >= 0, the number of
        units running and sharing the load, and Generator_Full is fixed to 0). Only acts together
        with uc_integral=True and not plain_mode. With it, the unchanged constraints below give
            n*Min_output*P <= E_GEN <= n*P,   n <= Generator_Units,
        and the unchanged fuel expression gives n*start_cost + marginal_milp*E_GEN, i.e. the
        clustered unit-commitment form for identical units. Why it exists: under the default,
        outputs between one unit at full load and one full unit plus a second at its minimum
        (470-587.5 kW for the Gili Ketapang 470 kW units at 25%) are unreachable, and since
        Generator_Energy_Total <= demand and Lost_Load = 0, storage must bridge every such hour.
        The real plant runs both units in parallel at low load above 420 kW (Falfi 2025, ITB
        thesis, p. 16), so that band is a property of the formulation, not of the plant. Every
        default-mode dispatch is feasible here at the same fuel cost (N_full full units plus one
        partial unit at e = n = N_full+1 units sharing N_full*P+e >= n*Min_output*P), so this is
        an exact relaxation: a pinned design's NPC can only fall. The relaxed branch
        (uc_integral=False) is left as it is on purpose: with Generator_Full continuous the
        minimum load never binds and fuel is marginal*E_GEN in both formulations, so their
        relaxations -- and hence every free-sizing result -- are identical.
    salvage_floor / battery_min_capacity / battery_discharge_cap: see this module's docstring --
        all default True (the Pyomo-faithful behavior); set False only to reproduce pre-fix
        historical numerics.
    objective: "npc" (default, minimize NPC -- every stage before Stage 6 only ever did this) or
        "co2" (minimize CO2_emission instead -- Stage 6's epsilon-constraint Pareto sweep needs
        this for its own "minimize emissions alone" solve). NPC is still built and returned
        either way; only which expression is actually handed to m.add_objective() changes.
    co2_equals: None (no constraint, default -- every stage before Stage 6) or a float -- adds
        `CO2_emission == co2_equals` (mirrors Pyomo's mutable `instance.e`/`C_e` epsilon
        constraint, Model_Resolution.py L1430-1431), pinning emissions to an exact target while
        NPC is minimized subject to it. This is how the Pareto sweep traces each point.
    lp_mode: False (default -- MILP, every certified Gili Ketapang design) or True (Stage 8,
        full-parity roadmap: the LP/continuous formulation, `Constraints_Brownfield`/
        `Constraints_Greenfield` in Constraints.py, non-`_Milp` classes). Confirmed by direct
        reading of Constraints.py L682-1353 (Brownfield LP class): RES keeps the exact same
        `RES_Units * RES_Nominal_Capacity` structure as MILP (just continuous instead of
        integer); but Battery and Generator capacity are genuinely different variables there --
        `Battery_Nominal_Capacity[ut]`/`Generator_Nominal_Capacity[ut,g]` are free continuous
        decision variables directly in kWh/kW, not `Units * fixed_nominal_size`. Every single
        place this function multiplies `Battery_Units`/`Generator_Units` by
        `inp.BATT_NOM_CAP_KWH`/`inp.GEN_NOM_CAP_KW` (investment/O&M/salvage/CO2/capacity-floor/
        no-decommission/energy-balance terms -- verified by grep, no exception exists) does so
        consistently, so setting that per-unit size to 1.0 when lp_mode is on turns
        `Battery_Units`/`Generator_Units` themselves into exactly
        `Battery_Nominal_Capacity`/`Generator_Nominal_Capacity` in kWh/kW directly -- the same
        variable-declaration code, reused, rather than a duplicated LP-specific cost/constraint
        stack. LP mode also has no partial-load generator unit-commitment machinery at all
        (Constraints.py's LP dispatch is the single `Generator_Energy_Production <=
        Generator_Nominal_Capacity*Delta_Time` cap) -- this is exactly what `plain_mode=True`
        already builds here, so lp_mode forces `plain_mode=True` internally regardless of the
        caller's own `plain_mode` argument.

        Verified 2026-09-04 on smoketest_1sc: (1) regression -- lp_mode=False (the default)
        reproduced the bit-identical established free-sizing NPC ($5,670,286.5578582985) through
        the real `linopy_free_sizing_crosscheck.py` script and its own solver recipe, confirming
        zero behavior change for every existing certified design; (2) functional -- a throwaway
        lp_mode=True free-sizing solve (same design, same solver recipe) found RES=5436.24 kW
        (continuous, no unit-count rounding), Battery_Nominal_Capacity=12020.92 kWh,
        Generator_Nominal_Capacity=472.46 kW, NPC=$5,658,231.42 -- correctly <= the MILP
        baseline (continuous capacity sizing relaxes the MILP's integer-unit-count granularity,
        so it can only find an equal-or-cheaper design), by a small and plausible ~0.21%.
        Export wiring (2026-09-04, same day): `compute_co2_emissions()` and
        `linopy_results_export.py`'s `build_adapter()` are now lp_mode-aware --
        `Battery_Units`/`Generator_Units` are treated as the capacity directly (per-unit scale
        1.0) instead of being multiplied by `inp.BATT_NOM_CAP_KWH`/`inp.GEN_NOM_CAP_KW`, `params`
        sets `MILP_Formulation=0`, and `variables` gains the LP-native attribute names
        Results.py's own `else` (non-MILP) branches read (`RES_Units`, `Battery_Nominal_Capacity`,
        `Generator_Nominal_Capacity`, `Generator_Energy_Production`) -- audited across every
        `MILP_Formulation` branch point in Results.py (EnergySystemCost, TimeSeries, SystemSize,
        SystemLand, generator-share/renewable-penetration tables). Verified end-to-end on
        `smoketest_1sc` with `LINOPY_LP_MODE=1`: full solve -> `linopy_results_export.py` ->
        real (unmodified) `Results.py` succeeded with no crash, producing a real
        `Results_Summary.xlsx`/`Time_Series_SC_1.xlsx` -- Solar PV 1793.96 kW, Battery bank
        12,020.92 kWh, Diesel Genset 472.46 kW, NPC $5,658.23k, LCOE $0.289/kWh, 87.8% average
        renewable penetration. One real bug found and fixed along the way: `_VarProxy` (in
        `pyomo_instance_adapter.py`) only implemented `.get_values()`, but Results.py's LP branch
        calls `.extract_values()` on `Battery_Nominal_Capacity`/`Generator_Nominal_Capacity` --
        real Pyomo Var objects support both identically, so `_VarProxy.extract_values()` was
        added as an alias of `.get_values()`. Regression confirmed twice more after this fix:
        the adapter smoke test (`_smoketest_results_export.py`) and a real MILP export both still
        reproduce their exact prior values ($5,670,286.5578582985 NPC) unchanged.

        **Cross-validated against real Pyomo, 2026-09-04 -- CONFIRMED, near-exact agreement.**
        Created `Code/Inputs/Parameters_1sc_10y_gili_ketapang_scpos2_LP.dat` (a permanent LP
        counterpart to the `smoketest_1sc` MILP `.dat`, `MILP_Formulation` flipped 1->0, nothing
        else changed) and ran it through the REAL, unmodified `MicroGrids.py` (Pyomo's own
        driver, `MGPY_EXPECT_MILP=0`). Pyomo's own solver log confirms this converges to an
        EXACT global optimum (`gap 0.0000%`, not a MIPGap-limited approximation) since the LP
        class is near-continuous -- the only integer artifact is the same
        Salvage_Positive_Flag floor-linearization binary this port also has, trivially branched
        in 3 nodes. Result: NPC=$5,540,329.46, Solar PV=1858.62kW, Battery=12,020.92kWh,
        Diesel Genset=470.00kW, LCOE=$0.283/kWh.

        Re-solving the SAME lp_mode=True linopy model with a tight MIPGap (1e-6, not the
        MILP-tuned 0.03 default -- see `LINOPY_LP_MODE`'s auto-tightening in
        `linopy_free_sizing_crosscheck.py`) reproduced this almost bit-for-bit: NPC=
        $5,540,329.464945197, RES=1858.6168862521763kW, Battery=12,020.9161291kWh, Generator=
        470.0kW -- agreement to 6+ significant figures, the same standard of agreement as this
        project's other tightly-converged cross-checks. **Important trap found and fixed**: the
        FIRST attempt at this comparison (using `linopy_free_sizing_crosscheck.py`'s default
        MIPGap=0.03, tuned for large real MILP designs) gave a plausible-looking but wrong
        answer ($5,658,231.42 NPC, RES=1793.96kW, Generator=472.46kW -- 2.13% off) purely from
        gap-based early termination, not a formulation bug -- `LP_MODE` now auto-tightens
        MIPGap/OptimalityTol to 1e-6 in that script so this trap does not recur.

        This closes the last open item from Stage 8 -- LP mode is now genuinely cross-validated
        against a real Pyomo RUN (not just an internal regression check or a hand-calculation),
        the first stage in this full-parity roadmap to reach that bar. Every other stage's new
        feature (CO2, land use, fuel escalation, WACC, grid connection, multi-objective,
        greenfield) still awaits its own real Pyomo cross-check on a non-smoketest design.
    """
    if bess_mode not in ("binary", "relaxed", "nobinary"):
        raise ValueError(f"bess_mode={bess_mode!r} -- must be 'binary', 'relaxed', or 'nobinary'")
    if objective not in ("npc", "co2"):
        raise ValueError(f"objective={objective!r} -- must be 'npc' or 'co2'")
    if lp_mode:
        plain_mode = True

    ut_idx, s_idx, yt_idx, t_idx = inp.ut_idx, inp.s_idx, inp.yt_idx, inp.t_idx
    ut_of, r, YEARS, STEPS_NUMBER, PERIODS = inp.ut_of, inp.r, inp.YEARS, inp.STEPS_NUMBER, inp.PERIODS
    cf, demand = inp.cf, inp.demand

    # LP mode (Stage 8): per-unit size of 1.0 collapses "Units * fixed_nominal_size" into the
    # Units variable itself being the continuous capacity (kWh/kW) directly -- see the lp_mode
    # docstring note above for why this is a faithful, verified port and not an approximation.
    batt_unit_kwh = 1.0 if lp_mode else inp.BATT_NOM_CAP_KWH
    gen_unit_kw = 1.0 if lp_mode else inp.GEN_NOM_CAP_KW

    m = linopy.Model()
    RES_Units = m.add_variables(lower=0, coords=[ut_idx], name="RES_Units", integer=not lp_mode)
    Battery_Units = m.add_variables(lower=0, coords=[ut_idx], name="Battery_Units", integer=not lp_mode)
    Generator_Units = m.add_variables(lower=0, coords=[ut_idx], name="Generator_Units", integer=not (relax_gen_units or lp_mode))
    RES_Energy_Production = m.add_variables(lower=0, coords=[s_idx, yt_idx, t_idx], name="RES_Energy_Production")
    Generator_Energy_Total = m.add_variables(lower=0, coords=[s_idx, yt_idx, t_idx], name="Generator_Energy_Total")
    Battery_Inflow = m.add_variables(lower=0, coords=[s_idx, yt_idx, t_idx], name="Battery_Inflow")
    Battery_Outflow = m.add_variables(lower=0, coords=[s_idx, yt_idx, t_idx], name="Battery_Outflow")
    Battery_SOC = m.add_variables(lower=0, coords=[s_idx, yt_idx, t_idx], name="Battery_SOC")
    Lost_Load = m.add_variables(lower=0, coords=[s_idx, yt_idx, t_idx], name="Lost_Load")
    Energy_Curtailment = m.add_variables(lower=0, coords=[s_idx, yt_idx, t_idx], name="Energy_Curtailment")
    Generator_Energy_Partial = m.add_variables(lower=0, coords=[s_idx, yt_idx, t_idx], name="Generator_Energy_Partial")
    if bess_mode == "binary":
        Single_Flow_BESS = m.add_variables(coords=[s_idx, yt_idx, t_idx], name="Single_Flow_BESS", binary=True)
    elif bess_mode == "relaxed":
        Single_Flow_BESS = m.add_variables(lower=0, upper=1, coords=[s_idx, yt_idx, t_idx], name="Single_Flow_BESS")
    if uc_integral and not plain_mode and gen_parallel:
        # Parallel load sharing (see the gen_parallel docstring note): Generator_Partial counts the
        # units running, each between its minimum and full output; Generator_Full is unused.
        Generator_Partial = m.add_variables(lower=0, coords=[s_idx, yt_idx, t_idx], name="Generator_Partial", integer=True)
        Generator_Full = m.add_variables(lower=0, coords=[s_idx, yt_idx, t_idx], name="Generator_Full", integer=True)
        m.add_constraints(Generator_Full == 0, name="Generator_Full_unused_parallel")
    elif uc_integral and not plain_mode:
        Generator_Partial = m.add_variables(coords=[s_idx, yt_idx, t_idx], name="Generator_Partial", binary=True)
        Generator_Full = m.add_variables(lower=0, coords=[s_idx, yt_idx, t_idx], name="Generator_Full", integer=True)
    else:
        Generator_Partial = m.add_variables(lower=0, upper=1, coords=[s_idx, yt_idx, t_idx], name="Generator_Partial")
        Generator_Full = m.add_variables(lower=0, coords=[s_idx, yt_idx, t_idx], name="Generator_Full")

    # Grid connection (Stage 5, full-parity roadmap) -- Model_Components==0 only (the only
    # structure this file builds); see DesignInputs' own comment for the deterministic-
    # always-available-grid simplification. Off-grid (the only case any certified Gili
    # Ketapang design uses) takes grid_import_term=grid_export_term=0, reducing Energy_balance
    # below to exactly its original off-grid form.
    Energy_From_Grid = Energy_To_Grid = None
    grid_import_term = grid_export_term = 0
    if inp.GRID_CONNECTION == 1:
        year_connected = xr.DataArray(yt_idx.values >= inp.YEAR_GRID_CONNECTION, coords={"yt": yt_idx})
        Energy_From_Grid = m.add_variables(lower=0, coords=[s_idx, yt_idx, t_idx], name="Energy_From_Grid")
        Energy_To_Grid = m.add_variables(lower=0, coords=[s_idx, yt_idx, t_idx], name="Energy_To_Grid")
        m.add_constraints(Energy_From_Grid <= inp.MAXIMUM_GRID_POWER * inp.grid_availability * year_connected, name="Maximum_Power_From_Grid")
        if inp.GRID_CONNECTION_TYPE == 0:
            m.add_constraints(Energy_To_Grid <= inp.MAXIMUM_GRID_POWER * inp.grid_availability * year_connected, name="Maximum_Power_To_Grid")
        else:
            m.add_constraints(Energy_To_Grid == 0, name="Energy_To_Grid_unused_importonly")
        if inp.GRID_CONNECTION_TYPE == 0 and bess_mode == "binary":
            Single_Flow_Grid = m.add_variables(coords=[s_idx, yt_idx, t_idx], name="Single_Flow_Grid", binary=True)
            m.add_constraints(Energy_To_Grid <= Single_Flow_Grid * inp.GRID_LARGE_CONSTANT, name="Single_Flow_Energy_To_Grid")
            # Written as Energy_From_Grid + Single_Flow_Grid*M <= M rather than the algebraically
            # identical (1-Single_Flow_Grid)*M form -- `int - Variable` (reflected subtraction)
            # isn't supported in this linopy version (same issue documented on the Salvage floor
            # constraint above); this form only ever does Variable +/- Variable.
            m.add_constraints(Energy_From_Grid + Single_Flow_Grid * inp.GRID_LARGE_CONSTANT <= inp.GRID_LARGE_CONSTANT, name="Single_Flow_Energy_From_Grid")
        elif inp.GRID_CONNECTION_TYPE == 0 and bess_mode == "relaxed":
            Single_Flow_Grid = m.add_variables(lower=0, upper=1, coords=[s_idx, yt_idx, t_idx], name="Single_Flow_Grid")
            m.add_constraints(Energy_To_Grid <= Single_Flow_Grid * inp.GRID_LARGE_CONSTANT, name="Single_Flow_Energy_To_Grid")
            # Written as Energy_From_Grid + Single_Flow_Grid*M <= M rather than the algebraically
            # identical (1-Single_Flow_Grid)*M form -- `int - Variable` (reflected subtraction)
            # isn't supported in this linopy version (same issue documented on the Salvage floor
            # constraint above); this form only ever does Variable +/- Variable.
            m.add_constraints(Energy_From_Grid + Single_Flow_Grid * inp.GRID_LARGE_CONSTANT <= inp.GRID_LARGE_CONSTANT, name="Single_Flow_Energy_From_Grid")
        # bess_mode == "nobinary" or import-only: no simultaneous-flow gate -- same caveat as the
        # battery's own nobinary mode applies if binary/relaxed isn't used here.
        grid_import_term = Energy_From_Grid * inp.grid_availability
        grid_export_term = Energy_To_Grid * inp.grid_availability if inp.GRID_CONNECTION_TYPE == 0 else 0

    m.add_constraints(RES_Energy_Production == cf * inp.RES_INVERTER_EFF * RES_Units.sel(ut=ut_of), name="Renewable_Energy")
    if not plain_mode:
        m.add_constraints(Generator_Energy_Partial >= inp.GEN_NOM_CAP_KW * inp.GEN_MIN_OUTPUT * inp.DELTA_TIME * Generator_Partial, name="Minimum_Generator_Energy_Partial")
        m.add_constraints(Generator_Energy_Partial <= inp.GEN_NOM_CAP_KW * Generator_Partial * inp.DELTA_TIME, name="Maximum_Generator_Energy_Partial")
        m.add_constraints(Generator_Energy_Total == Generator_Full * inp.GEN_NOM_CAP_KW + Generator_Energy_Partial, name="Generator_Energy_Total_def")
        m.add_constraints(Generator_Units.sel(ut=ut_of) >= Generator_Full + Generator_Partial, name="Generator_Units_Total")
    else:
        # Plain (Generator_Partial_Load=0): declared for return-dict/exporter compatibility but
        # pinned to 0, matching Pyomo's own plain branch (no constraint/objective term ever
        # references them there either).
        m.add_constraints(Generator_Partial == 0, name="Generator_Partial_unused_plain")
        m.add_constraints(Generator_Full == 0, name="Generator_Full_unused_plain")
        m.add_constraints(Generator_Energy_Partial == 0, name="Generator_Energy_Partial_unused_plain")
    m.add_constraints(Generator_Energy_Total <= gen_unit_kw * Generator_Units.sel(ut=ut_of) * inp.DELTA_TIME, name="Maximum_Generator_Energy_Total_1")
    m.add_constraints(Generator_Energy_Total <= demand * inp.DELTA_TIME, name="Maximum_Generator_Energy_Total_2")
    m.add_constraints(Generator_Units * gen_unit_kw <= inp.GEN_MAX_CAPACITY_KW, name="Generator_Max_Units")
    m.add_constraints((RES_Energy_Production + Generator_Energy_Total + grid_import_term - grid_export_term - Battery_Inflow + Battery_Outflow + Lost_Load - Energy_Curtailment) == demand, name="Energy_balance")

    batt_cap = batt_unit_kwh * Battery_Units.sel(ut=ut_of)
    batt_max_charge_power = batt_cap / inp.BATT_MIN_CHARGE_TIME
    m.add_constraints(Battery_SOC <= batt_cap, name="Maximum_Charge")
    m.add_constraints(Battery_SOC >= batt_cap * (1 - inp.BATT_DOD), name="Minimum_Charge")
    m.add_constraints(Battery_Inflow <= batt_max_charge_power * inp.DELTA_TIME, name="Max_Bat_in")
    if battery_discharge_cap:
        # Discharge C-rate cap (see the module docstring's fourth discrepancy). Independent of
        # bess_mode and lp_mode: Pyomo applies it in every formulation.
        batt_max_discharge_power = batt_cap / inp.BATT_MIN_DISCHARGE_TIME
        m.add_constraints(Battery_Outflow <= batt_max_discharge_power * inp.DELTA_TIME, name="Max_Bat_flow_out")
    m.add_constraints(Battery_Outflow <= demand, name="Max_Bat_out")
    if bess_mode in ("binary", "relaxed"):
        m.add_constraints(Battery_Outflow <= demand * Single_Flow_BESS, name="Battery_Single_Flow_Discharge")
        m.add_constraints(Battery_Inflow + inp.BATT_MAX_INFLOW_KWH * Single_Flow_BESS <= inp.BATT_MAX_INFLOW_KWH, name="Battery_Single_Flow_Charge")
    # bess_mode == "nobinary": no gate -- caller MUST run a simultaneous-flow check on the result.

    for yt in yt_idx:
        lhs = Battery_SOC.sel(yt=yt)
        charge = Battery_Inflow.sel(yt=yt) * inp.BATT_CHARGE_EFF
        discharge = Battery_Outflow.sel(yt=yt) / inp.BATT_DISCHARGE_EFF
        if yt == 1:
            rhs_t1 = batt_cap.sel(yt=yt) * inp.BATT_INITIAL_SOC - discharge.sel(t=1) + charge.sel(t=1)
        else:
            rhs_t1 = Battery_SOC.sel(yt=yt - 1, t=PERIODS) - discharge.sel(t=1) + charge.sel(t=1)
        m.add_constraints(lhs.sel(t=1) == rhs_t1, name=f"State_of_Charge_t1_yt{yt}")
        if PERIODS > 1:
            t_rest = t_idx[1:]
            rhs_rest = lhs.sel(t=t_rest.values - 1).assign_coords(t=t_rest) - discharge.sel(t=t_rest) + charge.sel(t=t_rest)
            m.add_constraints(lhs.sel(t=t_rest) == rhs_rest, name=f"State_of_Charge_rest_yt{yt}")

    m.add_constraints(RES_Units.sel(ut=1) * inp.RES_NOM_CAP_KW >= inp.RES_EXISTING_KW, name="RES_Capacity_floor")
    m.add_constraints(Battery_Units.sel(ut=1) * batt_unit_kwh >= inp.BATT_EXISTING_KWH, name="BESS_Capacity_floor")
    m.add_constraints(Generator_Units.sel(ut=1) * gen_unit_kw >= inp.GEN_EXISTING_KW, name="GEN_Capacity_floor")
    if STEPS_NUMBER > 1:
        m.add_constraints(RES_Units.sel(ut=ut_idx[1:]) >= RES_Units.sel(ut=ut_idx[:-1]).assign_coords(ut=ut_idx[1:]), name="RES_no_decommission")
        m.add_constraints(Battery_Units.sel(ut=ut_idx[1:]) >= Battery_Units.sel(ut=ut_idx[:-1]).assign_coords(ut=ut_idx[1:]), name="Battery_no_decommission")
        m.add_constraints(Generator_Units.sel(ut=ut_idx[1:]) >= Generator_Units.sel(ut=ut_idx[:-1]).assign_coords(ut=ut_idx[1:]), name="Generator_no_decommission")
    if inp.LOST_LOAD_MAX_FRACTION > 0:
        m.add_constraints(Lost_Load.sum(["yt", "t"]) <= inp.LOST_LOAD_MAX_FRACTION * demand.sum(["yt", "t"]), name="Maximum_Lost_Load")
    else:
        m.add_constraints(Lost_Load == 0, name="Maximum_Lost_Load")

    # Battery_Min_Capacity (2026-08-28 fix, see module docstring point 1) -- the Battery_Independence
    # autonomy-day floor. Mirrors Initialize.py's Initialize_Battery_Minimum_Capacity: for each
    # investment step, the scenario-weighted AVERAGE one-day (or Battery_Independence-day) energy
    # demand within that step's year window, divided by Battery_Depth_of_Discharge.
    if battery_min_capacity and inp.BATT_INDEPENDENCE_DAYS > 0:
        periods_per_day = int(round(24 / inp.DELTA_TIME))
        window_periods = int(inp.BATT_INDEPENDENCE_DAYS * periods_per_day)
        # The autonomy windows tile the WHOLE horizon, not each year separately: Initialize.py
        # builds its Grouper over Periods*Years rows before slicing by step. Tiling per year
        # gives the same answer whenever PERIODS % window_periods == 0 (the one-day case:
        # 8760/24 = 365) but is simply infeasible for two days (8760/48 = 182.5), which is what
        # the aut2 sensitivity hit on 2026-09-13. Over the horizon it divides: 175200/48 = 3650.
        total_periods = PERIODS * YEARS
        if total_periods < window_periods:
            raise ValueError(f"Battery_Independence window ({window_periods} periods) is longer than the horizon ({total_periods} periods)")
        # Rows past the last whole window get no Grouper in Initialize.py and drop out of the
        # mean; reproduce that rather than averaging over a short final window.
        n_windows_total = total_periods // window_periods
        grouped_periods = n_windows_total * window_periods
        batt_min_cap_vals = []
        for ut in ut_idx:
            yts_in_step = [yt for yt in yt_idx if int(ut_of.sel(yt=yt)) == ut]
            pos = np.concatenate([(yt - 1) * PERIODS + np.arange(PERIODS) for yt in yts_in_step])
            step = demand.sel(yt=yts_in_step).stack(ht=("yt", "t"))
            step = step.reset_index("ht", drop=True).assign_coords(ht=np.arange(pos.size))
            keep = np.flatnonzero(pos < grouped_periods)
            step, pos_kept = step.isel(ht=keep), pos[keep]
            window_idx = xr.DataArray(pos_kept // window_periods, coords={"ht": step.ht}, name="window")
            # A step boundary can fall inside a window (5-year steps, 2-day windows: 43800/48 =
            # 912.5), leaving a part-window at each end of the slice -- Initialize.py's .loc
            # slicing keeps those partial groups too, so the groupby below matches it.
            avg_per_s = step.groupby(window_idx).sum().mean(dim="window")
            batt_min_cap_vals.append(float((avg_per_s * inp.scenario_weight).sum()) / inp.BATT_DOD)
        batt_min_cap = xr.DataArray(batt_min_cap_vals, coords={"ut": ut_idx})
        m.add_constraints(Battery_Units * batt_unit_kwh >= batt_min_cap, name="Battery_Min_Capacity")

    # Renewables_Max_Land_Use (Stage 2, full-parity roadmap) -- mirrors
    # Constraints_Brownfield_Milp's own rule (Constraints.py ~L2603-2604): total RES footprint
    # (pre-existing + new units, this project's single renewable_sources index collapsed to a
    # scalar like every other RES_* parameter here) must not exceed Renewables_Total_Area, one
    # constraint per investment step. Pyomo only builds this when Land_Use==1
    # (Model_Resolution.py L687-689); every design solved so far has Land_Use=0, so this has
    # been a no-op in every certified result to date -- it only starts binding once a site's
    # .dat actually turns the flag on.
    if inp.LAND_USE == 1:
        land_footprint = inp.RES_EXISTING_AREA + RES_Units * inp.RES_NOM_CAP_KW * inp.RES_SPECIFIC_AREA
        m.add_constraints(land_footprint <= inp.RENEWABLES_TOTAL_AREA, name="Renewables_Max_Land_Use")

    # A partial pin (only some keys, e.g. {"Generator": ...} from free sizing's
    # LINOPY_FIX_GENERATOR_UNITS, 2026-09-11) fixes just those units and leaves the rest free.
    if pinned is not None:
        if "RES" in pinned:
            m.add_constraints(RES_Units == pinned["RES"], name="Fix_RES_Units")
        if "Battery" in pinned:
            m.add_constraints(Battery_Units == pinned["Battery"], name="Fix_Battery_Units")
        if "Generator" in pinned:
            m.add_constraints(Generator_Units == pinned["Generator"], name="Fix_Generator_Units")

    inv_res = RES_Units.sel(ut=1) * inp.RES_NOM_CAP_KW * inp.RES_SPECIFIC_COST
    inv_gen = Generator_Units.sel(ut=1) * gen_unit_kw * inp.GEN_SPECIFIC_COST
    inv_bat = Battery_Units.sel(ut=1) * batt_unit_kwh * inp.BATT_SPECIFIC_COST
    investment_cost = inv_res + inv_gen + inv_bat
    investment_const_offset = -(inp.RES_EXISTING_KW * inp.RES_SPECIFIC_COST + inp.GEN_EXISTING_KW * inp.GEN_SPECIFIC_COST + inp.BATT_EXISTING_KWH * inp.BATT_SPECIFIC_COST)
    for ut in ut_idx[1:]:
        yt0 = inp.step_start_year[ut]
        disc = (1 + r) ** (yt0 - 1)
        investment_cost = investment_cost + ((RES_Units.sel(ut=ut) - RES_Units.sel(ut=ut - 1)) * inp.RES_NOM_CAP_KW * inp.RES_SPECIFIC_COST + (Generator_Units.sel(ut=ut) - Generator_Units.sel(ut=ut - 1)) * gen_unit_kw * inp.GEN_SPECIFIC_COST + (Battery_Units.sel(ut=ut) - Battery_Units.sel(ut=ut - 1)) * batt_unit_kwh * inp.BATT_SPECIFIC_COST) / disc

    om_res = om_gen = om_bat = 0
    for yt in yt_idx:
        disc = (1 + r) ** yt
        ut = int(ut_of.sel(yt=yt))
        om_res = om_res + RES_Units.sel(ut=ut) * inp.RES_NOM_CAP_KW * inp.RES_SPECIFIC_COST * inp.RES_OM_FRAC / disc
        om_gen = om_gen + Generator_Units.sel(ut=ut) * gen_unit_kw * inp.GEN_SPECIFIC_COST * inp.GEN_OM_FRAC / disc
        om_bat = om_bat + Battery_Units.sel(ut=ut) * batt_unit_kwh * inp.BATT_SPECIFIC_COST * inp.BATT_OM_FRAC / disc
    om_cost = om_res + om_gen + om_bat

    # Grid connection cost/O&M (Stage 5, full-parity roadmap) -- mirrors Constraints.py's
    # Investment_Cost/Operation_Maintenance_Cost_Act Inv_Grid/OyM_Grid accumulators exactly
    # (incurred every year from Year_Grid_Connection onward, discounted). Zero when
    # GRID_CONNECTION=0 (every certified design so far). Kept OUT of investment_cost/om_cost
    # themselves (which feed m.add_objective() below) because they are pure Python constants
    # with no decision variable in them -- linopy's objective rejects a bare constant term
    # ("Constant values in objective function not supported", hit and fixed 2026-09-04); folded
    # into grid_const_offset and added back into total_const_offset after solving instead, same
    # mechanism investment_const_offset already uses for the RES/GEN/BATT existing-capacity
    # credit.
    grid_const_offset = 0
    if inp.GRID_CONNECTION == 1:
        for yt in yt_idx:
            yt_val = int(yt)
            if yt_val >= inp.YEAR_GRID_CONNECTION:
                grid_const_offset += (inp.GRID_CONNECTION_COST * inp.GRID_DISTANCE) / (1 + r) ** (yt_val - 1)
                grid_const_offset += (inp.GRID_CONNECTION_COST * inp.GRID_DISTANCE * inp.GRID_MAINTENANCE_COST) / (1 + r) ** yt_val

    disc_yt = (1 + r) ** xr.DataArray(yt_idx.values, coords={"yt": yt_idx})
    # Fuel cost (Stage 3, full-parity roadmap): Fuel_Specific_Cost_Calculation==1 uses the
    # per-year marginal-cost/start-cost DataArrays DesignInputs already computed (escalation-rate
    # sub-mode only -- see that class's docstring note); ==0 keeps the original flat-price scalars
    # unchanged, so every existing certified design (all of which use ==0) is byte-for-byte
    # identical to before this stage.
    if inp.FUEL_SPECIFIC_COST_CALCULATION == 1:
        gen_marginal_cost, gen_marginal_cost_milp_1, gen_start_cost_1 = (
            inp.gen_marginal_cost_yearly, inp.gen_marginal_cost_milp_yearly, inp.gen_start_cost_yearly)
    else:
        gen_marginal_cost, gen_marginal_cost_milp_1, gen_start_cost_1 = (
            inp.GEN_MARGINAL_COST, inp.GEN_MARGINAL_COST_MILP_1, inp.GEN_START_COST_1)
    if not plain_mode:
        fuel_cost = (((Generator_Full * inp.GEN_NOM_CAP_KW * gen_marginal_cost + Generator_Energy_Partial * gen_marginal_cost_milp_1 + Generator_Partial * gen_start_cost_1) / disc_yt).sum(["yt", "t"]) * inp.scenario_weight).sum("s")
    else:
        fuel_cost = ((Generator_Energy_Total * gen_marginal_cost / disc_yt).sum(["yt", "t"]) * inp.scenario_weight).sum("s")
    lost_load_cost = ((Lost_Load * inp.LOST_LOAD_COST / disc_yt).sum(["yt", "t"]) * inp.scenario_weight).sum("s")
    batt_repl_cost = (((Battery_Inflow + Battery_Outflow) * inp.BATT_REPL_UNIT_COST / disc_yt).sum(["yt", "t"]) * inp.scenario_weight).sum("s")
    # Variable O&M (see DesignInputs): only built when a .dat sets it, so the objective of every
    # earlier design is untouched.
    var_om_cost = 0
    if inp.GEN_VAR_OM > 0 or inp.BATT_VAR_OM > 0:
        var_om_cost = (((Generator_Energy_Total * inp.GEN_VAR_OM + Battery_Outflow * inp.BATT_VAR_OM) / disc_yt).sum(["yt", "t"]) * inp.scenario_weight).sum("s")

    # Grid electricity purchase cost / export revenue (Stage 5, full-parity roadmap) -- mirrors
    # Total_Electricity_Cost_Act / Total_Revenues_Act exactly. Zero when GRID_CONNECTION=0.
    grid_electricity_cost = 0
    grid_electricity_revenue = 0
    if inp.GRID_CONNECTION == 1:
        grid_electricity_cost = ((Energy_From_Grid * inp.grid_availability * inp.GRID_PURCHASED_EL_PRICE / disc_yt).sum(["yt", "t"]) * inp.scenario_weight).sum("s")
        if inp.GRID_CONNECTION_TYPE == 0:
            grid_electricity_revenue = ((Energy_To_Grid * inp.grid_availability * inp.GRID_SOLD_EL_PRICE / disc_yt).sum(["yt", "t"]) * inp.scenario_weight).sum("s")

    def salvage_term(units_var, existing_kw, existing_years, nom_cap, specific_cost, lifetime):
        step1_factor = specific_cost * (lifetime - YEARS) / lifetime / (1 + r) ** YEARS
        sv = units_var.sel(ut=1) * nom_cap * step1_factor
        const = -existing_kw * step1_factor
        const = const + existing_kw * specific_cost * (lifetime - existing_years - YEARS) / lifetime / (1 + r) ** YEARS
        for ut in ut_idx[1:]:
            yt0 = inp.step_start_year[ut]
            increment = (units_var.sel(ut=ut) - units_var.sel(ut=ut - 1)) * nom_cap
            sv = sv + increment * specific_cost * (lifetime + (yt0 - 1) - YEARS) / lifetime / (1 + r) ** YEARS
        return sv, const

    sv_res, const_res = salvage_term(RES_Units, inp.RES_EXISTING_KW, inp.RES_EXISTING_YEARS, inp.RES_NOM_CAP_KW, inp.RES_SPECIFIC_COST, inp.RES_LIFETIME)
    sv_gen, const_gen = salvage_term(Generator_Units, inp.GEN_EXISTING_KW, inp.GEN_EXISTING_YEARS, gen_unit_kw, inp.GEN_SPECIFIC_COST, inp.GEN_LIFETIME)
    raw_salvage = sv_res + sv_gen + (const_res + const_gen)
    if inp.GRID_CONNECTION == 1:
        # SV_Grid (Stage 5, full-parity roadmap) -- mirrors Constraints.py's Salvage_Value
        # SV_Grid term exactly: discounted residual value of the grid connection investment.
        raw_salvage = raw_salvage + (inp.GRID_DISTANCE * inp.GRID_CONNECTION_COST) / (1 + r) ** (YEARS - inp.YEAR_GRID_CONNECTION)

    # Salvage_Value floor-at-zero (2026-08-28 fix, see module docstring point 2). Pyomo's own
    # model declares Salvage_Value as NonNegativeReals pinned by hard equality to this same
    # vintage-by-vintage formula -- a design that front-loads short-lifetime vintages heavily
    # enough can drive the raw total negative, which is infeasible against that >=0 domain in
    # Pyomo but was previously silently ALLOWED here. No-op for every design solved so far at
    # 9sc scale (raw total stays non-negative); only matters once a design pushes it negative.
    if salvage_floor:
        salvage_positive_flag = m.add_variables(binary=True, name="Salvage_Positive_Flag")
        salvage_value = m.add_variables(lower=0, name="Salvage_Value_Floored")
        m.add_constraints(salvage_value <= raw_salvage + big_m_salvage - big_m_salvage * salvage_positive_flag, name="Salvage_Floor_Upper_Raw")
        m.add_constraints(salvage_value <= big_m_salvage * salvage_positive_flag, name="Salvage_Floor_Upper_Flag")
        total_const_offset = investment_const_offset + grid_const_offset
    else:
        salvage_value = m.add_variables(lower=0, name="Salvage_Value_Floored")
        m.add_constraints(salvage_value == raw_salvage, name="Salvage_Value_Historical")
        total_const_offset = investment_const_offset + grid_const_offset

    npc = investment_cost + om_cost + fuel_cost + lost_load_cost + batt_repl_cost + grid_electricity_cost - grid_electricity_revenue - salvage_value
    if inp.GEN_VAR_OM > 0 or inp.BATT_VAR_OM > 0:
        npc = npc + var_om_cost

    npc_var = co2_var = None
    if objective == "npc" and co2_equals is None:
        # Every stage before Stage 6, and every default call today, take this exact path --
        # `m.add_objective(npc)` directly, no extra variable/constraint. Kept byte-for-byte
        # unchanged from before Stage 6 existed: introducing the NPC_f1/CO2_emission_f2
        # auxiliary variables unconditionally (even when unused) was tried and reverted
        # (2026-09-04) -- it is mathematically inert but changes the MILP's variable/constraint
        # count, which perturbed Gurobi's B&B path enough to land on a different (still within
        # MIPGap, but NOT bit-identical) solution for smoketest_1sc. That is a real regression
        # against this project's own bit-identical-reproduction standard, so the two new
        # variables are now built ONLY when Stage 6 actually needs them (below).
        m.add_objective(npc)
    else:
        # CO2 emission as a LIVE linopy expression (Stage 6, full-parity roadmap) -- mirrors
        # compute_co2_emissions()'s math exactly but built from the model's own decision
        # variables (not solved DataArrays), so it can be constrained (epsilon-constraint
        # Pareto sweep, co2_equals) or set as the objective directly. Off-grid RES/GEN/BESS/FUEL
        # terms always built; GRID term only when GRID_CONNECTION==1. f1/f2 mirror Pyomo's own
        # auxiliary Var pattern (Model_Resolution.py L1360-1367: model.f1==NPC,
        # model.f2==CO2_emission) so either can be read post-solve via `.solution` regardless of
        # which one is actually the objective.
        res_co2 = inp.RES_UNIT_CO2 * (RES_Units.sel(ut=1) * inp.RES_NOM_CAP_KW - inp.RES_EXISTING_KW)
        gen_co2 = inp.GEN_UNIT_CO2 * (Generator_Units.sel(ut=1) * gen_unit_kw - inp.GEN_EXISTING_KW)
        bess_co2 = inp.BESS_UNIT_CO2 * (Battery_Units.sel(ut=1) * batt_unit_kwh - inp.BATT_EXISTING_KWH)
        for ut in ut_idx[1:]:
            res_co2 = res_co2 + inp.RES_UNIT_CO2 * (RES_Units.sel(ut=ut) - RES_Units.sel(ut=ut - 1)) * inp.RES_NOM_CAP_KW
            gen_co2 = gen_co2 + inp.GEN_UNIT_CO2 * (Generator_Units.sel(ut=ut) - Generator_Units.sel(ut=ut - 1)) * gen_unit_kw
            bess_co2 = bess_co2 + inp.BESS_UNIT_CO2 * (Battery_Units.sel(ut=ut) - Battery_Units.sel(ut=ut - 1)) * batt_unit_kwh
        fuel_co2_factor = inp.FUEL_UNIT_CO2 / (inp.FUEL_LHV * inp.GEN_EFFICIENCY)
        scenario_co2 = (Generator_Energy_Total * fuel_co2_factor).sum(["yt", "t"])
        if inp.GRID_CONNECTION == 1:
            scenario_co2 = scenario_co2 + (Energy_From_Grid * inp.grid_availability * inp.NATIONAL_GRID_CO2).sum(["yt", "t"])
        co2_emission_expr = res_co2 + gen_co2 + bess_co2 + (scenario_co2 * inp.scenario_weight).sum("s")

        npc_var = m.add_variables(name="NPC_f1")
        co2_var = m.add_variables(lower=0, name="CO2_emission_f2")
        m.add_constraints(npc_var == npc, name="NPC_def")
        m.add_constraints(co2_var == co2_emission_expr, name="CO2_emission_def")
        if co2_equals is not None:
            m.add_constraints(co2_var == co2_equals, name="CO2_fixed")
        # `1 *` coerces the bare Variable into a LinearExpression -- add_objective() rejects a
        # raw Variable outright.
        m.add_objective(1 * npc_var if objective == "npc" else 1 * co2_var)

    return dict(
        m=m, total_const_offset=total_const_offset, demand=demand,
        plain_mode=plain_mode, lp_mode=lp_mode,
        RES_Units=RES_Units, Battery_Units=Battery_Units, Generator_Units=Generator_Units,
        RES_Energy_Production=RES_Energy_Production, Generator_Energy_Total=Generator_Energy_Total,
        Generator_Energy_Partial=Generator_Energy_Partial, Generator_Partial=Generator_Partial,
        Generator_Full=Generator_Full, Battery_Inflow=Battery_Inflow, Battery_Outflow=Battery_Outflow,
        Battery_SOC=Battery_SOC, Energy_Curtailment=Energy_Curtailment, Lost_Load=Lost_Load,
        Energy_From_Grid=Energy_From_Grid, Energy_To_Grid=Energy_To_Grid,
        NPC_f1=npc_var, CO2_emission_f2=co2_var,
    )


def compute_co2_emissions(inp: DesignInputs, res_units_da, battery_units_da, generator_units_da,
                           generator_energy_total_da, energy_from_grid_da=None, lp_mode=False):
    """Off-grid CO2 emissions accounting (Stage 1), mirroring Constraints_Brownfield_Milp's
    Model_Components==0 branch (Constraints.py ~L2735-2790: RES_emission, GEN_emission,
    BESS_emission net of pre-existing brownfield capacity credit, plus per-scenario fuel-burn
    emissions). GRID_emission (Stage 5, full-parity roadmap) is now also included when
    energy_from_grid_da is provided (pass build_core_model()'s Energy_From_Grid.solution for a
    grid-connected design; leave None for off-grid, the only case any certified Gili Ketapang
    design uses) -- mirrors GRID_emission[s,y,t] == Energy_From_Grid[s,y,t] *
    National_Grid_Specific_CO2_emissions exactly.

    Computed post-hoc from the SOLVED sizing/dispatch arrays (same pattern
    linopy_results_export.py already uses for investment_cost/om_cost/salvage_value), not as
    linopy decision variables -- CO2 does not enter any constraint or the (still single-)
    objective yet, so there is nothing for the solver to determine here.

    Returns a dict with the same names/shapes as the matching Pyomo variables in
    Constraints_Brownfield_Milp, so linopy_results_export.py can map them 1:1 into
    Results.py's variables dict instead of zero-filling: CO2_emission (scalar),
    Scenario_CO2_emission ({s: value}), RES_emission/GEN_emission/BESS_emission (scalars,
    LCA/investment emissions -- same for every scenario, no `s` index in Pyomo either),
    Scenario_FUEL_emission ({s: value}), FUEL_emission (an (s,yt,t) DataArray, operational
    fuel-burn emissions). Units: kg CO2-equivalent, matching the RES_unit_CO2_emission/
    BESS_unit_CO2_emission/GEN_unit_CO2_emission/FUEL_unit_CO2_emission parameters in
    Code/Inputs/*.dat.

    lp_mode: False (default -- MILP, every certified Gili Ketapang design) or True (Stage 8's
        LP/continuous formulation). Mirrors build_core_model()'s own lp_mode trick exactly: in
        LP mode, battery_units_da/generator_units_da ARE Battery_Nominal_Capacity/
        Generator_Nominal_Capacity directly (kWh/kW), not a unit COUNT to be multiplied by a
        fixed per-unit nominal size -- so the per-unit scale used here must be 1.0 for those two,
        same as build_core_model()'s batt_unit_kwh/gen_unit_kw. RES is unaffected (RES_Units
        always needs multiplying by RES_Nominal_Capacity in both MILP and LP).
    """
    ut_idx = inp.ut_idx
    batt_unit_kwh = 1.0 if lp_mode else inp.BATT_NOM_CAP_KWH
    gen_unit_kw = 1.0 if lp_mode else inp.GEN_NOM_CAP_KW

    res_emission = inp.RES_UNIT_CO2 * (float(res_units_da.sel(ut=1)) * inp.RES_NOM_CAP_KW - inp.RES_EXISTING_KW)
    for ut in ut_idx[1:]:
        increment_kw = float(res_units_da.sel(ut=ut) - res_units_da.sel(ut=ut - 1)) * inp.RES_NOM_CAP_KW
        res_emission += inp.RES_UNIT_CO2 * increment_kw

    gen_emission = inp.GEN_UNIT_CO2 * (float(generator_units_da.sel(ut=1)) * gen_unit_kw - inp.GEN_EXISTING_KW)
    for ut in ut_idx[1:]:
        increment_kw = float(generator_units_da.sel(ut=ut) - generator_units_da.sel(ut=ut - 1)) * gen_unit_kw
        gen_emission += inp.GEN_UNIT_CO2 * increment_kw

    bess_emission = inp.BESS_UNIT_CO2 * (float(battery_units_da.sel(ut=1)) * batt_unit_kwh - inp.BATT_EXISTING_KWH)
    for ut in ut_idx[1:]:
        increment_kwh = float(battery_units_da.sel(ut=ut) - battery_units_da.sel(ut=ut - 1)) * batt_unit_kwh
        bess_emission += inp.BESS_UNIT_CO2 * increment_kwh

    # Fuel-burn emissions: energy / (LHV x generator efficiency) x fuel emission factor, summed
    # over years/periods per scenario -- same formula regardless of plain_mode/partial-load,
    # since Constraints.py's CO2 block reads Generator_Energy_Total directly, not the
    # partial/full split (that split only matters for the fuel-COST objective term, not CO2).
    fuel_factor = inp.FUEL_UNIT_CO2 / (inp.FUEL_LHV * inp.GEN_EFFICIENCY)
    fuel_emission_da = generator_energy_total_da * fuel_factor  # (s, yt, t) -- matches Pyomo's
    # per-(s,yt,g,t) FUEL_emission with the single generator type's `g` dimension dropped (this
    # project always has exactly one generator type, same simplification already used
    # throughout this file for RES/Generator/Battery unit costs).
    scenario_fuel_emission_da = fuel_emission_da.sum(["yt", "t"])

    scenario_fuel_emission = {int(s): float(scenario_fuel_emission_da.sel(s=s)) for s in inp.s_idx}

    # GRID_emission (Stage 5, full-parity roadmap) -- import energy x national grid CO2 factor.
    # Only meaningful when GRID_CONNECTION==1 and a solved Energy_From_Grid is supplied.
    scenario_grid_emission = {int(s): 0.0 for s in inp.s_idx}
    if energy_from_grid_da is not None:
        grid_emission_da = energy_from_grid_da * inp.grid_availability * inp.NATIONAL_GRID_CO2
        scenario_grid_emission_da = grid_emission_da.sum(["yt", "t"])
        scenario_grid_emission = {int(s): float(scenario_grid_emission_da.sel(s=s)) for s in inp.s_idx}

    scenario_co2 = {s: res_emission + gen_emission + bess_emission + scenario_fuel_emission[s] + scenario_grid_emission[s]
                     for s in scenario_fuel_emission}
    co2_emission_total = sum(scenario_co2[s] * float(inp.scenario_weight.sel(s=s)) for s in scenario_co2)
    return dict(
        CO2_emission=co2_emission_total, Scenario_CO2_emission=scenario_co2,
        RES_emission=res_emission, GEN_emission=gen_emission, BESS_emission=bess_emission,
        Scenario_FUEL_emission=scenario_fuel_emission, FUEL_emission=fuel_emission_da,
        Scenario_GRID_emission=scenario_grid_emission,
    )


def run_pareto_sweep(inp: DesignInputs, pinned=None, bess_mode="binary", plain_mode=False,
                      relax_gen_units=False, solver_kwargs=None, n_points=None, include_max_cost=None,
                      progress_callback=None, solver_name="gurobi"):
    """Epsilon-constraint NPC-vs-CO2 Pareto sweep (Stage 6, full-parity roadmap), mirroring
    Model_Resolution.py's Multiobjective_Optimization==1 branch (L1358-1499) phase for phase:

      Phase 1: solve for minimum NPC alone (the normal, single-objective path every other stage
               uses) -> NPC_min. CO2 at that design (CO2emission_max, the Pareto front's
               highest-emission point) is read via compute_co2_emissions() on the solved sizing
               -- no need for the live CO2 variable for this phase.
      Phase 2: solve for minimum CO2 alone (objective="co2") -> CO2emission_min, the lowest
               emissions this design space can achieve at all.
      Phase 3: solve for minimum NPC s.t. CO2 == CO2emission_min (co2_equals) -> NPC_max, the
               cost of the cleanest possible design. Matches Pyomo's own explicit third solve
               even though the main sweep below would reach the same point at its top step --
               kept for exact structural parity with the source algorithm.
      Phase 4: sweep n_points (inp.PARETO_POINTS if not overridden) epsilon values between
               CO2emission_min and CO2emission_max, solving minimum NPC s.t. CO2 == epsilon at
               each one, collecting the (NPC, CO2) Pareto front.

    This is inherently n+3 full MILP solves -- expensive by construction, matching Pyomo's own
    algorithm exactly (it is not a shortcut this port could take that Pyomo doesn't already
    require). Each phase calls build_core_model() completely fresh (matching Pyomo's own
    `model.create_instance(datapath)` per phase) rather than mutating one linopy Model in place.

    progress_callback(phase_name, result_dict) is called after each solve if provided, so a
    caller can log CPU/RAM/wall-clock per phase (this session's standing instrumentation
    requirement) without this function hardcoding any particular logging mechanism.

    solver_name: "gurobi" (default, matches every other driver in this project) or any other
        backend linopy itself supports (e.g. "highs", "cbc", "glpk", "cplex"). solver_kwargs is
        passed straight through to that backend's own Python API (2026-09-05, LINOPY_SOLVER
        mechanism) -- it is NOT abstracted across solvers, so a Gurobi-tuned solver_kwargs dict
        (MIPFocus, Crossover, BarHomogeneous, ...) is only valid when solver_name="gurobi"; the
        caller is responsible for passing an empty or backend-appropriate dict otherwise.

    Returns dict(NPC_min, CO2_max, NPC_max, CO2_min, pareto_npc=[...], pareto_co2=[...]).
    """
    solver_kwargs = dict(solver_kwargs or {})
    n_points = n_points if n_points is not None else inp.PARETO_POINTS
    include_max_cost = include_max_cost if include_max_cost is not None else bool(inp.PLOT_MAX_COST)

    def _solve(objective, co2_equals=None, phase_name=""):
        built = build_core_model(inp, pinned=pinned, bess_mode=bess_mode, plain_mode=plain_mode,
                                  relax_gen_units=relax_gen_units, objective=objective, co2_equals=co2_equals)
        status, condition = built["m"].solve(solver_name=solver_name, **solver_kwargs)
        if status != "ok":
            raise RuntimeError(f"Pareto sweep phase {phase_name!r} did not solve (status={status}, condition={condition})")
        if progress_callback is not None:
            progress_callback(phase_name, built)
        return built

    # Phase 1: minimum NPC alone -- the normal path, no aux variables built at all.
    built1 = _solve(objective="npc", phase_name="min_npc")
    npc_min = built1["m"].objective.value + built1["total_const_offset"]
    co2_at_min_npc = compute_co2_emissions(
        inp, built1["RES_Units"].solution, built1["Battery_Units"].solution, built1["Generator_Units"].solution,
        built1["Generator_Energy_Total"].solution,
        energy_from_grid_da=built1["Energy_From_Grid"].solution if built1["Energy_From_Grid"] is not None else None,
    )["CO2_emission"]

    # Phase 2: minimum CO2 alone.
    built2 = _solve(objective="co2", phase_name="min_co2")
    co2_min = float(built2["CO2_emission_f2"].solution)
    npc_at_min_co2 = float(built2["NPC_f1"].solution) + built2["total_const_offset"]

    # Phase 3: minimum NPC s.t. CO2 == co2_min (the cleanest-design cost point).
    built3 = _solve(objective="npc", co2_equals=co2_min, phase_name="npc_at_min_co2")
    npc_max = float(built3["NPC_f1"].solution) + built3["total_const_offset"]

    # Phase 4: sweep.
    co2_max = co2_at_min_npc
    if co2_max <= co2_min:
        raise ValueError(f"CO2emission_max ({co2_max}) <= CO2emission_min ({co2_min}) -- no real "
                          f"range to sweep; check the design actually has room to trade cost for emissions.")
    step = (co2_max - co2_min) / (n_points - 1 if include_max_cost else n_points)
    steps = list(np.arange(co2_min, co2_max, step))
    if not include_max_cost:
        steps = steps[1:] if steps else steps

    pareto_npc, pareto_co2 = [], []
    for i, eps in enumerate(steps):
        built_i = _solve(objective="npc", co2_equals=float(eps), phase_name=f"sweep_{i}")
        npc_i = float(built_i["NPC_f1"].solution) + built_i["total_const_offset"]
        co2_i = float(built_i["CO2_emission_f2"].solution)
        pareto_npc.append(npc_i)
        pareto_co2.append(co2_i)

    if len(pareto_npc) < n_points:  # ensure the min-cost/max-emission endpoint is included
        pareto_npc.append(npc_min)
        pareto_co2.append(co2_max)

    return dict(
        NPC_min=npc_min, CO2_max=co2_max, NPC_max=npc_max, CO2_min=co2_min,
        pareto_npc=pareto_npc, pareto_co2=pareto_co2,
    )


def resolve_plain_mode(inp, env=None):
    """plain_mode for this design, taken from the .dat's own Generator_Partial_Load flag
    (0 -> plain dispatch, 1 -> partial-load unit commitment), matching what Pyomo builds.

    LINOPY_PLAIN is still honoured for backward compatibility, but only if it agrees with the
    .dat -- a mismatch raises instead of silently building a different model than Pyomo would."""
    env = os.environ if env is None else env
    from_dat = inp.GENERATOR_PARTIAL_LOAD == 0
    raw = env.get("LINOPY_PLAIN", "").strip()
    if raw == "":
        return from_dat
    requested = raw == "1"
    if requested != from_dat:
        raise ValueError(
            f"LINOPY_PLAIN={raw} asks for the {'plain' if requested else 'partial-load'} "
            f"formulation, but {os.path.basename(inp.params_dat)} sets Generator_Partial_Load="
            f"{inp.GENERATOR_PARTIAL_LOAD}. Unset LINOPY_PLAIN (the .dat now decides), or fix the "
            f".dat.")
    return requested


def check_simultaneous_flows(battery_inflow_da, battery_outflow_da, threshold_kwh=0.001):
    """Mirrors Model_Resolution.py's Check_Simultaneous_Flows -- mandatory for bess_mode='nobinary',
    and reused by fixed-design verification's own battery-overlap check. Returns
    (worst_overlap_kwh, n_offenders, total_periods) -- a superset of what each original call site
    used (free-sizing only printed; fixed-design captured all three)."""
    overlap = xr.apply_ufunc(np.minimum, battery_inflow_da, battery_outflow_da)
    offenders = overlap > threshold_kwh
    n_offenders = int(offenders.sum())
    total_periods = int(offenders.size)
    worst = float(overlap.max())
    print(f"Simultaneous charge/discharge check over {total_periods} periods:")
    print(f"  worst overlap      = {worst:.4f} kWh")
    print(f"  periods above {threshold_kwh} kWh = {n_offenders}")
    if n_offenders == 0:
        print("VERDICT: CERTIFIED FEASIBLE (no simultaneous flows found).")
    else:
        print("VERDICT: INVALID -- simultaneous charge/discharge found. This result is only a "
              "LOWER BOUND on the true (single-flow-constrained) cost, not a valid answer.")
    return worst, n_offenders, total_periods


def recover_memlimit_incumbent(m):
    """Gurobi status 17 (MEM_LIMIT, raised by SoftMemLimit) keeps the incumbent, but linopy maps it
    to "internal_solver_error" and loads no solution (2026-09-18: the 12sc _aut0 parallel
    certification on HPC lost a 0.35%-gap incumbent this way). Loads the incumbent into the linopy
    variables exactly as linopy's own Model.solve() does and returns True, or returns False if there
    is nothing to recover (no solution yet, e.g. a stop in presolve). A hard MemLimit raises "Out of
    memory" inside Gurobi instead and cannot be recovered, which is why LINOPY_SOFTMEMLIMIT should
    be set below --memlimit. Shared by the free-sizing and fixed-design scripts."""
    from linopy.common import set_int_index
    gm = getattr(m, "solver_model", None)
    try:
        if gm is None or int(gm.Status) != 17 or int(gm.SolCount) == 0:
            return False
        sol = pd.Series({v.VarName: v.X for v in gm.getVars()}, dtype=float)
        objective = float(gm.ObjVal)
    except Exception:  # noqa: BLE001 -- nothing recoverable; the caller reports the error as before
        return False
    sol = set_int_index(sol)
    sol.loc[-1] = np.nan
    for _, var in m.variables.items():
        idx = np.ravel(var.labels)
        var.solution = xr.DataArray(sol.reindex(idx).values.reshape(var.labels.shape), var.coords)
    m.objective._value = objective
    # linopy's .solution accessors refuse to read unless the model status is "ok".
    m.status = "ok"
    m.termination_condition = "terminated_by_limit"
    return True


def check_fractional_commitment(partial_sol, full_sol):
    """Mirrors Model_Resolution.py's Check_Generator_Commitment_Integrality."""
    def worst_frac(x):
        return float(np.abs(x.values - np.round(x.values)).max())
    return worst_frac(partial_sol), worst_frac(full_sol)


# ---- Resource monitor (moved here from linopy_free_sizing_crosscheck.py so both scripts share
# it -- see this module's docstring, "THIRD DISCREPANCY"). Polls RSS + CPU% in a background
# thread every RESOURCE_POLL_SECONDS so a long solve's log carries a memory/CPU trace without
# needing an external `ps` check. ----
RESOURCE_POLL_SECONDS = 60
_resource_monitor_stop = None
_resource_monitor_thread = None


def start_resource_monitor():
    global _resource_monitor_stop, _resource_monitor_thread
    if _resource_monitor_stop is not None:
        print("A resource monitor is already running -- ignoring duplicate start.")
        return
    proc = psutil.Process(os.getpid())
    proc.cpu_percent(interval=None)  # first call always returns 0.0/garbage -- prime it, discard
    stop_event = threading.Event()

    def _loop():
        t0 = time.time()
        peak_rss_mb = 0.0
        while not stop_event.is_set():
            rss_mb = proc.memory_info().rss / (1024**2)
            cpu_pct = proc.cpu_percent(interval=None)
            peak_rss_mb = max(peak_rss_mb, rss_mb)
            print(f"[resource monitor] t={time.time()-t0:.0f}s  RSS={rss_mb:.0f} MB "
                  f"(peak so far {peak_rss_mb:.0f} MB)  CPU={cpu_pct:.1f}%", flush=True)
            stop_event.wait(RESOURCE_POLL_SECONDS)

    thread = threading.Thread(target=_loop, daemon=True)
    _resource_monitor_stop = stop_event
    _resource_monitor_thread = thread
    thread.start()
    print(f"Resource monitor started (polling every {RESOURCE_POLL_SECONDS}s)")


def stop_resource_monitor():
    # Joining here -- without it this daemon thread could still be mid-psutil-call when the
    # interpreter starts finalizing at normal script exit, and a write to stdout from a thread
    # racing shutdown is a likely cause of spurious "Error in sys.excepthook" tracebacks after
    # "Done" (harmless -- output already written by then -- but cheap to close off).
    global _resource_monitor_stop, _resource_monitor_thread
    if _resource_monitor_stop is None:
        return
    _resource_monitor_stop.set()
    if _resource_monitor_thread is not None:
        _resource_monitor_thread.join(timeout=5)
    _resource_monitor_stop = None
    _resource_monitor_thread = None
    print("Resource monitor stopped.")
