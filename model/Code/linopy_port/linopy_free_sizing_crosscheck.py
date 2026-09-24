"""GENERALIZED real-scale FREE-SIZING solve -- the sizing-decision counterpart to
linopy_fixed_design_crosscheck.py (which only verifies an ALREADY-PINNED design). This one
lets RES_Units/Battery_Units/Generator_Units be free integer decision variables, matching
what a real "which design is cheapest" run needs (gili_ketapang_checklist_dlensemble.md §4).

WHY THIS IS LOWER-RISK THAN IT SOUNDS: the formulation used here --
Generator_Full/Generator_Partial/Single_Flow_BESS relaxed to continuous, RES_Units/
Battery_Units/Generator_Units left as real integers -- is structurally IDENTICAL to what
MGPY_RELAX_UC=1 does in the real Pyomo pipeline (Model_Creation.py:18, 422-435): relax the
per-hour unit-commitment binaries, leave the per-step sizing integers alone. The checklist
found this relaxation necessary to make free-sizing solvable in reasonable time at all for
this design family (capexp5_9sc's first attempt timed out without it; the retry with it
solved in ~18 min). This file's "relaxed" mode already IS that recipe -- it was simply never
invoked with pinned=None (i.e. as a free-sizing solve) before 2026-08-28.

VALIDATION STATUS, read before trusting a result from this file: the underlying constraint
set (Renewable_Energy, Energy_balance, SOC recursion, partial-load generator logic, etc.) is
the SAME code validated extensively against Pyomo for capexp5_9sc's FIXED-DESIGN solve
(agreed to 0.0031% on the relaxed NPC, and linopy's integral mode produced the only certified
integral answer for that design after Pyomo's own attempt died in presolve -- see
gili_ketapang_checklist_dlensemble.md §5a-final). But this file's SIZING decision itself
(which RES/Battery/Generator quantities to build) has never been cross-checked against Pyomo
for any design -- there is no existing "right answer" to compare against for a genuinely new
design the way there was for capexp5_9sc's fixed-design step. Per session discussion: run
this ALONGSIDE Pyomo's own free-sizing solve for the same design, not instead of it, until
enough designs agree to trust linopy free-sizing solo on genuinely new points (capexp4/capexp6).

+++ DO NOT RUN WITHOUT EXPLICIT CONFIRMATION -- LINOPY_CONFIRM_REAL_SCALE=1 gate below. +++
"""
import os
import numpy as np

from linopy_core_model_builder import (
    DesignInputs, build_core_model, check_simultaneous_flows,
    start_resource_monitor, stop_resource_monitor, load_dat_scalar, resolve_plain_mode,
    recover_memlimit_incumbent,
)

INPUTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "Code", "Inputs"))

# ---- one entry per checklist design queued for free-sizing. ----
DESIGNS = {
    "capexp5_6sc": dict(
        params_dat="Parameters_6sc_20y_gili_ketapang_capexp5_MILP_dieselops.dat",
        demand_csv="Demand_6sc_20y_gili_ketapang.csv",
        res_csv="RES_Time_Series_6sc_gili_ketapang.csv",
        max_battery_kwh=74675.0,  # 6sc uses a DIFFERENT ceiling than 9sc's 62031 -- checklist §4 flags this explicitly
        max_generator_kw=9400.0,
    ),
    # LOCAL SMOKE TEST ONLY (2026-08-29) -- real Gili Ketapang data but S=1/Years=10/single-step,
    # NOT part of the dieselops checklist family (this .dat has Generator_Partial_Load=0; the
    # linopy port always builds the partial-load formulation regardless, so this is purely for
    # exercising real-scale numerics -- e.g. BIG_M_SALVAGE tightening -- locally in the `mgpy`
    # conda env, not a validated checklist design). Ceilings sized down from the 6sc/9sc values
    # to match this design's much smaller single-scenario demand (peak ~795kW, year-10 max daily
    # energy ~15,550kWh) rather than reusing an oversized constant.
    "smoketest_1sc": dict(
        params_dat="Parameters_1sc_10y_gili_ketapang_scpos2_MILP.dat",
        demand_csv="Demand_1sc_10y_gili_ketapang_scpos2.csv",
        res_csv="RES_Time_Series_1sc_gili_ketapang_scpos2.csv",
        max_battery_kwh=25000.0,
        max_generator_kw=4000.0,
    ),
    # smoketest_1sc_2y (2026-09-04, Stage 1-7 Pyomo cross-check roadmap): the SAME real design as
    # smoketest_1sc, just Years=2 instead of 10 (Demand truncated to the first 2 real year-columns,
    # same base profile/growth rate, nothing synthesized) -- built because smoketest_1sc's own
    # free-sizing MILP search (6584/12021/94 optimal units) doesn't close to a tight gap in
    # reasonable time even at S=1/Y=10, and Stage 1-7's cross-validation needs a genuine
    # free-sizing (not fixed-design-pin) comparison against real Pyomo on each stage. Same
    # ceilings reused (conservative upper bounds, not binding).
    "smoketest_1sc_2y": dict(
        params_dat="Parameters_1sc_2y_gili_ketapang_scpos2_MILP.dat",
        demand_csv="Demand_1sc_2y_gili_ketapang_scpos2.csv",
        res_csv="RES_Time_Series_1sc_gili_ketapang_scpos2.csv",
        max_battery_kwh=25000.0,
        max_generator_kw=4000.0,
    ),
    # smoketest_1sc_2y_co2factors (2026-09-04, Stage 1/6 Pyomo cross-check): same 2y base, with
    # RES/BESS/GEN_unit_CO2_emission set to round nonzero test values (1000/100/500 kg-CO2-eq per
    # kW/kWh/kW) so the RES_emission/BESS_emission/GEN_emission formulas actually get exercised
    # (the plain 2y base has all three at 0, only FUEL_unit_CO2_emission nonzero). Also the base
    # for Stage 6's multi-objective sweep, which needs a meaningful (non-degenerate) CO2 axis.
    "smoketest_1sc_2y_co2factors": dict(
        params_dat="Parameters_1sc_2y_gili_ketapang_scpos2_CO2FACTORS_MILP.dat",
        demand_csv="Demand_1sc_2y_gili_ketapang_scpos2.csv",
        res_csv="RES_Time_Series_1sc_gili_ketapang_scpos2.csv",
        max_battery_kwh=25000.0,
        max_generator_kw=4000.0,
    ),
    # smoketest_1sc_2y_landuse (2026-09-04, Stage 2 Pyomo cross-check): same 2y base,
    # Land_Use=1 + Renewables_Total_Area=5000 m2 (same value as the original Stage 2 functional
    # test) -- forces RES down from the unconstrained optimum, so (unlike every other stage
    # here) this needs a genuine fresh free-sizing search, not a design pin.
    "smoketest_1sc_2y_landuse": dict(
        params_dat="Parameters_1sc_2y_gili_ketapang_scpos2_LANDUSE_MILP.dat",
        demand_csv="Demand_1sc_2y_gili_ketapang_scpos2.csv",
        res_csv="RES_Time_Series_1sc_gili_ketapang_scpos2.csv",
        max_battery_kwh=25000.0,
        max_generator_kw=4000.0,
    ),
    # smoketest_1sc_2y_fuelesc (2026-09-04, Stage 3 Pyomo cross-check): same 2y base,
    # Fuel_Specific_Cost_Calculation=1 + Fuel_Specific_Cost_Rate=0.05 (5%/yr, same value as the
    # original Stage 3 functional test).
    "smoketest_1sc_2y_fuelesc": dict(
        params_dat="Parameters_1sc_2y_gili_ketapang_scpos2_FUELESC_MILP.dat",
        demand_csv="Demand_1sc_2y_gili_ketapang_scpos2.csv",
        res_csv="RES_Time_Series_1sc_gili_ketapang_scpos2.csv",
        max_battery_kwh=25000.0,
        max_generator_kw=4000.0,
    ),
    # smoketest_1sc_2y_wacc (2026-09-04, Stage 4 Pyomo cross-check): same 2y base, WACC_Calculation=1
    # -- cost_of_equity/cost_of_debt/tax/equity_share/debt_share are already real values, present
    # (if inert) in every .dat.
    "smoketest_1sc_2y_wacc": dict(
        params_dat="Parameters_1sc_2y_gili_ketapang_scpos2_WACC_MILP.dat",
        demand_csv="Demand_1sc_2y_gili_ketapang_scpos2.csv",
        res_csv="RES_Time_Series_1sc_gili_ketapang_scpos2.csv",
        max_battery_kwh=25000.0,
        max_generator_kw=4000.0,
    ),
    # smoketest_1sc_2y_grid (2026-09-04, Stage 5 Pyomo cross-check): same 2y base, Grid_Connection=1
    # + Grid_Availability_Simulation=1 with both outage params at 0 (forces Pyomo's own
    # auto-generated availability CSV to be constant 1.0, matching linopy's own always-available
    # simplification). Grid economics (price/cost/distance/max power) already real in the .dat.
    "smoketest_1sc_2y_grid": dict(
        params_dat="Parameters_1sc_2y_gili_ketapang_scpos2_GRID_MILP.dat",
        demand_csv="Demand_1sc_2y_gili_ketapang_scpos2.csv",
        res_csv="RES_Time_Series_1sc_gili_ketapang_scpos2.csv",
        max_battery_kwh=25000.0,
        max_generator_kw=4000.0,
    ),
    # smoketest_1sc_2y_greenfield (2026-09-04, Stage 7 Pyomo cross-check): same 2y base,
    # Greenfield_Investment=1 (DesignInputs already zeroes the five existing-capacity fields
    # when this is set -- no other .dat change needed).
    "smoketest_1sc_2y_greenfield": dict(
        params_dat="Parameters_1sc_2y_gili_ketapang_scpos2_GREENFIELD_MILP.dat",
        demand_csv="Demand_1sc_2y_gili_ketapang_scpos2.csv",
        res_csv="RES_Time_Series_1sc_gili_ketapang_scpos2.csv",
        max_battery_kwh=25000.0,
        max_generator_kw=4000.0,
    ),
    # smoketest_1sc_gridtest (Stage 5 functional test, full-parity roadmap): confirmed
    # 2026-09-04 that grid-connection machinery works end-to-end -- Grid_Connection=1 with this
    # project's own real (if previously inert) placeholder economics (Grid_Purchased_El_Price=
    # 0.138, Grid_Sold_El_Price=0, Maximum_Grid_Power=80kW, Grid_Connection_Cost=14000,
    # Grid_Distance=0.5km, Year_Grid_Connection=1) solved to NPC $5,692,394.50 (vs. off-grid's
    # $5,670,286.56 -- correctly higher, since the fixed grid connection cost outweighs the small
    # import savings at this 80kW cap; RES capacity dropped slightly, 2116.29kW vs 2172.72kW,
    # showing the optimizer did substitute some grid import for local generation). Full export
    # verified too: Results_Summary.xlsx correctly showed National Grid investment = $7,000
    # (exact match: 14000 x 0.5km), Grid electricity cost = $103, Grid CO2 emission = 0.315 ton
    # (closing Stage 1's originally-deferred GRID_emission gap). Found and fixed two real bugs
    # along the way: (1) `Energy_From_Grid <= (1-Single_Flow_Grid)*M` hit the same
    # int-minus-Variable unsupported-reflected-subtraction issue as the Salvage floor constraint
    # -- rewritten additively; (2) pyomo_instance_adapter.py's _ParamProxy had comparison/cast
    # operators but no arithmetic ones, crashing Results.py's YearlyCosts() at
    # `... * instance.Grid_Purchased_El_Price` (a raw-Param-in-arithmetic pattern no earlier,
    # all-off-grid code path had ever exercised) -- added __mul__/__add__/__sub__/__truediv__
    # and their reflected counterparts. KNOWN REMAINING GAP: Results.py's console PrintResults()
    # crashes on a MultiIndex level lookup for grid-connected designs specifically (Costs table
    # loses its 'Cost item' index level somewhere in Results.py's own ~600-line grid cost-table
    # concatenation, L503-620) -- the actual Results_Summary.xlsx/Time_Series files save
    # correctly BEFORE this point; only the cosmetic console summary fails. Not chased further
    # (pre-existing Results.py internals, no real design affected, console-only). Entry and its
    # throwaway .dat copy removed after confirming.
    # smoketest_1sc_landusetest (Stage 2 functional test, full-parity roadmap): confirmed
    # 2026-09-04 that Renewables_Max_Land_Use correctly binds -- Land_Use=1 +
    # Renewables_Total_Area=5000m2 forced RES from smoketest_1sc's own uncapped 2172.72 kW down
    # to 434.94 kW (== 5000/11.49 m2-per-kW, exact), NPC rose $5.67M -> $6.24M as expected.
    # Entry and its throwaway .dat copy removed after confirming; see this comment for the record.
    # Real dieselops formulation (Generator_Partial_Load=1, 470kW/unit -- unlike smoketest_1sc
    # above) at a small enough scale to solve locally: 6 scenarios but only 2 years (this .dat
    # was itself built 2026-08-24 as a local diagnostic to reproduce an HPC OOM crash at a
    # solvable scale, reusing the real 6sc demand/RES base). Same battery/generator ceilings as
    # the full 20y 6sc design -- these track system/demand SCALE (same underlying 6-scenario
    # demand curve, just truncated to 2 years), not project duration.
    "smoketest_6sc_2y_dieselops": dict(
        params_dat="Parameters_6sc_2y_gili_ketapang_MILP_dieselops.dat",
        demand_csv="Demand_6sc_2y_gili_ketapang_dieselops.csv",
        res_csv="RES_Time_Series_6sc_gili_ketapang.csv",
        max_battery_kwh=74675.0,
        max_generator_kw=9400.0,
    ),
    # --- merged from the HPC repo (commit 72f4c0b, 2026-08-31), verbatim ---
    "capexp5_9sc_dlensemble": dict(  # 2026-08-31 -- DL-ensemble re-certification, free-sizing
        # variant (gili_ketapang_checklist_dlensemble.md's own original purpose). The FIXED-
        # DESIGN version of this test (pinning capexp5_9sc_dieselops's own certified sizing,
        # RES/Battery/Generator = [4748,6975,10120,15407]/[10521,15806,23879,36282]/[2,2,2,4])
        # came back genuinely INFEASIBLE, not just costlier -- Lost_Load_Fraction=0.0 in this
        # .dat (zero unmet demand tolerated) and the DL-ensemble demand CSV's peak (3,433.07 kW)
        # is ~12.9% above the original Markov-chain+ERA5-proxy CSV's own peak (3,041.64 kW),
        # exceeding the pinned design's max deliverable capacity at some hour/scenario. This entry
        # instead free-sizes from scratch against the DL-ensemble demand/PV data, to find out how
        # much capacity DL-ensemble scenarios actually require -- the informative number is the
        # gap between THIS design's NPC and capexp5_9sc_dieselops's own certified 9,785.13 kUSD,
        # not a pass/fail on the old design.
        params_dat="Parameters_9sc_20y_gili_ketapang_capexp5_MILP_dieselops.dat",
        demand_csv="Demand_9sc_20y_gili_ketapang_DLensemble.csv",
        res_csv="RES_Time_Series_9sc_gili_ketapang_DLensemble.csv",
        max_battery_kwh=62031.0, max_generator_kw=9400.0),
    # --- end of merged block ---
    "capexp4_9sc": dict(  # re-enabled 2026-08-29 -- first free-sizing attempt on a design Pyomo
        # hasn't solved either; no ground-truth NPC to cross-check against yet, unlike
        # capexp5_6sc's validation-against-Pyomo runs above.
        params_dat="Parameters_9sc_20y_gili_ketapang_capexp4_MILP_dieselops.dat",
        demand_csv="Demand_9sc_20y_gili_ketapang.csv", res_csv="RES_Time_Series_9sc_gili_ketapang.csv",
        max_battery_kwh=62031.0, max_generator_kw=9400.0),
    "capexp4_9sc_5kwgen_diagnostic": dict(  # 2026-08-30 -- DIAGNOSTIC ONLY, not physically valid:
        # reverts Generator_Nominal_Capacity_milp to the OLD 5kW default (real units are 470kW
        # Cummins gensets) to test whether a fine-grained generator unit size -- like RES_Units'
        # own 0.33kW/unit, Battery_Units' 1.0kWh/unit -- makes Generator_Units' LP relaxation
        # nearly-integral on its own, avoiding the B&B stall seen at the real 470kW/unit
        # resolution. Same .dat file used for the parallel Pyomo test
        # (driver_freesizing_9sc_capexp4_5kwgen_diagnostic.py).
        params_dat="Parameters_9sc_20y_gili_ketapang_capexp4_5kwgen_diagnostic_MILP_dieselops.dat",
        demand_csv="Demand_9sc_20y_gili_ketapang.csv", res_csv="RES_Time_Series_9sc_gili_ketapang.csv",
        max_battery_kwh=62031.0, max_generator_kw=9400.0),
    "capexp4_9sc_plain_reproduce": dict(  # 2026-08-30 -- reproduction of the plain (non-dieselops)
        # capexp4_9sc free-sizing solve that originally certified the paper's Table 6/7 headline
        # (Pyomo NPC 9,646.33 kUSD, gap 0.0605%, 1 node explored, 477.55s -- logs/Solver_Output_
        # 20260820_165133_9sc_20y_capexp4_5step.log). Requires LINOPY_PLAIN=1 (see that flag's
        # comment above) -- without it this file always builds the full unit-commitment
        # formulation regardless of which .dat is pointed at, which would NOT be a valid
        # reproduction (see capexp4_9sc_5kwgen_diagnostic above, which already showed 5kW alone
        # does not avoid the stall when dieselops complexity stays in).
        params_dat="Parameters_9sc_20y_gili_ketapang_capexp4_MILP.dat",
        demand_csv="Demand_9sc_20y_gili_ketapang.csv", res_csv="RES_Time_Series_9sc_gili_ketapang.csv",
        max_battery_kwh=62031.0, max_generator_kw=9400.0),
    "capexp5_9sc_plain_reproduce": dict(  # 2026-08-30 -- same as capexp4_9sc_plain_reproduce
        # above but the 4-step point (Step_Duration=5). Paper headline Pyomo NPC 9,670.08 kUSD.
        # Requires LINOPY_PLAIN=1.
        params_dat="Parameters_9sc_20y_gili_ketapang_capexp5_MILP.dat",
        demand_csv="Demand_9sc_20y_gili_ketapang.csv", res_csv="RES_Time_Series_9sc_gili_ketapang.csv",
        max_battery_kwh=62031.0, max_generator_kw=9400.0),
    "capexp2_9sc_plain_new_point": dict(  # 2026-08-30 -- NEW plain-model point for the paper's
        # Table 6/7 step-duration trend (Step_Duration=2, 10 even 2-year steps -- finest tested
        # so far). Built from capexp5_9sc's own plain .dat, Step_Duration line only (diffed
        # clean, 1 line). Same settings as the other points: LINOPY_PLAIN=1 +
        # LINOPY_HISTORICAL_REVERT=1, default BESS_FORM=binary.
        params_dat="Parameters_9sc_20y_gili_ketapang_capexp2_MILP.dat",
        demand_csv="Demand_9sc_20y_gili_ketapang.csv", res_csv="RES_Time_Series_9sc_gili_ketapang.csv",
        max_battery_kwh=62031.0, max_generator_kw=9400.0),
    "capexp6_9sc_plain_new_point": dict(  # 2026-08-30 -- NEW plain-model point for the paper's
        # Table 6/7 step-duration trend (Step_Duration=6, 3 full 6y steps + uneven final 2y step)
        # -- did not exist before (checklist_dlensemble.md explicitly noted only a DIESELOPS
        # capexp6 was built, "no plain (non-dieselops) capexp6"). Built from capexp5_9sc's own
        # plain .dat, Step_Duration line only (diffed clean, 1 line). Must be run with the SAME
        # settings as the other 4 points in that table for a fair comparison: LINOPY_PLAIN=1 +
        # LINOPY_HISTORICAL_REVERT=1 (default BESS_FORM=binary, NOT nobinary -- the dieselops-
        # family capexp6_9sc run's nobinary setting does not apply here, different table/purpose).
        params_dat="Parameters_9sc_20y_gili_ketapang_capexp6_MILP.dat",
        demand_csv="Demand_9sc_20y_gili_ketapang.csv", res_csv="RES_Time_Series_9sc_gili_ketapang.csv",
        max_battery_kwh=62031.0, max_generator_kw=9400.0),
    "capexp2_6sc_plain_new_point": dict(  # 2026-08-30 -- 6sc parity for capexp2_9sc_plain_new_point
        # (Step_Duration=2, 10 even 2-year steps). Built from the 6sc capexp5 plain .dat,
        # Step_Duration line only. Uses 6sc's own ceiling (74675.0 kWh, DIFFERENT from 9sc's
        # 62031.0 -- checklist §4 flags this explicitly). Run with LINOPY_PLAIN=1 +
        # LINOPY_HISTORICAL_REVERT=1, default BESS_FORM=binary, matching the 9sc point.
        params_dat="Parameters_6sc_20y_gili_ketapang_capexp2_MILP.dat",
        demand_csv="Demand_6sc_20y_gili_ketapang.csv", res_csv="RES_Time_Series_6sc_gili_ketapang.csv",
        max_battery_kwh=74675.0, max_generator_kw=9400.0),
    "capexp6_6sc_plain_new_point": dict(  # 2026-08-30 -- 6sc parity for capexp6_9sc_plain_new_point
        # (Step_Duration=6, 3 full 6y steps + uneven final 2y step). Built from the 6sc capexp5
        # plain .dat, Step_Duration line only. Same 74675.0 kWh battery ceiling as capexp2_6sc
        # above (6sc-specific, per checklist §4). Run with LINOPY_PLAIN=1 +
        # LINOPY_HISTORICAL_REVERT=1, default BESS_FORM=binary.
        params_dat="Parameters_6sc_20y_gili_ketapang_capexp6_MILP.dat",
        demand_csv="Demand_6sc_20y_gili_ketapang.csv", res_csv="RES_Time_Series_6sc_gili_ketapang.csv",
        max_battery_kwh=74675.0, max_generator_kw=9400.0),
    "capexp4_9sc_10y_diagnostic": dict(  # 2026-08-29 -- quick diagnostic: does a shorter horizon
        # (half the dispatch variables) avoid the B&B node-degeneracy stall seen on capexp4_9sc's
        # real 20y run? NOTE: this .dat's Step_Duration=20>=Years=10 collapses to STEPS_NUMBER=1
        # (single investment step, ~3 integers) instead of capexp4_9sc's real 5-step/~15-integer
        # structure -- confounds "shorter horizon" with "far fewer integers." Proceeding anyway
        # as a quick/rough signal, not a controlled isolation of horizon length alone.
        params_dat="Parameters_9sc_10y_gili_ketapang_MILP_dieselops.dat",
        demand_csv="Demand_9sc_10y_gili_ketapang_dieselops.csv",
        res_csv="RES_Time_Series_9sc_gili_ketapang.csv",
        max_battery_kwh=62031.0, max_generator_kw=9400.0),
    "capexp4_9sc_10y_5step_clean": dict(  # 2026-08-29 -- CONTROLLED diagnostic: same 5-step
        # structure as the real 20y capexp4_9sc (Step_Duration halved 4->2 alongside Years
        # halved 20->10, so STEPS_NUMBER=ceil(10/2)=5 -- IDENTICAL integer sizing search space,
        # ~15 integers, to the real design). Only the continuous dispatch LP shrinks (half the
        # years). Isolates whether LP-size reduction ALONE (not fewer integers) avoids the B&B
        # node-degeneracy stall -- the uncontrolled capexp4_9sc_10y_diagnostic above also
        # collapsed to a single step, confounding the two effects. demand_csv built by slicing
        # the first 10 years of each of the 9 scenario blocks from the real 20y demand CSV
        # (columns (s-1)*20+y for y=1..10, re-keyed to (s-1)*10+y).
        params_dat="Parameters_9sc_10y_gili_ketapang_capexp4_5step_clean_MILP_dieselops.dat",
        demand_csv="Demand_9sc_10y_gili_ketapang_capexp4_5step_clean.csv",
        res_csv="RES_Time_Series_9sc_gili_ketapang.csv",
        max_battery_kwh=62031.0, max_generator_kw=9400.0),
    "capexp4_9sc_20y_1step_clean": dict(  # 2026-08-29 -- COMPANION controlled diagnostic to
        # capexp4_9sc_10y_5step_clean above: this time holds the REAL 20-year horizon and
        # demand/RES data IDENTICAL to capexp4_9sc (same dispatch LP size as the stuck
        # capexp4_9sc_export2 job), but collapses Step_Duration 4->20 so STEPS_NUMBER=1
        # (~3 integers instead of ~15). If this solves fast despite the full 20y LP, it
        # confirms step-count -- not horizon/LP size -- is the actual B&B degeneracy driver.
        params_dat="Parameters_9sc_20y_gili_ketapang_capexp4_1step_clean_MILP_dieselops.dat",
        demand_csv="Demand_9sc_20y_gili_ketapang.csv",
        res_csv="RES_Time_Series_9sc_gili_ketapang.csv",
        max_battery_kwh=62031.0, max_generator_kw=9400.0),
    "capexp6_9sc": dict(  # re-enabled 2026-08-29 -- third parallel free-sizing job alongside
        # capexp5_6sc and capexp4_9sc, run at reduced Threads=8 since the SLURM allocation is
        # capped at 24 CPUs and the other two already reserve 12 each. No warm-start file exists
        # for this design yet -- starts cold.
        params_dat="Parameters_9sc_20y_gili_ketapang_capexp6_MILP_dieselops.dat",
        demand_csv="Demand_9sc_20y_gili_ketapang.csv", res_csv="RES_Time_Series_9sc_gili_ketapang.csv",
        max_battery_kwh=62031.0, max_generator_kw=9400.0),
    # --- merged from the HPC repo (commit 72f4c0b, 2026-08-31), verbatim ---
    "capexp4_6sc": dict(  # 2026-08-31 -- 6sc parity for capexp4_9sc, part of the four remaining
        # dieselops step-duration points (checklist_dlensemble.md §1's table). Never built before
        # today (was a placeholder comment only). 5-step (Step_Duration=4), 6sc's own battery
        # ceiling (74675.0, DIFFERENT from 9sc's 62031 -- checklist §4).
        params_dat="Parameters_6sc_20y_gili_ketapang_capexp4_MILP_dieselops.dat",
        demand_csv="Demand_6sc_20y_gili_ketapang.csv", res_csv="RES_Time_Series_6sc_gili_ketapang.csv",
        max_battery_kwh=74675.0, max_generator_kw=9400.0),
    "capexp6_6sc": dict(  # 2026-08-31 -- 6sc parity for capexp6_9sc (dieselops, NOT to be
        # confused with capexp6_6sc_plain_new_point above, which is the plain/non-dieselops
        # sibling for the Table 6/7 sweep). Never built before today. 3-step-uneven
        # (Step_Duration=6, 20y/6y steps do not divide evenly -- see capexp6_9sc's own comment
        # elsewhere in this project for the uneven-final-step handling, which is generic).
        params_dat="Parameters_6sc_20y_gili_ketapang_capexp6_MILP_dieselops.dat",
        demand_csv="Demand_6sc_20y_gili_ketapang.csv", res_csv="RES_Time_Series_6sc_gili_ketapang.csv",
        max_battery_kwh=74675.0, max_generator_kw=9400.0),
    # --- end of merged block ---
}

# LINOPY_SITE (2026-09-08) -- config-driven alternative to LINOPY_DESIGN/DESIGNS: reads
# sites/<name>.conf (the SAME convention Pyomo's run_site_interactive.sh/submit_site.sh already
# use) instead of requiring a hand-edited DESIGNS entry. See site_config.py's own docstring.
_LINOPY_SITE = os.environ.get("LINOPY_SITE")
if _LINOPY_SITE:
    if os.environ.get("LINOPY_DESIGN"):
        raise ValueError("Set only one of LINOPY_SITE or LINOPY_DESIGN, not both.")
    import site_config
    DESIGN_NAME = _LINOPY_SITE
    D = site_config.load_site_design(_LINOPY_SITE)
else:
    DESIGN_NAME = os.environ.get("LINOPY_DESIGN", "capexp5_6sc")
    D = DESIGNS[DESIGN_NAME]

RES_CSV = os.path.join(INPUTS_DIR, D["res_csv"])
DEMAND_CSV = os.path.join(INPUTS_DIR, D["demand_csv"])
PARAMS_DAT = os.path.join(INPUTS_DIR, D["params_dat"])


# RES_CF_FLOOR (physically negligible dawn/dusk irradiance, <0.5% of nameplate -- the actual
# source of the model's ~3e-3 matrix-range floor) now lives in DesignInputs.RES_CF_FLOOR
# (linopy_core_model_builder.py) -- HISTORICAL_REVERT below disables it via
# apply_res_cf_floor=False at construction time instead of a local module constant.

# LINOPY_BESS_FORM (2026-08-29/30) -- mirrors Pyomo's MGPY_BESS_FORM. "binary" (default) is the
# genuine Single_Flow_BESS mutual-exclusion binary (the 2026-08-29 fix, matching Model_Creation.py
# unconditionally). "nobinary" drops that binary and its two gate constraints entirely, betting
# round-trip efficiency losses make simultaneous charge/discharge uneconomic -- the same bet
# Model_Resolution.py's MGPY_BESS_FORM=nobinary makes, with the same mandatory post-hoc validity
# check (see check_simultaneous_flows() below) required to trust the result. Testing this because
# even with Single_Flow_BESS correctly binary, capexp4_9sc/capexp6 still hit severe B&B
# degeneracy (~1.58M binaries is exactly the kind of scale the literature on this exact
# complementarity-constraint relaxation targets -- see session research, IEEE/arXiv sufficient-
# exactness-condition papers).
BESS_FORM = os.environ.get("LINOPY_BESS_FORM", "binary").strip().lower()
if BESS_FORM not in ("binary", "nobinary"):
    raise ValueError(f"LINOPY_BESS_FORM={BESS_FORM!r} not recognised -- use 'binary' or 'nobinary'.")

# LINOPY_RELAX_GEN_UNITS (2026-08-30) -- relaxes Generator_Units (470 kW/unit -- the real
# physical Cummins genset size) to continuous during free-sizing, while RES_Units (0.33 kW/unit)
# and Battery_Units (1.0 kWh/unit) stay genuine integers as before. Hypothesis: RES/Battery are
# already so fine-grained per unit that their LP relaxation is nearly integral on its own
# (rounding ~4744 units by <1 changes total capacity by <1 kW out of thousands) -- they were
# never really the source of B&B difficulty. Generator_Units, being coarse (only 2-6 units total
# across a 9400kW ceiling, 470kW each), has a much larger relaxation-to-integer rounding gap in
# relative terms, and is the more plausible driver. This flag tests that in isolation: solve with
# Generator_Units continuous (fast, if the hypothesis holds), then round the printed result to
# the nearest integer and re-verify via linopy_fixed_design_crosscheck.py at the true 470kW
# resolution -- the same relax-then-round-then-certify workflow this project already uses for
# fixed-design verification, just applied one level up to the sizing decision itself.
RELAX_GEN_UNITS = os.environ.get("LINOPY_RELAX_GEN_UNITS", "0") == "1"

# LINOPY_PLAIN (2026-08-30) -- mirrors Pyomo's Generator_Partial_Load=0 branch
# (Constraints.py ~1930/2664): drops Generator_Partial/Generator_Full/Generator_Energy_Partial
# and the Generator_Units_Total/Generator_Energy_Total_def constraints entirely; generator
# dispatch is then just Generator_Energy_Total bounded by installed capacity and demand
# (Maximum_Generator_Energy_Total_1/2, already unconditional below), fuel cost a flat
# Generator_Energy_Total * marginal-cost term with no start-cost/partial-load split. This is
# what the paper's own Table 6/7 headline (plain, non-dieselops, 5kW/unit) actually used.
# 2026-09-11: PLAIN_MODE is now derived from the .dat's own Generator_Partial_Load flag right
# after DesignInputs loads below (resolve_plain_mode()); LINOPY_PLAIN is still accepted if it
# agrees with the .dat. Before this, the file built the unit-commitment formulation for every
# .dat unless LINOPY_PLAIN=1 was set by hand.

# LINOPY_LP_MODE (2026-09-04, Stage 8 of the full-parity roadmap) -- switches build_core_model()
# to the LP/continuous formulation (Constraints_Brownfield's non-_Milp class): RES/Battery/
# Generator sizing all continuous, Battery_Units/Generator_Units become
# Battery_Nominal_Capacity/Generator_Nominal_Capacity directly (kWh/kW, not unit counts), and
# lp_mode forces plain_mode=True internally (LP has no partial-load unit-commitment machinery).
# No certified Gili Ketapang design uses this -- see build_core_model()'s own docstring for the
# verified small-scale functional-test result before trusting a real LP-mode design's NPC.
LP_MODE = os.environ.get("LINOPY_LP_MODE", "0") == "1"

# LINOPY_SOLVER (2026-09-05) -- mirrors Pyomo's own Solver .dat parameter (0=Gurobi/1=GLPK/
# 2=CPLEX, genuinely switchable there with no code change). linopy itself supports many backends
# (gurobi/highs/cbc/glpk/cplex/...), but every driver script in this port had solver_name="gurobi"
# hardcoded until now. Default "gurobi" preserves every existing regression baseline exactly --
# SOLVER_KWARGS below is a Gurobi-tuned recipe (MIPFocus/Crossover/BarHomogeneous/...) built up
# over many real solves; those option NAMES are Gurobi-specific (linopy passes kwargs straight
# through to whichever backend's own Python API, no cross-solver abstraction), so switching away
# from "gurobi" discards that whole recipe rather than risk passing an unrecognized option to a
# different solver -- this project has no tuned recipe for any other backend yet, and none has
# ever actually been run/validated here (same honest-gap standard as everywhere else in this
# port: untested territory is stated as such, not silently assumed to work).
LINOPY_SOLVER = os.environ.get("LINOPY_SOLVER", "gurobi").strip().lower()

# LINOPY_HISTORICAL_REVERT (2026-08-30) -- diagnostic-only mirror of the same flag on the Pyomo
# side (Constraints.py/Model_Resolution.py/Initialize.py): disables the RES capacity-factor floor
# (RES_CF_FLOOR) and the Salvage_Value floor-at-zero Big-M linearization, reverting to a plain
# equality Salvage_Value == raw formula -- both of these already existed in this file from day
# one (unlike Pyomo, this port was written AFTER those fixes existed upstream), so this flag
# exists purely to reproduce the pre-fix numerics for the capexp4_9sc_plain_reproduce historical-
# match test. NOT meant to be left on.
HISTORICAL_REVERT = os.environ.get("LINOPY_HISTORICAL_REVERT", "0") == "1"


# Scenarios/Years/Periods/Step_Duration/cf/demand/all cost & tech parameters are now loaded by
# the shared DesignInputs (linopy_core_model_builder.py, Stage 0 of the full-parity roadmap).
# apply_res_cf_floor=not HISTORICAL_REVERT reproduces this file's own historical-revert behavior
# for the RES capacity-factor floor exactly as before.
inputs = DesignInputs(
    params_dat=PARAMS_DAT, demand_csv=DEMAND_CSV, res_csv=RES_CSV,
    max_battery_kwh=D["max_battery_kwh"], max_generator_kw=D["max_generator_kw"],
    apply_res_cf_floor=not HISTORICAL_REVERT,
)
PLAIN_MODE = resolve_plain_mode(inputs)
print(f"Generator_Partial_Load={inputs.GENERATOR_PARTIAL_LOAD} -> "
      f"{'plain dispatch' if PLAIN_MODE else 'partial-load unit commitment'} formulation")
S, YEARS, PERIODS, STEP_DURATION, STEPS_NUMBER = inputs.S, inputs.YEARS, inputs.PERIODS, inputs.STEP_DURATION, inputs.STEPS_NUMBER
ut_idx = inputs.ut_idx
print(f"Design '{DESIGN_NAME}' (FREE-SIZING): S={S}, YEARS={YEARS}, PERIODS={PERIODS}, STEP_DURATION={STEP_DURATION}, STEPS_NUMBER={STEPS_NUMBER}")

# Same recipe as the fixed-design tool; TimeLimit added since this is genuinely untested at
# real scale for free-sizing -- Pyomo's own successful capexp5_9sc free-sizing (WITH
# MGPY_RELAX_UC=1) took ~18 min to gap 0.24%, so 2h is generous headroom, not a measured need.
# Restored (2026-08-29, v12 recipe) -- this block keeps getting reverted on disk by concurrent
# edits elsewhere; re-applying each time so any NEW process launched from this file matches the
# THREE JOBS ALREADY RUNNING (capexp5_6sc/capexp4_9sc/capexp6_9sc export runs, launched with this
# exact recipe already in memory -- they're unaffected by on-disk reverts, but a new launch isn't).
# What actually closed capexp5_6sc's gap to 0.30% (v12, see logs/linopy_freesizing_capexp5_6sc_
# v12_*.log and [[mgpy-free-sizing-gurobi-tuning]]) was MIPFocus=3/Cuts=1/CutPasses=1 (matching
# linopy_fixed_design_crosscheck.py's own proven recipe) + the NodefileStart fix below + a real
# 5h TimeLimit.
SOLVER_KWARGS = dict(
    OutputFlag=1,
    Threads=int(os.environ.get("SLURM_CPUS_PER_TASK", 24)),
    Method=2, BarHomogeneous=1, Crossover=1,   # root LP treatment -- millions of rows/cols,
    # needs barrier + homogeneous + crossover to solve at all in reasonable time.
    MIPFocus=3, Cuts=1, CutPasses=1,           # the v12 recipe (see comment above).
    BarConvTol=1e-3, OptimalityTol=1e-3, FeasibilityTol=1e-4,
    MIPGap=0.03, NodefileStart=100,           # was 0.1 (2026-08-28) -- NodefileStart is an
    # ABSOLUTE GB threshold, NOT a fraction of MemLimit (confirmed via gurobipy's own Params
    # docstring: "memory nodes may use (in GB) before being written to disk"). 0.1 meant 100MB,
    # an absurdly low threshold that triggered node-file disk swapping to /scratch2/.../grbnodes
    # almost immediately for a model this size -- node exploration was running at ~1500s/node (4
    # nodes in 7205s), and disk I/O at this problem scale (millions of rows) is the leading
    # suspect. 100 -> 100GB in-memory threshold, comfortably above the observed peak RSS
    # (~35-38GB).
    MemLimit=120,                              # was 220 (2026-08-29) -- lowered to fit multiple
    # designs run in parallel under this session's real 250GB SLURM budget. Observed peak RSS
    # per process is only ~35-38GB, so this is still generous headroom.
    TimeLimit=5 * 3600,                        # was 2*3600 -- restored to the v12 recipe's
    # budget; every prior 2h-capped attempt hit the wall without closing to the 3% MIPGap target.
)

# Optional single-knob overrides for isolated diagnostic runs (2026-08-29) -- e.g. testing
# whether DegenMoves=0 breaks the astronomically-low-node-throughput degeneracy pattern seen on
# capexp4_9sc/capexp5_6sc (200k-300k simplex iterations per node, bound frozen for hours -- same
# signature as https://support.gurobi.com/hc/en-us/community/posts/360077721291), without
# touching the base recipe every other launch relies on. Each is a no-op unless its env var is
# set, so this doesn't change behavior for the three jobs already running or any future default
# launch.
if os.environ.get("LINOPY_DEGENMOVES") is not None:
    SOLVER_KWARGS["DegenMoves"] = int(os.environ["LINOPY_DEGENMOVES"])
if os.environ.get("LINOPY_NODEMETHOD") is not None:
    SOLVER_KWARGS["NodeMethod"] = int(os.environ["LINOPY_NODEMETHOD"])
if os.environ.get("LINOPY_MIPFOCUS") is not None:
    SOLVER_KWARGS["MIPFocus"] = int(os.environ["LINOPY_MIPFOCUS"])
if os.environ.get("LINOPY_CUTS") is not None:
    SOLVER_KWARGS["Cuts"] = int(os.environ["LINOPY_CUTS"])
if os.environ.get("LINOPY_CUTPASSES") is not None:
    SOLVER_KWARGS["CutPasses"] = int(os.environ["LINOPY_CUTPASSES"])
elif os.environ.get("LINOPY_CUTS") is not None:
    SOLVER_KWARGS.pop("CutPasses", None)  # base recipe's CutPasses=1 only makes sense paired with Cuts=1
if os.environ.get("LINOPY_MEMLIMIT") is not None:
    SOLVER_KWARGS["MemLimit"] = int(os.environ["LINOPY_MEMLIMIT"])
if os.environ.get("LINOPY_SOFTMEMLIMIT") is not None:
    # SoftMemLimit (2026-09-12) -- unlike MemLimit, Gurobi STOPS and keeps its incumbent instead
    # of raising "Out of memory": the 12sc 5-step run lost a 6%-gap design to MemLimit=100.
    SOLVER_KWARGS["SoftMemLimit"] = float(os.environ["LINOPY_SOFTMEMLIMIT"])
if os.environ.get("LINOPY_NODEFILESTART") is not None:
    # Absolute GB of branch-and-bound tree kept in RAM before spilling to NodefileDir (disk).
    SOLVER_KWARGS["NodefileStart"] = float(os.environ["LINOPY_NODEFILESTART"])
if os.environ.get("LINOPY_NODEFILEDIR") is not None:
    SOLVER_KWARGS["NodefileDir"] = os.environ["LINOPY_NODEFILEDIR"]
if os.environ.get("LINOPY_THREADS") is not None:
    # Fewer threads means less memory: each thread keeps its own copy of the node LP.
    SOLVER_KWARGS["Threads"] = int(os.environ["LINOPY_THREADS"])
if os.environ.get("LINOPY_SOLFILES") is not None:
    # SolFiles (2026-09-12) -- Gurobi writes <prefix>_<n>.sol EVERY time it improves the
    # incumbent, so a run killed later (memory, time limit, session end) still leaves its best
    # design on disk. Feed one back into the next run with LINOPY_WARMSTART_MST: that restores
    # the incumbent, not the search tree or the bound, so the root relaxation is recomputed.
    SOLVER_KWARGS["SolFiles"] = os.environ["LINOPY_SOLFILES"]
# LINOPY_SCALEFLAG (2026-08-31, merged from the HPC repo) -- mirrors the same flag in
# linopy_fixed_design_crosscheck.py and Pyomo's own MGPY_SCALEFLAG: unset = automatic (Gurobi
# default, -1); 0 = none, 1 = equilibrium, 2 = geometric mean, 3 = aggressive. Proven on
# capexp4_9sc's fixed-design certification (turned a run that never finished into one solving in
# 159s) -- tried here on free-sizing too since the same badly-scaled coefficient structure
# (GEN_NOM_CAP_KW=470 as a Big-M multiplier) is present in both scripts.
if os.environ.get("LINOPY_SCALEFLAG") is not None:
    SOLVER_KWARGS["ScaleFlag"] = int(os.environ["LINOPY_SCALEFLAG"])
# LINOPY_TIMELIMIT (2026-08-31, merged from the HPC repo) -- override the base recipe's 5h
# TimeLimit (seconds) for the Gurobi path. Before this merge the laptop copy honoured it only on
# the HiGHS branch, although docs/microgridspy_pipeline_usage.tex already described it as a
# general free-sizing cap.
if os.environ.get("LINOPY_TIMELIMIT") is not None:
    SOLVER_KWARGS["TimeLimit"] = int(os.environ["LINOPY_TIMELIMIT"])
if os.environ.get("LINOPY_CUTOFF") is not None:
    # LINOPY_CUTOFF (2026-09-12, 12sc genset enumeration) -- Gurobi's Cutoff: prune everything
    # whose objective is worse than this value. Set it to the best OBJECTIVE found so far for the
    # same design (the "Best objective" line, i.e. NPC true minus the constant offset) when
    # enumerating genset patterns: a pattern that cannot beat it then ends quickly as
    # infeasible/cutoff instead of spending hours closing its own gap. A run that ends this way
    # proves the pattern is NOT better -- it does not give that pattern's own optimum.
    SOLVER_KWARGS["Cutoff"] = float(os.environ["LINOPY_CUTOFF"])
if os.environ.get("LINOPY_MIPGAP") is not None:
    # General override (2026-09-04, Stage 1-7 Pyomo cross-check roadmap) -- e.g. tightening to
    # 1e-4/1e-5 on the small 2y cross-validation designs, which converge fast enough at that
    # scale to make a genuinely tight comparison against Pyomo worthwhile (unlike the real
    # capexp5_9sc-scale designs this base 0.03 recipe is tuned for).
    SOLVER_KWARGS["MIPGap"] = float(os.environ["LINOPY_MIPGAP"])

# LP_MODE tightening (2026-09-04, Stage 8 cross-validation): the base recipe's MIPGap=0.03 is
# tuned for genuinely large MILP problems (capexp5_9sc scale) where closing to a tight gap is
# infeasible in reasonable time. lp_mode's own problem is near-continuous (RES/Battery/Generator
# capacity is all free continuous; the only integer artifact is the single Salvage_Positive_Flag
# floor-linearization binary, exactly like Pyomo's own LP class) -- Gurobi can and should close
# it to an EXACT (0.0000%) gap quickly, same as it does when Pyomo's own MicroGrids.py solves the
# real LP formulation. Confirmed by a real cross-validation run on smoketest_1sc, 2026-09-04:
# leaving MIPGap=0.03 here gave a plausible-looking but LOOSE $5,658,231.42 (RES=1793.96kW,
# Gen=472.46kW) -- 2.13% off Pyomo's real, exact-gap LP answer; re-solving this exact same
# lp_mode model with a tight MIPGap=1e-6 instead reproduced Pyomo's $5,540,329.46 (RES=
# 1858.62kW, Battery=12,020.92kWh, Generator=470.00kW) to 6+ significant figures. Tightening
# automatically here (rather than requiring every lp_mode caller to remember a special recipe)
# closes off that same trap for any future run.
if LP_MODE:
    SOLVER_KWARGS["MIPGap"] = 1e-6
    SOLVER_KWARGS["OptimalityTol"] = 1e-6

# start_resource_monitor/stop_resource_monitor now live in linopy_core_model_builder.py (Stage 0
# -- moved there so linopy_fixed_design_crosscheck.py gets one too, see that module's docstring).


# check_simultaneous_flows now lives in linopy_core_model_builder.py (Stage 0 -- imported above).


def build_and_solve_free_sizing():
    """mode is always 'relaxed' here -- Generator_Partial/Generator_Full continuous (==
    MGPY_RELAX_UC=1's actual effect per Model_Creation.py:422-435), RES_Units/Battery_Units/
    Generator_Units left as real integers (the actual sizing decision). No Fix_* constraints --
    sizing is free.
    Single_Flow_BESS is a genuine Binary here (2026-08-29 fix) -- Model_Creation.py:379 shows
    the real Pyomo pipeline hardcodes this as Binary unconditionally, NOT gated by RELAX_UC at
    all. This file previously (wrongly) also relaxed it to continuous, based on an inaccurate
    module-docstring claim that all three were relaxed together under MGPY_RELAX_UC. That EXTRA
    relaxation is the likely root cause of the severe B&B node-degeneracy this file's free-sizing
    solves have been hitting: relaxing a mutual-exclusion binary (charge XOR discharge) to a
    continuous fraction creates a large, flat, degenerate optimal face (many near-equal
    fractional splits), which is exactly the "astronomically low node throughput" signature
    Gurobi support has documented elsewhere. Pyomo's own logs (e.g.
    Solver_Output_20260825_203026_9sc_20y_gili_ketapang_MILP_dieselops_relaxuc_cuts1_cutpasses1_maxgen.log)
    show it keeps ~1.58M Single_Flow_BESS binaries genuinely integral and still solves this exact
    design in ~25 minutes to 1.65% gap -- so the scale itself was never the problem."""
    # LINOPY_FIX_GENERATOR_UNITS (2026-09-11): comma-separated CUMULATIVE generator unit counts
    # per step (e.g. "2,2,3"). Pins only Generator_Units; PV and battery stay free. Used to
    # enumerate the few possible genset counts of the 12sc designs, whose full free sizing stalls
    # at the root node with a 2.7-8.7% gap.
    fix_gen = os.environ.get("LINOPY_FIX_GENERATOR_UNITS", "").strip()
    pinned = None
    if fix_gen:
        import xarray as xr
        vals = [float(v) for v in fix_gen.split(",")]
        if len(vals) != len(inputs.ut_idx):
            raise SystemExit(f"LINOPY_FIX_GENERATOR_UNITS has {len(vals)} value(s), the design has "
                             f"{len(inputs.ut_idx)} investment step(s)")
        pinned = {"Generator": xr.DataArray(vals, coords={"ut": list(inputs.ut_idx)}, dims="ut")}
        print(f"LINOPY_FIX_GENERATOR_UNITS: Generator_Units pinned to {vals} (PV and battery free)")
    built = build_core_model(inputs, pinned=pinned, bess_mode=BESS_FORM, plain_mode=PLAIN_MODE,
                              relax_gen_units=RELAX_GEN_UNITS, salvage_floor=not HISTORICAL_REVERT,
                              battery_min_capacity=True, lp_mode=LP_MODE)
    m = built["m"]
    # Optional MIP warm start (2026-08-28): LINOPY_WARMSTART_MST points at a Gurobi .mst file
    # seeding the integer sizing variables (RES_Units/Battery_Units/Generator_Units) from a
    # prior run's incumbent. A partial start; Gurobi fills in the rest (continuous dispatch,
    # Salvage_Positive_Flag) via its own root heuristics. Goal: skip re-discovering a design B&B
    # already found, and spend the time budget closing the gap instead.
    if LINOPY_SOLVER == "gurobi":
        solve_kwargs = dict(SOLVER_KWARGS)
        warmstart_fn = os.environ.get("LINOPY_WARMSTART_MST")
        if warmstart_fn:
            print(f"Warm-starting from MIP start file: {warmstart_fn}")
            solve_kwargs["warmstart_fn"] = warmstart_fn
    elif LINOPY_SOLVER == "highs":
        # 2026-09-05: three real tests on smoketest_1sc_2y, not assumed. (1) No time limit/gap
        # tuning: 4,914.83s (~82 min) to reach HiGHS's own "optimal" -- vs. Gurobi's few-second
        # solve on the identical design, but at least landed on the correct answer
        # (NPC $2,713,937 vs the Gurobi/Pyomo-validated $2,713,778, agreeing closely). (2) The
        # Gurobi-recipe's own MIPGap=0.03 inherited naively: HiGHS reported "optimal" at 3% but
        # actually returned a MATERIALLY WRONG, cheaper-looking answer -- NPC $2,788,967 (2.77%
        # too high) with RES=789kW vs the correct ~1487kW, a genuinely different technology mix,
        # not just a looser number on the same design. (3) Tightening to MIPGap=1e-4 with a 300s
        # time budget did NOT fix it -- HiGHS hit the time limit stuck on the IDENTICAL wrong
        # incumbent, meaning this isn't purely a gap-tuning problem: this project's MILP
        # structure (partial-load generator unit commitment) appears genuinely hard for HiGHS's
        # own branching/cuts, independent of the gap target. **Conclusion: do not trust a HiGHS
        # "optimal" status on this project's designs without an independent Gurobi/Pyomo
        # cross-check** -- a falsely-confident wrong answer is worse than an honest timeout, so
        # the default gap here is kept tight (not the loose Gurobi-tuned 0.03) so HiGHS is more
        # likely to honestly report `time_limit` than to falsely claim `optimal` on a bad
        # incumbent; override via LINOPY_MIPGAP if you understand this tradeoff. Option names
        # below are HiGHS's own (`highspy`'s documented names, NOT Gurobi's -- `time_limit` not
        # `TimeLimit`, `mip_rel_gap` not `MIPGap`).
        solve_kwargs = dict(
            time_limit=float(os.environ.get("LINOPY_TIMELIMIT", 1800)),
            mip_rel_gap=float(os.environ.get("LINOPY_MIPGAP", 1e-4)),
        )
    else:
        # SOLVER_KWARGS/warmstart_fn above are Gurobi-specific option names -- see LINOPY_SOLVER's
        # own comment. No tuned recipe (or even a basic time-limit default) exists yet for any
        # backend besides gurobi/highs -- both confirmed installed and working in this
        # environment (linopy.solvers.available_solvers); anything else is genuinely untested.
        solve_kwargs = {}
        print(f"LINOPY_SOLVER={LINOPY_SOLVER!r} -- using that backend's own defaults, "
              f"none of this project's Gurobi-tuned solver options apply, and no time limit is "
              f"set. Confirm this solver is actually installed before trusting this to return.")
    start_resource_monitor()
    try:
        status, condition = m.solve(solver_name=LINOPY_SOLVER, **solve_kwargs)
    finally:
        stop_resource_monitor()
    # A SoftMemLimit stop (Gurobi status 17) keeps its incumbent but linopy reports it as an
    # error and loads nothing; recover it and treat it like a time-limit stop (2026-09-18).
    if status != "ok" and recover_memlimit_incumbent(m):
        print("Gurobi stopped on its memory limit (status 17) with an incumbent: recovered it; "
              "it is a feasible design, not a proven optimum -- read the gap.")
        status, condition = "ok", "memory_limit"
    # A run stopped by its time or (soft) memory limit still has an incumbent worth keeping, so
    # the NPC is read whenever a solution exists, not only when the status is "ok" (2026-09-12).
    try:
        npc_true = m.objective.value + built["total_const_offset"]
    except Exception:
        npc_true = None
    if BESS_FORM == "nobinary" and status == "ok":
        print("\n" + "=" * 78)
        print("LINOPY_BESS_FORM=nobinary -- MANDATORY VALIDITY CHECK")
        print("=" * 78)
        check_simultaneous_flows(built["Battery_Inflow"].solution, built["Battery_Outflow"].solution)
        print("=" * 78 + "\n")
    return dict(m=m, status=status, condition=condition, npc_true=npc_true, total_const_offset=built["total_const_offset"], **{
        k: v for k, v in built.items() if k not in ("m", "total_const_offset")
    })


if os.environ.get("LINOPY_CONFIRM_REAL_SCALE") != "1":
    print("\n" + "!" * 78)
    print(f"STOPPING HERE for design '{DESIGN_NAME}' (free-sizing). Set LINOPY_CONFIRM_REAL_SCALE=1")
    print("to solve -- this is a real, UNVALIDATED HPC-scale resource commitment (free-sizing has")
    print("never been run in linopy at real scale before -- see module docstring).")
    print("!" * 78)
    raise SystemExit(0)

print("\n" + "=" * 78)
print(f"FREE-SIZING '{DESIGN_NAME}' (relaxed dispatch == MGPY_RELAX_UC=1 equivalent)")
print("=" * 78)
result = build_and_solve_free_sizing()
print("status:", result["status"], result["condition"], " NPC true:", result["npc_true"])
if result["status"] == "ok":
    res_units = result["RES_Units"].solution.values
    batt_units = result["Battery_Units"].solution.values
    gen_units = result["Generator_Units"].solution.values
    res_kw = res_units * inputs.RES_NOM_CAP_KW
    batt_kwh = batt_units * inputs.BATT_NOM_CAP_KWH
    gen_kw = gen_units * inputs.GEN_NOM_CAP_KW
    # FIXED (2026-08-28): the old print here labeled the RAW UNIT COUNT with a "(unit = X kW)"
    # annotation meaning "each unit equals X kW" -- easy to misread as "this number is already
    # in kW" (which is exactly what happened once, feeding a ~3x-too-large RES_KW pin into a
    # Pyomo MGPY_FIX_* verification and producing a spurious "infeasible" result that had
    # nothing to do with the actual design). Print BOTH forms explicitly labeled now.
    print(f"RES_Units  cumulative UNIT COUNT per step: {res_units}   (x{inputs.RES_NOM_CAP_KW} kW/unit)")
    print(f"RES_Units  cumulative KW        per step: {res_kw}")
    print(f"Battery_Units cumulative UNIT COUNT per step: {batt_units}   (x{inputs.BATT_NOM_CAP_KWH} kWh/unit)")
    print(f"Battery_Units cumulative KWH        per step: {batt_kwh}")
    print(f"Generator_Units cumulative UNIT COUNT per step: {gen_units}   (x{inputs.GEN_NOM_CAP_KW} kW/unit)")
    print(f"Generator_Units cumulative KW        per step: {gen_kw}")
    if RELAX_GEN_UNITS:
        gen_units_rounded = np.round(gen_units)
        gen_kw_rounded = gen_units_rounded * inputs.GEN_NOM_CAP_KW
        print("\nLINOPY_RELAX_GEN_UNITS=1 was active -- Generator_Units above is a CONTINUOUS")
        print("relaxation, not a certified integer answer. Rounded suggestion (nearest integer")
        print("per step, NOT yet re-verified):")
        print(f"Generator_Units ROUNDED cumulative UNIT COUNT per step: {gen_units_rounded}")
        print(f"Generator_Units ROUNDED cumulative KW        per step: {gen_kw_rounded}")
        print("Re-run via linopy_fixed_design_crosscheck.py (or the Pyomo fixed-design route)")
        print("with these ROUNDED per-step INCREMENTS pinned, to get the true certified cost --")
        print("do NOT report the continuous NPC above as the final answer.")
    print("\nFor an MGPY_FIX_* pin, use the 'cumulative KW/KWH' lines above (converted to")
    print("PER-STEP INCREMENTS, i.e. subtract consecutive cumulative values) -- NOT the raw")
    print("unit counts. MGPY_FIX_RES_KW/BATTERY_KWH/GEN_KW all want physical kW/kWh, matching")
    print("Results.py's own Size sheet convention.")
    print("\nCompare this design and NPC against Pyomo's own free-sizing run for the same design")
    print("(checklist §4) -- if they agree closely (design AND NPC), that's the confirmation")
    print("needed to trust linopy free-sizing solo on genuinely new points later.")
