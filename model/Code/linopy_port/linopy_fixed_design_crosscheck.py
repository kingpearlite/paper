"""GENERALIZED fixed-design relaxed cross-check -- reusable across every design in
gili_ketapang_checklist_dlensemble.md's sweep (9sc/6sc x capexp4/capexp5/capexp6, all
dieselops), not just the 9sc/capexp5 case this was validated against.

WHAT THIS IS: the same validated logic as
linopy_core_model_partial_load_fixed_design_real_scale_relaxed_only.py (real-scale relaxed
fixed-design solve + battery-overlap check + generator-integrality check), generalized so a
new design only needs a new DESIGN entry below, not a new copy of the whole file. Confirmed
against gili_ketapang_9sc_20y_MILP_capexp5_dieselops on two machines/OSes: linopy relaxed NPC
agreed with Pyomo's own relaxed NPC to 0.0031% ($9,755,652.69 vs $9,755,953.73), and both
independently found the same Generator_Full~0.5 fractional-relaxation artifact -- see that
file's docstring and Code/linopy_port/ session notes for the full validation history,
including the RES_Inverter_Efficiency bug this port had until 2026-08-28 (fixed here too).

WHY THIS EXISTS: gili_ketapang_checklist_dlensemble.md §5a-correction found that
Check_Simultaneous_Flows (battery-overlap) alone is NOT sufficient to trust a
MGPY_FIX_RELAX_BINARIES=1 fixed-design certification for this design family --
Generator_Full/Generator_Partial can be fractional even when the battery check passes clean,
silently making the reported NPC an unproven lower bound rather than a certified number. The
real codebase's Check_Generator_Commitment_Integrality() (added to Model_Resolution.py,
2026-08-27) now catches this on the Pyomo side too -- this file is the SECOND, independent
implementation, for exactly the same reason every other cross-check in this port exists: an
independently-built model agreeing on both the NPC and the same fractional-relaxation pattern
is much stronger evidence than either implementation's own self-report.

HOW TO USE FOR THE NEXT DESIGN IN THE QUEUE:
  1. Run that design's free-sizing solve in Pyomo (checklist §4), then its fixed-design
     verification with MGPY_FIX_RELAX_BINARIES=1 (checklist §5, updated item 1).
  2. Take the printed "FIXED-DESIGN VERIFICATION -- SIZING PINNED" per-step CUMULATIVE unit
     counts (RES_Units_milp / Battery_Units / Generator_Units) and Pyomo's reported relaxed
     objective from that log.
  3. Add a new entry to DESIGNS below (copy the CAPEXP5_9SC block, change PARAMS_DAT/DEMAND_CSV/
     RES_CSV/MAX_BATTERY_KWH per the checklist's own table, and the pinned/pyomo_npc values).
  4. Set DESIGN_NAME to that entry's key, set LINOPY_CONFIRM_REAL_SCALE=1, run.
  5. Compare against Pyomo's OWN Check_Generator_Commitment_Integrality output for the same
     run -- if both sides agree the generator check fails (or both agree it passes), that's
     strong joint confidence either way, same pattern as the capexp5 case.

+++ STILL DO NOT RUN WITHOUT EXPLICIT CONFIRMATION FIRST -- see LINOPY_CONFIRM_REAL_SCALE gate
below. Real-scale HPC-class resource commitment, same caveats as the file this generalizes. +++
"""
import os
import time

import xarray as xr

from linopy_core_model_builder import (
    DesignInputs, build_core_model, check_simultaneous_flows, check_fractional_commitment,
    resolve_plain_mode, start_resource_monitor, stop_resource_monitor, recover_memlimit_incumbent,
)

INPUTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "Code", "Inputs"))

# ---- one entry per checklist design. Add new ones here as their free-sizing + Pyomo
#      fixed-design step complete; do not create new copies of this file. ----
DESIGNS = {
    "capexp5_9sc": dict(
        params_dat="Parameters_9sc_20y_gili_ketapang_capexp5_MILP_dieselops.dat",
        demand_csv="Demand_9sc_20y_gili_ketapang.csv",
        res_csv="RES_Time_Series_9sc_gili_ketapang.csv",
        max_battery_kwh=62031.0,
        max_generator_kw=9400.0,
        # cumulative per-step unit counts, from Pyomo's own "SIZING PINNED" printout
        # (Combined_Output_20260827_150940_9sc_20y_capexp5_dieselops.log)
        pinned_res=[4748, 6975, 10120, 15407],
        pinned_battery=[10521, 15806, 23879, 36282],
        pinned_generator=[2, 2, 2, 4],
        pyomo_relaxed_npc=9_755_953.73,
        pyomo_battery_verdict="CERTIFIED FEASIBLE",
        pyomo_generator_verdict="NOT INTEGRAL (Full worst=0.5 in 8.5%, Partial worst=0.499969 in 2.6%)",
    ),
    # --- merged from the HPC repo (commit 72f4c0b, 2026-08-31), verbatim ---
    "capexp5_9sc_dlensemble_resized": dict(
        # 2026-08-31 -- follow-up to capexp5_9sc_dlensemble below: that entry pinned the ORIGINAL
        # capexp5_9sc_dieselops sizing (sized for Markov-chain+ERA5-proxy scenarios) against
        # DL-ensemble demand/PV and came back genuinely INFEASIBLE (DL-ensemble peak demand
        # 3,433.07 kW is ~12.9% above the original scenario set's 3,041.64 kW, exceeding what
        # that sizing can deliver at Lost_Load_Fraction=0). A free-sizing re-solve against
        # DL-ensemble data (LINOPY_TIMELIMIT=3000, MIPFocus=3, ScaleFlag=2, via
        # linopy_results_export.py, Code/Results/20260831_135729_capexp5_9sc_dlensemble_tl3000/)
        # found a genuinely larger design at gap 7.04% (time-limited, not proven optimal),
        # NPC 13,464.43 kUSD -- +37.6% over the original 9,785.13 kUSD. This entry pins THAT new
        # sizing and re-solves as a genuine non-relaxed MILP (LINOPY_FIXDESIGN_MODE=milp) to get
        # the true certified cost, since free-sizing's own 7.04% gap is too loose to trust as-is.
        params_dat="Parameters_9sc_20y_gili_ketapang_capexp5_MILP_dieselops.dat",
        demand_csv="Demand_9sc_20y_gili_ketapang_DLensemble.csv",
        res_csv="RES_Time_Series_9sc_gili_ketapang_DLensemble.csv",
        max_battery_kwh=62031.0, max_generator_kw=9400.0,
        pinned_res=[6831, 10112, 14954, 19896],
        pinned_battery=[15041, 22609, 34178, 51963],
        pinned_generator=[2, 2, 2, 4],
        pyomo_relaxed_npc=13_464_425.04,  # this design's own free-sizing NPC (time-limited,
        # gap 7.04% -- an approximate cross-check target only, not a certified number)
        pyomo_battery_verdict="not checked yet (no Pyomo run for this design)",
        pyomo_generator_verdict="not checked yet (no Pyomo run for this design)",
    ),
    "capexp5_9sc_dlensemble": dict(
        # 2026-08-31 -- DL-ensemble re-certification (gili_ketapang_checklist_dlensemble.md's own
        # original purpose, §5 steps 3-4): does capexp5_9sc_dieselops (the cheapest of all ten
        # certified dieselops designs, §11, 9,785.13 kUSD under the Markov-chain+ERA5-proxy
        # demand/PV scenarios) survive being re-evaluated against DL-ensemble-derived demand/PV
        # scenarios instead? SAME pinned sizing as capexp5_9sc above (design held fixed -- this
        # tests dispatch-cost sensitivity to the demand/PV scenario set, not a new sizing search),
        # only demand_csv/res_csv swapped to the DL-ensemble files (row/column shape confirmed
        # identical to the originals: both 8761 lines, both 181 columns -- safe drop-in swap).
        params_dat="Parameters_9sc_20y_gili_ketapang_capexp5_MILP_dieselops.dat",
        demand_csv="Demand_9sc_20y_gili_ketapang_DLensemble.csv",
        res_csv="RES_Time_Series_9sc_gili_ketapang_DLensemble.csv",
        max_battery_kwh=62031.0, max_generator_kw=9400.0,
        pinned_res=[4748, 6975, 10120, 15407],
        pinned_battery=[10521, 15806, 23879, 36282],
        pinned_generator=[2, 2, 2, 4],
        pyomo_relaxed_npc=9_785_130.44,  # the Markov-chain+ERA5-proxy TRUE integral NPC (§5a-final)
        # -- comparison target for this DL-ensemble re-certification, NOT a Pyomo run for this
        # specific (DL-ensemble) scenario set, which doesn't exist.
        pyomo_battery_verdict="N/A -- comparison baseline is this same design's own certified NPC under the original demand/PV scenarios, not a separate Pyomo run",
        pyomo_generator_verdict="N/A -- see battery verdict",
    ),
    # --- end of merged block ---
    "capexp5_6sc_PYOMO_DESIGN": dict(
        # DIAGNOSTIC ENTRY (2026-08-28): pins PYOMO'S OWN free-sizing design (not linopy's) for
        # capexp5_6sc, to isolate whether linopy's free-sizing/cost formula agrees with Pyomo
        # when given the SAME design -- linopy's own free-sizing run found a wildly different,
        # much cheaper design ($4.43M vs Pyomo's $10.75M) and it's not yet known whether that's
        # a genuine better optimum or a bug. If linopy's fixed-design NPC for THIS (Pyomo's)
        # design comes back close to Pyomo's own $10,749,760, the cost formula is fine and the
        # free-sizing divergence is a real solution-quality gap, not a bug. If it comes back
        # very different, that points at a genuine bug independent of the sizing search.
        params_dat="Parameters_6sc_20y_gili_ketapang_capexp5_MILP_dieselops.dat",
        demand_csv="Demand_6sc_20y_gili_ketapang.csv",
        res_csv="RES_Time_Series_6sc_gili_ketapang.csv",
        max_battery_kwh=74675.0,   # 6sc uses a different ceiling than 9sc's 62031 -- checklist §4 flags this explicitly
        max_generator_kw=9400.0,
        # cumulative unit counts converted from Pyomo's own reported kW/kWh Size table
        # (increments PV 1568.49;720.06;1461.24;1348.05, Battery 10619;5977;9818;16345,
        # Genset 940;0;470;940 -- converted via /RES_Nominal_Capacity(0.33) and
        # /Generator_Nominal_Capacity_milp(470), Battery unit=1kWh so no conversion needed;
        # all divisions came out exact, confirming the increments-to-units conversion is clean)
        pinned_res=[4753, 6935, 11363, 15448],
        pinned_battery=[10619, 16596, 26414, 42759],
        pinned_generator=[2, 2, 3, 5],
        pyomo_relaxed_npc=10_749_760.0,
        pyomo_battery_verdict="not checked yet (Pyomo's own certificate for this design not run)",
        pyomo_generator_verdict="not checked yet",
    ),
    # --- merged from the HPC repo (commits 10cf3c1/72f4c0b, 2026-08-31), verbatim ---
    "capexp4_9sc": dict(
        # 2026-08-31 -- ScaleFlag diagnostic entry. Pyomo's own MGPY_FIX_RELAX_BINARIES=1
        # fixed-design run for this design (Combined_Output_20260827_141305_9sc_20y_capexp4_
        # dieselops.log) never finished: barrier+crossover converged cleanly in ~390s to
        # ~9.7799e6, but the post-crossover simplex cleanup ground through 8.5M+ iterations
        # over 1900+s with DualInf shrinking only slowly (5.8e4 -> 1.05e3) before the run died
        # (VS Code disconnect, not a solver error) -- classic degenerate-LP signature. Gurobi's
        # own coefficient statistics for that run: Matrix range [2e-03,1e+04] (ratio ~5e6,
        # above the ideal <1e6 guideline) and RHS range [2e-01,1e+07] (ratio ~5e7, well above
        # ideal). This entry exists to test MGPY_SCALEFLAG's linopy equivalent (LINOPY_SCALEFLAG,
        # see SOLVER_KWARGS below) as a cheap, zero-risk lever on the SAME LP structure via a
        # different (faster, no .lp-file-writing overhead) engine.
        params_dat="Parameters_9sc_20y_gili_ketapang_capexp4_MILP_dieselops.dat",
        demand_csv="Demand_9sc_20y_gili_ketapang.csv",
        res_csv="RES_Time_Series_9sc_gili_ketapang.csv",
        max_battery_kwh=62031.0,
        max_generator_kw=9400.0,
        # cumulative per-step unit counts, from Pyomo's own "FIXED-DESIGN VERIFICATION --
        # SIZING PINNED" printout (Combined_Output_20260827_141305_9sc_20y_capexp4_dieselops.log,
        # lines 42-61), themselves pinned from the successful free-sizing run
        # (Combined_Output_20260827_132744_..., optimal in 924s, gap 0.5966%, MIPGap target 3%).
        pinned_res=[4624, 6316, 8479, 12165, 16069],
        pinned_battery=[10080, 13945, 19362, 26983, 37740],
        pinned_generator=[2, 2, 2, 3, 4],
        pyomo_relaxed_npc=9_798_000.0,  # UNCONFIRMED -- Pyomo's fixed-design run died mid-simplex
        # before converging; last observed objective was 9.7798973e6 and still (slowly)
        # improving/cleaning up dual infeasibility when it died. Not a certified number -- use
        # only as an approximate cross-check target, not a pass/fail threshold.
        pyomo_battery_verdict="not checked yet (Pyomo's own run died before completing -- see docstring)",
        pyomo_generator_verdict="not checked yet (Pyomo's own run died before completing -- see docstring)",
    ),
    # 2026-08-31 -- the four remaining dieselops step-duration points (checklist_dlensemble.md
    # §1's table), certified via linopy only (no Pyomo attempt for any of these four -- Pyomo's
    # own path has failed on every design at this scale it's been tried on so far, see capexp4_9sc/
    # capexp5_9sc's own notes). Pinned sizing taken verbatim from each design's own linopy
    # free-sizing run this session (LINOPY_SCALEFLAG=2, LINOPY_MIPFOCUS=1,
    # logs/capexp*_freesizing_scaleflag2_mipfocus1.log's own "cumulative UNIT COUNT per step"
    # printout). Suffixed `_dieselops` to avoid colliding with the PLAIN_MODE entries of the same
    # base name already in this dict (capexp6_9sc/capexp6_6sc below are the non-dieselops Table
    # 6/7 sweep points -- different design family, same .dat naming root).
    "capexp4_6sc_dieselops": dict(
        # 5-step (Step_Duration=4), 6sc. Free-sizing: OPTIMAL, gap 0.0374% (the only one of the
        # four that converged cleanly within the 5h TimeLimit).
        params_dat="Parameters_6sc_20y_gili_ketapang_capexp4_MILP_dieselops.dat",
        demand_csv="Demand_6sc_20y_gili_ketapang.csv",
        res_csv="RES_Time_Series_6sc_gili_ketapang.csv",
        max_battery_kwh=74675.0, max_generator_kw=9400.0,
        pinned_res=[4577, 6313, 8631, 10852, 13841],
        pinned_battery=[10134, 14439, 20818, 30357, 44736],
        pinned_generator=[2, 2, 2, 3, 5],
        pyomo_relaxed_npc=10_699_814.55,  # actually linopy free-sizing NPC (no Pyomo run exists)
        pyomo_battery_verdict="not checked yet (no Pyomo run for this design)",
        pyomo_generator_verdict="not checked yet (no Pyomo run for this design)",
    ),
    "capexp6_6sc_dieselops": dict(
        # 3-step-uneven (Step_Duration=6), 6sc. Free-sizing: TIME LIMIT, gap 8.93% (unproven,
        # but only free-sizing candidate available -- fixed-design MILP mode is what actually
        # certifies this design's true cost, independent of the free-sizing gap).
        params_dat="Parameters_6sc_20y_gili_ketapang_capexp6_MILP_dieselops.dat",
        demand_csv="Demand_6sc_20y_gili_ketapang.csv",
        res_csv="RES_Time_Series_6sc_gili_ketapang.csv",
        max_battery_kwh=74675.0, max_generator_kw=9400.0,
        pinned_res=[4826, 7895, 11455, 14812],
        pinned_battery=[11137, 19124, 33691, 49112],
        pinned_generator=[2, 2, 4, 5],
        pyomo_relaxed_npc=10_939_933.30,
        pyomo_battery_verdict="not checked yet (no Pyomo run for this design)",
        pyomo_generator_verdict="not checked yet (no Pyomo run for this design)",
    ),
    "capexp5_6sc_dieselops": dict(
        # 4-step (Step_Duration=5), 6sc. Free-sizing: TIME LIMIT, gap 9.26%, but NPC
        # (10,747,742.73) essentially matches this design's own earlier PROVEN-optimal free-sizing
        # result from before ScaleFlag/MIPFocus=1 were tried (10,745,495.83, +0.02%) -- the design
        # itself is trustworthy even though this particular run's own gap didn't close.
        params_dat="Parameters_6sc_20y_gili_ketapang_capexp5_MILP_dieselops.dat",
        demand_csv="Demand_6sc_20y_gili_ketapang.csv",
        res_csv="RES_Time_Series_6sc_gili_ketapang.csv",
        max_battery_kwh=74675.0, max_generator_kw=9400.0,
        pinned_res=[4753, 7107, 9811, 16810],
        pinned_battery=[10619, 16596, 26414, 42758],
        pinned_generator=[2, 2, 3, 5],
        pyomo_relaxed_npc=10_747_742.73,
        pyomo_battery_verdict="not checked yet (no Pyomo run for this design)",
        pyomo_generator_verdict="not checked yet (no Pyomo run for this design)",
    ),
    "capexp6_9sc_dieselops": dict(
        # 3-step-uneven (Step_Duration=6), 9sc. Free-sizing: TIME LIMIT, gap 9.75%, consistent
        # with two earlier time-limited attempts on this same design (~9,934,470, before
        # ScaleFlag/MIPFocus=1) -- this run's 9,930,258.30 is -0.04% vs. those, not a regression.
        params_dat="Parameters_9sc_20y_gili_ketapang_capexp6_MILP_dieselops.dat",
        demand_csv="Demand_9sc_20y_gili_ketapang.csv",
        res_csv="RES_Time_Series_9sc_gili_ketapang.csv",
        max_battery_kwh=62031.0, max_generator_kw=9400.0,
        pinned_res=[4864, 7826, 11862, 15056],
        pinned_battery=[10990, 17935, 29511, 40915],
        pinned_generator=[2, 2, 3, 4],
        pyomo_relaxed_npc=9_930_258.30,
        pyomo_battery_verdict="not checked yet (no Pyomo run for this design)",
        pyomo_generator_verdict="not checked yet (no Pyomo run for this design)",
    ),
    # --- end of merged block ---
    # 2026-08-30 -- four NEW plain-model (non-dieselops) points for the paper's Table 6/7
    # step-duration trend, requiring PLAIN_MODE=1 (see flag below) since this file otherwise
    # always builds the full dieselops unit-commitment formulation regardless of which .dat is
    # pointed at. pinned_res/battery/generator are cumulative per-step UNIT COUNTS (not kW),
    # taken verbatim from each design's own linopy free-sizing "cumulative UNIT COUNT per step"
    # printout (Code/linopy_port/logs/capexp{6,2}_{9,6}sc_plain_reproduce.log) -- no kW/unit
    # conversion needed here, unlike the Pyomo MGPY_FIX_* kW-increment convention. NOTE:
    # `pyomo_relaxed_npc` below is repurposed to hold the LINOPY free-sizing NPC, not a Pyomo one
    # -- no Pyomo free-sizing run exists for these four plain-model points (only Pyomo
    # fixed-design certification runs, launched separately this session).
    "capexp6_9sc": dict(
        params_dat="Parameters_9sc_20y_gili_ketapang_capexp6_MILP.dat",
        demand_csv="Demand_9sc_20y_gili_ketapang.csv",
        res_csv="RES_Time_Series_9sc_gili_ketapang.csv",
        max_battery_kwh=62031.0, max_generator_kw=9400.0,
        pinned_res=[5476, 9193, 14985, 20762],
        pinned_battery=[10990, 17935, 29511, 40915],
        pinned_generator=[94, 128, 248, 280],
        pyomo_relaxed_npc=9_861_867.27,  # actually linopy free-sizing NPC, see note above
        pyomo_battery_verdict="not checked yet (plain model, no dieselops Single_Flow_BESS nobinary check applicable)",
        pyomo_generator_verdict="N/A -- plain model, no Generator_Partial/Full commitment to check",
    ),
    "capexp2_9sc": dict(
        params_dat="Parameters_9sc_20y_gili_ketapang_capexp2_MILP.dat",
        demand_csv="Demand_9sc_20y_gili_ketapang.csv",
        res_csv="RES_Time_Series_9sc_gili_ketapang.csv",
        max_battery_kwh=62031.0, max_generator_kw=9400.0,
        pinned_res=[4708, 5514, 6486, 7658, 9000, 11050, 12994, 15218, 17831, 20763],
        pinned_battery=[9268, 10891, 12810, 15080, 17769, 20956, 24738, 29228, 34566, 40915],
        pinned_generator=[94, 94, 94, 94, 94, 111, 140, 177, 222, 280],
        pyomo_relaxed_npc=9_622_384.54,  # actually linopy free-sizing NPC, see note above
        pyomo_battery_verdict="not checked yet", pyomo_generator_verdict="N/A -- plain model",
    ),
    "capexp6_6sc": dict(
        params_dat="Parameters_6sc_20y_gili_ketapang_capexp6_MILP.dat",
        demand_csv="Demand_6sc_20y_gili_ketapang.csv",
        res_csv="RES_Time_Series_6sc_gili_ketapang.csv",
        max_battery_kwh=74675.0, max_generator_kw=9400.0,  # 6sc uses a different battery ceiling than 9sc's 62031 -- checklist §4 flags this explicitly
        pinned_res=[5451, 9746, 18084, 30341],
        pinned_battery=[11137, 19124, 33691, 49112],
        pinned_generator=[94, 158, 325, 351],
        pyomo_relaxed_npc=10_974_488.97,  # actually linopy free-sizing NPC, see note above
        pyomo_battery_verdict="not checked yet", pyomo_generator_verdict="N/A -- plain model",
    ),
    "capexp2_6sc": dict(
        params_dat="Parameters_6sc_20y_gili_ketapang_capexp2_MILP.dat",
        demand_csv="Demand_6sc_20y_gili_ketapang.csv",
        res_csv="RES_Time_Series_6sc_gili_ketapang.csv",
        max_battery_kwh=74675.0, max_generator_kw=9400.0,
        pinned_res=[4673, 5547, 6687, 8137, 10346, 12837, 15961, 19667, 24405, 30342],
        pinned_battery=[9255, 11013, 13144, 15734, 18891, 22746, 27464, 33250, 40360, 49112],
        pinned_generator=[94, 94, 94, 94, 104, 133, 170, 217, 276, 351],
        pyomo_relaxed_npc=10_663_998.86,  # actually linopy free-sizing NPC, see note above
        pyomo_battery_verdict="not checked yet", pyomo_generator_verdict="N/A -- plain model",
    ),
    "smoketest_1sc": dict(
        # LOCAL SMOKE TEST ONLY (Stage 0 validation, full-parity roadmap plan) -- pins the
        # sizing linopy_free_sizing_crosscheck.py's own post-refactor run just found for this
        # same design (S=1/Years=10/single-step, real Gili Ketapang data, small enough to solve
        # on a laptop). No Pyomo ground truth exists for this design -- this only checks that
        # the refactored fixed-design pipeline runs end-to-end and lands in the same ballpark as
        # free-sizing's own NPC, not an exact-match certification.
        params_dat="Parameters_1sc_10y_gili_ketapang_scpos2_MILP.dat",
        demand_csv="Demand_1sc_10y_gili_ketapang_scpos2.csv",
        res_csv="RES_Time_Series_1sc_gili_ketapang_scpos2.csv",
        max_battery_kwh=25000.0, max_generator_kw=4000.0,
        pinned_res=[6584], pinned_battery=[12021], pinned_generator=[94],
        pyomo_relaxed_npc=None, pyomo_battery_verdict="N/A", pyomo_generator_verdict="N/A",
    ),
    "smoketest_6sc_2y_dieselops": dict(
        # LOCAL REGRESSION FIXTURE for the integral fixed-design path (2026-09-11) -- the only
        # small (S=6/Years=2, laptop-solvable) design with Generator_Partial_Load=1, so the only
        # one where the generator-commitment check and the integral re-solve are meaningful.
        # Pinned sizing = linopy_free_sizing_crosscheck.py's own LINOPY_DESIGN=
        # smoketest_6sc_2y_dieselops result the same day (NPC $2,800,611.88; cumulative UNIT
        # COUNTS RES 4214 x0.33kW, Battery 9255 x1kWh, Generator 2 x470kW). No Pyomo value yet.
        params_dat="Parameters_6sc_2y_gili_ketapang_MILP_dieselops.dat",
        demand_csv="Demand_6sc_2y_gili_ketapang_dieselops.csv",
        res_csv="RES_Time_Series_6sc_gili_ketapang.csv",
        max_battery_kwh=74675.0, max_generator_kw=9400.0,
        pinned_res=[4214], pinned_battery=[9255], pinned_generator=[2],
        pyomo_relaxed_npc=None, pyomo_battery_verdict="N/A", pyomo_generator_verdict="N/A",
    ),
    # smoketest_1sc_greenfieldtest (Stage 7 functional test, full-parity roadmap): confirmed
    # 2026-09-04 that removing the existing-capacity credit correctly raises NPC -- same pinned
    # sizing as smoketest_1sc but Greenfield_Investment=1 (vs. Brownfield's real 470kW existing
    # generator credit) moved the fixed-design relaxed NPC from $5,572,186.85 to $5,625,509.22
    # (+$53,322.37) -- correct direction, plausible magnitude relative to the pure
    # investment-credit component alone (470kW x $160.58/kW = ~$75,473, partially offset by the
    # salvage-value credit also disappearing). Entry and its throwaway .dat copy removed after
    # confirming.
    # smoketest_1sc_fuelesctest (Stage 3 functional test, full-parity roadmap): confirmed
    # 2026-09-04 that time-varying fuel cost correctly raises the NPC -- same pinned sizing as
    # smoketest_1sc but Fuel_Specific_Cost_Calculation=1 with a 5%/yr escalation rate moved the
    # fixed-design relaxed NPC from $5,572,186.85 (flat) to $5,618,946.24 (escalating), the
    # expected direction and a plausible magnitude given this design's generator share is only
    # ~11% of energy production. Entry and its throwaway .dat copy removed after confirming.
}

# PLAIN_MODE (2026-09-11): now derived from the .dat's own Generator_Partial_Load flag right
# after DesignInputs loads below (resolve_plain_mode() in linopy_core_model_builder.py) -- it
# used to come only from the LINOPY_PLAIN env var, which defaulted to the partial-load
# formulation even for Generator_Partial_Load=0 designs. LINOPY_PLAIN is still accepted if it
# agrees with the .dat.

# LINOPY_INTEGRAL (2026-09-11) -- the integral fixed-design re-solve: sizing pinned, and EVERY
# discrete variable at its real domain (Single_Flow_BESS binary, Generator_Partial binary,
# Generator_Full integer). This is the certified number whenever the relaxed solve fails either
# check. Same method as the archived linopy_core_model_partial_load_fixed_design_real_scale.py
# STEP 2b, which produced capexp5_9sc's integral NPC $9,785,130.44 (2026-08-28) after Pyomo's own
# integral attempt died in presolve.
#   auto (default) -- re-solve only if the relaxed solve fails the battery or generator check
#   1              -- always re-solve (e.g. to measure the relaxed-vs-integral gap)
#   0              -- never (the pre-2026-09-11 behaviour: relaxed only)
# LINOPY_FIXDESIGN_MODE (the HPC repo's own 2026-08-31 switch for the same thing, merged
# 2026-09-11) is accepted as an alias so commands written on HPC keep working:
#   LINOPY_FIXDESIGN_MODE=milp    == LINOPY_INTEGRAL=1  (always run the integral solve)
#   LINOPY_FIXDESIGN_MODE=relaxed == LINOPY_INTEGRAL=0  (relaxed only)
# Setting both to contradicting values is an error rather than a silent guess.
_fixdesign_mode = os.environ.get("LINOPY_FIXDESIGN_MODE", "").strip().lower()
if _fixdesign_mode:
    if _fixdesign_mode not in ("relaxed", "milp"):
        raise ValueError(f"LINOPY_FIXDESIGN_MODE={_fixdesign_mode!r} -- must be 'relaxed' or 'milp'")
    _alias = "1" if _fixdesign_mode == "milp" else "0"
    if os.environ.get("LINOPY_INTEGRAL") and os.environ["LINOPY_INTEGRAL"].strip().lower() != _alias:
        raise ValueError(f"LINOPY_FIXDESIGN_MODE={_fixdesign_mode} contradicts "
                         f"LINOPY_INTEGRAL={os.environ['LINOPY_INTEGRAL']} -- set only one of them.")
    os.environ["LINOPY_INTEGRAL"] = _alias
LINOPY_INTEGRAL = os.environ.get("LINOPY_INTEGRAL", "auto").strip().lower()
if LINOPY_INTEGRAL not in ("auto", "0", "1"):
    raise ValueError(f"LINOPY_INTEGRAL={LINOPY_INTEGRAL!r} -- must be 'auto', '0' or '1'")
# 4h default time cap, as in the archived file (a real MILP over millions of discrete variables at
# 20y/9sc scale -- capexp5_9sc took ~83 min on HPC). On hitting it, the incumbent is still a valid
# upper bound on the true cost, but the design is NOT certified.
INTEGRAL_TIMELIMIT = float(os.environ.get("LINOPY_INTEGRAL_TIMELIMIT", 4 * 3600))
# Tighter than the free-sizing recipe's 0.03: this is the certification number itself, and the
# dispatch-only MILP closes much more easily than free sizing (capexp5_9sc stopped at 0.294%).
INTEGRAL_MIPGAP = float(os.environ.get("LINOPY_INTEGRAL_MIPGAP", 0.005))
# LINOPY_GEN_PARALLEL (2026-09-18) -- integral re-solve only: 1 lets the installed units run in
# parallel sharing the load (build_core_model(gen_parallel=True), see its docstring), instead of
# the default at-most-one-unit-below-full commitment, which cannot produce 470-587.5 kW with the
# Gili Ketapang units. The relaxed solve is identical either way. Default 0 keeps every logged
# certification reproducible; results folders of a parallel run are tagged "_parallel".
_gen_parallel_raw = os.environ.get("LINOPY_GEN_PARALLEL", "0").strip()
if _gen_parallel_raw not in ("0", "1"):
    raise ValueError(f"LINOPY_GEN_PARALLEL={_gen_parallel_raw!r} -- must be '0' or '1'")
GEN_PARALLEL = _gen_parallel_raw == "1"

# LINOPY_SOLVER (2026-09-05) -- see linopy_free_sizing_crosscheck.py's own comment on this same
# name for the full explanation. Default "gurobi" preserves every existing regression baseline;
# SOLVER_KWARGS below is a Gurobi-tuned recipe, discarded (not translated) when switching away.
LINOPY_SOLVER = os.environ.get("LINOPY_SOLVER", "gurobi").strip().lower()

# LINOPY_SITE (2026-09-08) -- config-driven alternative to LINOPY_DESIGN/DESIGNS, see
# site_config.py's docstring and linopy_free_sizing_crosscheck.py's own identical branch. NOTE:
# a site loaded this way has no pinned_res/pinned_battery/pinned_generator/pyomo_relaxed_npc --
# those are diagnostic overlays from a specific already-completed Pyomo cross-check, not a site
# property, so they must still be passed separately (see the "pinned" construction below).
_LINOPY_SITE = os.environ.get("LINOPY_SITE")
if _LINOPY_SITE:
    if os.environ.get("LINOPY_DESIGN"):
        raise ValueError("Set only one of LINOPY_SITE or LINOPY_DESIGN, not both.")
    import site_config
    DESIGN_NAME = _LINOPY_SITE
    D = site_config.load_site_design(_LINOPY_SITE)
    D.setdefault("pinned_res", None)
    D.setdefault("pinned_battery", None)
    D.setdefault("pinned_generator", None)
    D.setdefault("pyomo_relaxed_npc", None)
else:
    DESIGN_NAME = os.environ.get("LINOPY_DESIGN", "capexp5_9sc")
    D = DESIGNS[DESIGN_NAME]

RES_CSV = os.path.join(INPUTS_DIR, D["res_csv"])
DEMAND_CSV = os.path.join(INPUTS_DIR, D["demand_csv"])
PARAMS_DAT = os.path.join(INPUTS_DIR, D["params_dat"])


# ---- Scenarios/Years/Periods/Step_Duration/RES-CF-floor/all cost & tech parameters are now
#      loaded by the shared DesignInputs (linopy_core_model_builder.py, Stage 0 of the
#      full-parity roadmap) -- this is what makes the file reusable across capexp4
#      (Step_Duration=4)/capexp5 (=5)/capexp6 (=6, uneven final step) and 9sc/6sc without any
#      code changes, only a new DESIGNS entry. ceil(yt/STEP_DURATION) handles an uneven final
#      step generically (capexp6: 20 years / 6-year steps -> steps at yt=1-6,7-12,13-18,19-20 --
#      the last step is short, matching the checklist's own description of this case).
inputs = DesignInputs(
    params_dat=PARAMS_DAT, demand_csv=DEMAND_CSV, res_csv=RES_CSV,
    max_battery_kwh=D["max_battery_kwh"], max_generator_kw=D["max_generator_kw"],
)
S, YEARS, PERIODS, STEP_DURATION, STEPS_NUMBER = inputs.S, inputs.YEARS, inputs.PERIODS, inputs.STEP_DURATION, inputs.STEPS_NUMBER
ut_idx = inputs.ut_idx
print(f"Design '{DESIGN_NAME}': S={S}, YEARS={YEARS}, PERIODS={PERIODS}, STEP_DURATION={STEP_DURATION}, STEPS_NUMBER={STEPS_NUMBER}")
PLAIN_MODE = resolve_plain_mode(inputs)
print(f"Generator_Partial_Load={inputs.GENERATOR_PARTIAL_LOAD} -> "
      f"{'plain dispatch' if PLAIN_MODE else 'partial-load unit commitment'} formulation")

if _LINOPY_SITE and D["pinned_res"] is None:
    # A site's .conf has no notion of "pinned sizing" -- that is the result of one specific
    # Pyomo free-sizing run, not a site property (see the LINOPY_SITE branch's own comment
    # above). Supply it via these three comma-separated env vars (per-step CUMULATIVE UNIT
    # COUNTS -- not physical kW/kWh -- same convention as every DESIGNS["pinned_*"] entry, i.e.
    # the "cumulative UNIT COUNT per step" line linopy_free_sizing_crosscheck.py prints, NOT its
    # "cumulative KW/KWH" line).
    missing = [v for v in ("LINOPY_PINNED_RES", "LINOPY_PINNED_BATTERY", "LINOPY_PINNED_GENERATOR")
               if not os.environ.get(v)]
    if missing:
        raise ValueError(
            f"LINOPY_SITE={_LINOPY_SITE!r} has no pinned sizing -- set {', '.join(missing)} "
            f"(comma-separated, one cumulative value per investment step) to pin a design for "
            f"fixed-design verification.")
    D["pinned_res"] = [float(x) for x in os.environ["LINOPY_PINNED_RES"].split(",")]
    D["pinned_battery"] = [float(x) for x in os.environ["LINOPY_PINNED_BATTERY"].split(",")]
    D["pinned_generator"] = [float(x) for x in os.environ["LINOPY_PINNED_GENERATOR"].split(",")]

assert len(D["pinned_res"]) == STEPS_NUMBER, f"pinned_res has {len(D['pinned_res'])} entries, expected {STEPS_NUMBER} steps -- check DESIGNS entry"
assert len(D["pinned_battery"]) == STEPS_NUMBER
assert len(D["pinned_generator"]) == STEPS_NUMBER
# Same recipe as the validated capexp5/9sc file, tuned for a 24cpu/250GB interactive session --
# re-check against whatever allocation is actually current before running a new design.
SOLVER_KWARGS = dict(
    OutputFlag=1,
    Threads=int(os.environ.get("SLURM_CPUS_PER_TASK", 24)),
    Method=2, BarHomogeneous=1,
    MIPFocus=3, Cuts=1, CutPasses=1, Crossover=1,
    BarConvTol=1e-3, OptimalityTol=1e-3, FeasibilityTol=1e-4,
    MIPGap=0.03, NodefileStart=0.1,
    MemLimit=220,
)

# LINOPY_SCALEFLAG / LINOPY_MIPFOCUS / LINOPY_MEMLIMIT (2026-08-31, merged from the HPC repo) --
# LINOPY_SCALEFLAG mirrors Pyomo's MGPY_SCALEFLAG (Model_Resolution.py):
#   unset = automatic (Gurobi default, -1)
#   0 = none, 1 = equilibrium, 2 = geometric mean, 3 = aggressive
# Zero-risk lever (no data/results change, only Gurobi's internal matrix scaling) being tested
# against the capexp4_9sc entry's badly-scaled LP (Matrix range ratio ~5e6, RHS range ratio
# ~5e7 in Pyomo's own coefficient statistics for this exact design -- see that entry's comment).
if os.environ.get("LINOPY_SCALEFLAG"):
    SOLVER_KWARGS["ScaleFlag"] = int(os.environ["LINOPY_SCALEFLAG"])
if os.environ.get("LINOPY_MIPFOCUS"):
    SOLVER_KWARGS["MIPFocus"] = int(os.environ["LINOPY_MIPFOCUS"])
if os.environ.get("LINOPY_MEMLIMIT"):
    SOLVER_KWARGS["MemLimit"] = int(os.environ["LINOPY_MEMLIMIT"])
if os.environ.get("LINOPY_SOFTMEMLIMIT"):
    # Same knobs as the free-sizing script (2026-09-12): SoftMemLimit stops the solve and
    # keeps the incumbent instead of raising "Out of memory"; the node file moves the
    # branch-and-bound tree to disk; fewer threads cut per-thread memory.
    SOLVER_KWARGS["SoftMemLimit"] = float(os.environ["LINOPY_SOFTMEMLIMIT"])
if os.environ.get("LINOPY_NODEFILESTART"):
    SOLVER_KWARGS["NodefileStart"] = float(os.environ["LINOPY_NODEFILESTART"])
if os.environ.get("LINOPY_NODEFILEDIR"):
    SOLVER_KWARGS["NodefileDir"] = os.environ["LINOPY_NODEFILEDIR"]
if os.environ.get("LINOPY_THREADS"):
    SOLVER_KWARGS["Threads"] = int(os.environ["LINOPY_THREADS"])
if os.environ.get("LINOPY_DUALREDUCTIONS") == "0":
    # Diagnosis only (2026-09-13): Gurobi's default dual reductions make it report
    # "infeasible_or_unbounded" without saying which. Setting DualReductions=0 costs
    # presolve strength but returns a definite INFEASIBLE or UNBOUNDED -- the difference
    # between a design that genuinely cannot be operated and a numerical artefact.
    SOLVER_KWARGS["DualReductions"] = 0


def _condition_name(condition):
    """linopy's termination condition as a plain string ('optimal', 'time_limit', ...)."""
    return str(getattr(condition, "value", condition))


def build_and_solve(pinned, mode="relaxed"):
    """mode='relaxed' (Stage 0, full-parity roadmap): bess_mode='relaxed' reproduces this file's
    historical Single_Flow_BESS treatment (continuous [0,1], both linking constraints active --
    the MGPY_FIX_RELAX_BINARIES=1 equivalent), with Generator_Partial/Full continuous too.
    Validated (Stage 0 step 3 of the plan) to reproduce capexp5_9sc's already-certified NPC.

    mode='integral' (2026-09-11): bess_mode='binary' + uc_integral=True -- every discrete
    variable at its real domain, sizing still pinned, i.e. a dispatch-only true MILP. Its NPC is
    the real cost of the pinned design (within INTEGRAL_MIPGAP), not a lower bound.

    salvage_floor=True and battery_min_capacity=True in both modes (see
    linopy_core_model_builder.py's module docstring)."""
    integral = mode == "integral"
    # Resource trace (2026-09-18): the same once-a-minute RSS/CPU monitor free sizing prints, over
    # build and solve (Gurobi runs in this process, so RSS includes the solver), plus phase times.
    t_start = time.time()
    start_resource_monitor()
    try:
        built = build_core_model(inputs, pinned=pinned, bess_mode="binary" if integral else "relaxed",
                                  plain_mode=PLAIN_MODE, uc_integral=integral,
                                  salvage_floor=True, battery_min_capacity=True,
                                  gen_parallel=GEN_PARALLEL and integral)
        t_built = time.time()
        return _solve_built(built, integral, mode, t_start, t_built)
    finally:
        stop_resource_monitor()


def _solve_built(built, integral, mode, t_start, t_built):
    m = built["m"]
    if LINOPY_SOLVER == "gurobi":
        solve_kwargs = dict(SOLVER_KWARGS)
        if integral:
            solve_kwargs["TimeLimit"] = INTEGRAL_TIMELIMIT
            solve_kwargs["MIPGap"] = INTEGRAL_MIPGAP
    elif LINOPY_SOLVER == "highs" and integral:
        solve_kwargs = dict(time_limit=INTEGRAL_TIMELIMIT,
                            mip_rel_gap=float(os.environ.get("LINOPY_MIPGAP", 1e-4)))
    elif LINOPY_SOLVER == "highs":
        # 2026-09-05: see linopy_free_sizing_crosscheck.py's own comment on this same branch for
        # the full real-test evidence -- HiGHS reported a falsely-confident "optimal" at the
        # Gurobi-tuned MIPGap=0.03 while actually 2.77% wrong with a materially different
        # technology mix, and tightening the gap alone did not fix it (hit the time limit stuck
        # on the identical wrong incumbent). Default gap kept tight here for the same reason:
        # an honest `time_limit` status beats a false `optimal` one. HiGHS's own option names
        # (not Gurobi's) are used, confirmed working by that testing.
        solve_kwargs = dict(
            time_limit=float(os.environ.get("LINOPY_TIMELIMIT", 1800)),
            mip_rel_gap=float(os.environ.get("LINOPY_MIPGAP", 1e-4)),
        )
    else:
        # SOLVER_KWARGS is a Gurobi-tuned recipe -- see LINOPY_SOLVER's own comment above. No
        # tuned recipe (or even a time-limit default) exists yet for any backend besides
        # gurobi/highs -- both confirmed installed and working in this environment.
        solve_kwargs = {}
        print(f"LINOPY_SOLVER={LINOPY_SOLVER!r} -- using that backend's own defaults, "
              f"none of this project's Gurobi-tuned solver options apply, and no time limit is "
              f"set. Confirm this solver is actually installed before trusting this to return.")
    status, condition = m.solve(solver_name=LINOPY_SOLVER, **solve_kwargs)
    if integral and status != "ok" and recover_memlimit_incumbent(m):
        print("Gurobi stopped on its memory limit (status 17) with an incumbent: recovered it; "
              "it is an UPPER bound on the design's cost, not a certified value.")
        status, condition = "ok", "memory_limit"
    t_solved = time.time()
    print(f"[timing] {mode}: build {t_built - t_start:.0f} s, solve {t_solved - t_built:.0f} s, "
          f"total {t_solved - t_start:.0f} s")
    npc_true = m.objective.value + built["total_const_offset"] if status == "ok" else None
    # Best bound (integral mode, Gurobi only): with the incumbent it brackets the true cost when
    # the solve stops on the time limit. linopy keeps the gurobipy model as m.solver_model.
    best_bound = None
    solver_model = getattr(m, "solver_model", None)
    if integral and status == "ok" and solver_model is not None:
        try:
            best_bound = float(solver_model.ObjBound) + built["total_const_offset"]
        except Exception:  # noqa: BLE001 -- purely informational; never block the result on it
            best_bound = None
    # Return the FULL build_core_model() dict (not a hand-picked subset) -- this is what makes
    # this solve's result usable by build_adapter()/linopy_fixed_design_export.py, which needs
    # RES_Energy_Production/Generator_Energy_Total/Battery_SOC/Energy_Curtailment/Lost_Load/
    # Energy_From_Grid/Energy_To_Grid/lp_mode/plain_mode too, not just the sizing variables this
    # file's own stdout printing used.
    return dict(status=status, condition=condition, npc_true=npc_true, mode=mode,
                best_bound=best_bound, **built)


pinned = {
    "RES": xr.DataArray(D["pinned_res"], coords={"ut": ut_idx}),
    "Battery": xr.DataArray(D["pinned_battery"], coords={"ut": ut_idx}),
    "Generator": xr.DataArray(D["pinned_generator"], coords={"ut": ut_idx}),
}
print("Pinned design (RES/Battery/Generator_Units by step, cumulative):")
print("  RES:", pinned["RES"].values, " (unit =", inputs.RES_NOM_CAP_KW, "kW)")
print("  Battery:", pinned["Battery"].values, " (unit =", inputs.BATT_NOM_CAP_KWH, "kWh)")
print("  Generator:", pinned["Generator"].values, " (unit =", inputs.GEN_NOM_CAP_KW, "kW)")

if os.environ.get("LINOPY_CONFIRM_REAL_SCALE") != "1":
    print("\n" + "!" * 78)
    print(f"STOPPING HERE for design '{DESIGN_NAME}'. Set LINOPY_CONFIRM_REAL_SCALE=1 to solve --")
    print("this is a real HPC-scale resource commitment for every design in DESIGNS, not just capexp5_9sc.")
    print("!" * 78)
    raise SystemExit(0)

def run_checks(res):
    """Battery single-flow check + generator-commitment integrality check on one solve.
    Returns (battery_ok, generator_ok)."""
    _, offenders, _ = check_simultaneous_flows(res["Battery_Inflow"].solution, res["Battery_Outflow"].solution)
    if PLAIN_MODE:
        # Generator_Partial/Full are pinned to 0 under plain dispatch -- there is no commitment to
        # be fractional, so this check carries no information and is not a certification gate.
        print("Generator commitment integrality: N/A -- plain dispatch (Generator_Partial_Load=0)")
        return offenders == 0, True
    worst_partial, worst_full = check_fractional_commitment(res["Generator_Partial"].solution, res["Generator_Full"].solution)
    print(f"Generator commitment integrality: Generator_Partial worst={worst_partial:.6g}, Generator_Full worst={worst_full:.6g}")
    ok = max(worst_partial, worst_full) <= 1e-3
    print(f"  VERDICT: {'integral' if ok else 'NOT INTEGRAL -- relaxed objective may understate true dispatch cost'}")
    return offenders == 0, ok


print("\n" + "=" * 78)
print(f"Solving '{DESIGN_NAME}', RELAXED (mirrors MGPY_FIX_RELAX_BINARIES=1)")
print("=" * 78)
relaxed = build_and_solve(pinned, "relaxed")
print("status:", relaxed["status"], relaxed["condition"], " NPC true:", relaxed["npc_true"])
if relaxed["status"] != "ok":
    raise SystemExit(f"Relaxed solve failed ({relaxed['status']}/{relaxed['condition']}) -- nothing to certify.")
battery_ok, generator_ok = run_checks(relaxed)
battery_verdict = "CERTIFIED FEASIBLE" if battery_ok else "INVALID FOR THIS SHORTCUT"
relaxed_certified = battery_ok and generator_ok

integral = None
if LINOPY_INTEGRAL == "1" or (LINOPY_INTEGRAL == "auto" and not relaxed_certified):
    why = "LINOPY_INTEGRAL=1" if LINOPY_INTEGRAL == "1" else "relaxed solve failed a check"
    print("\n" + "=" * 78)
    print(f"Solving '{DESIGN_NAME}', INTEGRAL ({why}; TimeLimit={INTEGRAL_TIMELIMIT:.0f}s, MIPGap={INTEGRAL_MIPGAP})")
    print("Genset commitment: " + ("PARALLEL load sharing (LINOPY_GEN_PARALLEL=1)" if GEN_PARALLEL
                                   else "at most one unit below full output (default)"))
    print("=" * 78)
    integral = build_and_solve(pinned, "integral")
    print("status:", integral["status"], integral["condition"], " NPC true:", integral["npc_true"],
          " best bound:", integral["best_bound"])
    if integral["npc_true"] is not None:
        # Sanity only: with binary Single_Flow_BESS and integral commitment both checks must pass
        # by construction. A failure here would mean a formulation bug, not a certification issue.
        int_battery_ok, int_generator_ok = run_checks(integral)
        if not (int_battery_ok and int_generator_ok):
            print("WARNING: the INTEGRAL solve itself failed a check -- this should be impossible; investigate before using it.")

# Which number this run certifies -- read by linopy_fixed_design_export.py.
integral_has_solution = integral is not None and integral["npc_true"] is not None
integral_optimal = integral_has_solution and _condition_name(integral["condition"]) == "optimal"
if integral_optimal:
    result, certified, cert_source = integral, True, "integral"
    verdict_tag = "fixeddesign_certified_integral"
elif relaxed_certified:
    result, certified, cert_source = relaxed, True, "relaxed"
    verdict_tag = "fixeddesign_certified"
elif integral_has_solution:
    # Stopped on the time or (soft) memory limit with an incumbent: a real dispatch, so a valid
    # UPPER bound on the design's cost, but not proven within INTEGRAL_MIPGAP -- not certified.
    result, certified, cert_source = integral, False, None
    stopped_by = "memlimit" if _condition_name(integral["condition"]) == "memory_limit" else "timelimit"
    verdict_tag = f"fixeddesign_UNCERTIFIED_integral_{stopped_by}"
else:
    result, certified, cert_source = relaxed, False, None
    verdict_tag = "fixeddesign_UNCERTIFIED"
if GEN_PARALLEL and integral is not None:
    verdict_tag += "_parallel"
# `result` (2026-09-08): linopy_results_export.py's build_adapter() reads `solved_module.result`
# by convention -- linopy_fixed_design_export.py reuses that same adapter builder.

print("\n" + "=" * 78)
print(f"SUMMARY -- '{DESIGN_NAME}'")
print("=" * 78)
print(f"linopy relaxed NPC : {relaxed['npc_true']:,.2f}  (battery: {battery_verdict}, "
      f"generator: {'N/A (plain)' if PLAIN_MODE else ('integral' if generator_ok else 'NOT INTEGRAL')})")
if integral is not None:
    print(f"integral commitment: {'parallel load sharing (LINOPY_GEN_PARALLEL=1)' if GEN_PARALLEL else 'at most one unit below full output'}")
    if integral_has_solution:
        bound_txt = f", best bound {integral['best_bound']:,.2f}" if integral["best_bound"] is not None else ""
        print(f"linopy integral NPC: {integral['npc_true']:,.2f}  ({_condition_name(integral['condition'])}{bound_txt})")
        gap = integral["npc_true"] - relaxed["npc_true"]
        print(f"integral - relaxed : {gap:,.2f} USD ({100 * gap / integral['npc_true']:.4f}%)")
    else:
        print(f"linopy integral    : no solution ({integral['status']}/{integral['condition']})")
if D.get("pyomo_relaxed_npc"):
    diff = relaxed["npc_true"] - D["pyomo_relaxed_npc"]
    print(f"Pyomo relaxed NPC  : {D['pyomo_relaxed_npc']:,.2f}  (battery: {D['pyomo_battery_verdict']}, generator: {D['pyomo_generator_verdict']})")
    print(f"linopy vs Pyomo (relaxed): {diff:,.2f} USD ({100 * diff / D['pyomo_relaxed_npc']:.4f}%)")
if certified:
    print(f"CERTIFIED NPC: {result['npc_true']:,.2f}  (source: {cert_source} solve)")
elif integral_has_solution:
    print(f"NOT CERTIFIED -- true cost lies between the relaxed NPC {relaxed['npc_true']:,.2f} (lower bound)")
    print(f"and the integral incumbent {integral['npc_true']:,.2f} (upper bound)"
          + (f", best bound {integral['best_bound']:,.2f}" if integral["best_bound"] is not None else "") + ".")
    if _condition_name(integral["condition"]) == "memory_limit":
        print("Stopped on the memory limit: re-run with more memory, fewer LINOPY_THREADS or a node file.")
    else:
        print("Re-run with a larger LINOPY_INTEGRAL_TIMELIMIT to close it.")
else:
    print("NOT CERTIFIED -- the relaxed NPC is only a LOWER bound on the true cost"
          + ("; set LINOPY_INTEGRAL=auto or 1 to run the integral re-solve." if LINOPY_INTEGRAL == "0" else "."))
