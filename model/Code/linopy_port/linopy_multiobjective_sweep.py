"""Driver for run_pareto_sweep() (Stage 6, full-parity roadmap: NPC-vs-CO2 epsilon-constraint
Pareto sweep, mirroring Model_Resolution.py's Multiobjective_Optimization==1 branch). This is
inherently n+3 full MILP solves -- expensive by construction, matching Pyomo's own algorithm
exactly, not a shortcut this port takes. No Gili Ketapang design has ever set
Multiobjective_Optimization=1 (confirmed inert in every .dat), so there is no certified NPC/CO2
Pareto front to cross-validate against yet -- treat any result from this file as unproven until
a real Pyomo run for the same design exists to compare against, same standard as every other
stage's own free-sizing result.

USAGE: LINOPY_CONFIRM_REAL_SCALE=1 LINOPY_DESIGN=<key> python linopy_multiobjective_sweep.py
(same DESIGNS-dict/env-var convention as linopy_free_sizing_crosscheck.py.)
"""
import os
import time

from linopy_core_model_builder import DesignInputs, run_pareto_sweep, start_resource_monitor, stop_resource_monitor, resolve_plain_mode

INPUTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "Code", "Inputs"))

DESIGNS = {
    # STAGE 6 FUNCTIONAL TEST -- reuses the same small local design as every other stage's own
    # functional test (S=1/Years=10/single-step, real Gili Ketapang data, solves fast enough for
    # an n+3-solve sweep on a laptop). Multiobjective_Optimization=0/Pareto_points=2 in the real
    # .dat are inert; this script always runs the sweep regardless of that flag (mirrors Pyomo's
    # OWN behavior only in the sense that Pyomo's Multiobjective_Optimization flag is what
    # SELECTS this code path at all -- this driver's whole purpose is to exercise that path).
    "smoketest_1sc": dict(
        params_dat="Parameters_1sc_10y_gili_ketapang_scpos2_MILP.dat",
        demand_csv="Demand_1sc_10y_gili_ketapang_scpos2.csv",
        res_csv="RES_Time_Series_1sc_gili_ketapang_scpos2.csv",
        max_battery_kwh=25000.0, max_generator_kw=4000.0,
    ),
    # smoketest_1sc_2y_multiobj (2026-09-04, Stage 6 Pyomo cross-check): the same 2y
    # cross-validation base used for Stages 0-5, with Stage 1's nonzero CO2 factors (so the CO2
    # axis is non-degenerate) and Multiobjective_Optimization=1 actually set (real .dat value,
    # not inert this time). Small scale keeps the n+3=5-solve sweep tractable.
    "smoketest_1sc_2y_multiobj": dict(
        params_dat="Parameters_1sc_2y_gili_ketapang_scpos2_MULTIOBJ_MILP.dat",
        demand_csv="Demand_1sc_2y_gili_ketapang_scpos2.csv",
        res_csv="RES_Time_Series_1sc_gili_ketapang_scpos2.csv",
        max_battery_kwh=25000.0, max_generator_kw=4000.0,
    ),
}

# LINOPY_SITE (2026-09-08) -- config-driven alternative to LINOPY_DESIGN/DESIGNS, see
# site_config.py's docstring and linopy_free_sizing_crosscheck.py's own identical branch.
_LINOPY_SITE = os.environ.get("LINOPY_SITE")
if _LINOPY_SITE:
    if os.environ.get("LINOPY_DESIGN"):
        raise ValueError("Set only one of LINOPY_SITE or LINOPY_DESIGN, not both.")
    import site_config
    DESIGN_NAME = _LINOPY_SITE
    D = site_config.load_site_design(_LINOPY_SITE)
else:
    DESIGN_NAME = os.environ.get("LINOPY_DESIGN", "smoketest_1sc")
    D = DESIGNS[DESIGN_NAME]

BESS_FORM = os.environ.get("LINOPY_BESS_FORM", "binary").strip().lower()
# PLAIN_MODE: derived from the .dat's Generator_Partial_Load right after DesignInputs loads below
# (2026-09-11, resolve_plain_mode()).
N_POINTS = int(os.environ.get("LINOPY_PARETO_POINTS", "0")) or None  # 0/unset -> use the .dat's own Pareto_points

# LINOPY_SOLVER (2026-09-05) -- see linopy_free_sizing_crosscheck.py's own comment on this same
# name for the full explanation. Default "gurobi" preserves every existing regression baseline;
# SOLVER_KWARGS below is a Gurobi-tuned recipe, discarded (not translated) when switching away.
LINOPY_SOLVER = os.environ.get("LINOPY_SOLVER", "gurobi").strip().lower()

if LINOPY_SOLVER == "gurobi":
    SOLVER_KWARGS = dict(
        OutputFlag=1,
        Threads=int(os.environ.get("SLURM_CPUS_PER_TASK", 24)),
        Method=2, BarHomogeneous=1, Crossover=1,
        MIPFocus=3, Cuts=1, CutPasses=1,
        BarConvTol=1e-3, OptimalityTol=1e-3, FeasibilityTol=1e-4,
        MIPGap=0.03, TimeLimit=int(os.environ.get("LINOPY_TIMELIMIT", 1800)),
    )
    if os.environ.get("LINOPY_MIPGAP") is not None:
        # 2026-09-04, Stage 1-7 Pyomo cross-check roadmap -- same override as
        # linopy_free_sizing_crosscheck.py, needed to get a genuinely tight (not 3%-MIPGap-loose)
        # comparison against Pyomo on the small 2y cross-validation designs.
        SOLVER_KWARGS["MIPGap"] = float(os.environ["LINOPY_MIPGAP"])
elif LINOPY_SOLVER == "highs":
    # 2026-09-05: see linopy_free_sizing_crosscheck.py's own comment on this same branch for the
    # full real-test evidence -- HiGHS reported a falsely-confident "optimal" at the Gurobi-tuned
    # MIPGap=0.03 while actually 2.77% wrong with a materially different technology mix, and
    # tightening the gap alone did not fix it (hit the time limit stuck on the identical wrong
    # incumbent). This sweep is n+3 solves, so both the time cost AND the false-confidence risk
    # compound. Default gap kept tight for the same reason: an honest `time_limit` beats a false
    # `optimal`. HiGHS's own option names (not Gurobi's) are used, confirmed working by testing.
    SOLVER_KWARGS = dict(
        time_limit=float(os.environ.get("LINOPY_TIMELIMIT", 1800)),
        mip_rel_gap=float(os.environ.get("LINOPY_MIPGAP", 1e-4)),
    )
else:
    # SOLVER_KWARGS above is a Gurobi-tuned recipe -- see LINOPY_SOLVER's own comment. No tuned
    # recipe (or even a time-limit default) exists yet for any backend besides gurobi/highs --
    # both confirmed installed and working in this environment.
    SOLVER_KWARGS = {}
    print(f"LINOPY_SOLVER={LINOPY_SOLVER!r} -- using that backend's own defaults, none of this "
          f"project's Gurobi-tuned solver options apply, and no time limit is set. Confirm this "
          f"solver is actually installed before trusting this to return.")

if os.environ.get("LINOPY_CONFIRM_REAL_SCALE") != "1":
    print("\n" + "!" * 78)
    print(f"STOPPING HERE for design '{DESIGN_NAME}' (Pareto sweep). Set LINOPY_CONFIRM_REAL_SCALE=1")
    print("to run -- this is n+3 full MILP solves, a real resource commitment multiplied by the")
    print("number of Pareto points. Never validated against Pyomo for any design -- see module docstring.")
    print("!" * 78)
    raise SystemExit(0)

inputs = DesignInputs(
    params_dat=os.path.join(INPUTS_DIR, D["params_dat"]),
    demand_csv=os.path.join(INPUTS_DIR, D["demand_csv"]),
    res_csv=os.path.join(INPUTS_DIR, D["res_csv"]),
    max_battery_kwh=D["max_battery_kwh"], max_generator_kw=D["max_generator_kw"],
)
PLAIN_MODE = resolve_plain_mode(inputs)
print(f"Design '{DESIGN_NAME}': S={inputs.S}, YEARS={inputs.YEARS}, STEPS_NUMBER={inputs.STEPS_NUMBER}, "
      f"PARETO_POINTS={N_POINTS or inputs.PARETO_POINTS}, PLOT_MAX_COST={inputs.PLOT_MAX_COST}")

_phase_t0 = {}


def _progress(phase_name, built):
    print(f"  [{phase_name}] solved.")


print("\n" + "=" * 78)
print(f"PARETO SWEEP '{DESIGN_NAME}' -- n+3 solves, resource-monitored")
print("=" * 78)
start_resource_monitor()
t0 = time.time()
try:
    result = run_pareto_sweep(inputs, bess_mode=BESS_FORM, plain_mode=PLAIN_MODE,
                               solver_kwargs=SOLVER_KWARGS, n_points=N_POINTS,
                               progress_callback=_progress, solver_name=LINOPY_SOLVER)
finally:
    stop_resource_monitor()
wall_clock = time.time() - t0

print(f"\nTotal wall clock for full sweep: {wall_clock:.1f}s")
print(f"NPC_min   = {result['NPC_min']:,.2f} USD   (CO2 at this point = {result['CO2_max']:,.3f} kg)")
print(f"CO2_min   = {result['CO2_min']:,.3f} kg    (NPC to reach it, NPC_max = {result['NPC_max']:,.2f} USD)")
print("\nPareto front (NPC, CO2):")
for npc_i, co2_i in zip(result["pareto_npc"], result["pareto_co2"]):
    print(f"  NPC={npc_i:,.2f} USD   CO2={co2_i:,.3f} kg")

if result["NPC_min"] > result["NPC_max"] + 1e-6:
    print("\nWARNING: NPC_min > NPC_max -- this should never happen (the cleanest-design cost "
          "must be >= the cheapest-design cost). Investigate before trusting this sweep.")
if result["pareto_npc"] and min(result["pareto_npc"]) < result["NPC_min"] - 1e-6:
    print("\nWARNING: a swept Pareto point is cheaper than NPC_min -- NPC_min should be the "
          "global cost minimum (any CO2 constraint can only make cost equal or worse). Confirmed "
          "2026-09-04 on smoketest_1sc: this is a real, expected artifact of the loose "
          f"MIPGap={SOLVER_KWARGS['MIPGap']} used here (and by Pyomo's own equivalent algorithm, "
          "Model_Resolution.py's MIPGap=0.01) -- each phase's solve only proves its result to "
          "within its own gap, so two independently gap-limited MIP solves are not guaranteed to "
          "rank consistently even when one is a genuine relaxation of the other. Tighten MIPGap "
          "(e.g. 0.005 or lower) for any Pareto front used for real decision-making, not just a "
          "functional smoke test of the mechanism.")

print("\nNo Pyomo comparison exists for this design's Pareto front -- treat this as an initial,")
print("unvalidated result, same standard as every other stage's own free-sizing output.")
