"""Runs a linopy fixed-design (RELAXED) verification (via linopy_fixed_design_crosscheck.py)
and feeds the solved values into the REAL project's own Results.py, UNMODIFIED -- the
fixed-design counterpart to linopy_results_export.py (which only ever wrapped the free-sizing
path). Closes the gap documented in docs/microgridspy_pipeline_usage.tex's Section "Known gaps"
("No fixed-design xlsx exporter yet").

WHY THIS WAS A SMALL GAP, NOT A NEW MECHANISM: build_core_model() (linopy_core_model_builder.py)
already returns every variable build_adapter() needs; linopy_fixed_design_crosscheck.py's own
build_and_solve_relaxed() just used to throw most of that dict away before this file existed.
build_adapter() itself (defined in linopy_results_export.py) is generic -- it only needs
`solved_module.result` (a dict) and `solved_module.PARAMS_DAT` (a path), both of which
linopy_fixed_design_crosscheck.py now exposes at module level. So this file is a thin driver,
not a second results-building implementation.

IMPORTANT -- what NPC this actually reports (updated 2026-09-11): linopy_fixed_design_crosscheck.py
first solves the RELAXED formulation (continuous Single_Flow_BESS/Generator_Partial/
Generator_Full, mirroring Pyomo's MGPY_FIX_RELAX_BINARIES=1) and runs both the battery
(Check_Simultaneous_Flows) and generator-commitment-integrality checks. If either fails, it
re-solves the same pinned design as a true INTEGRAL MILP (LINOPY_INTEGRAL=auto, the default).
The exported workbook is whichever solve that script chose as `result`, and the Results/<RUN_ID>
folder is tagged with its verdict:
  fixeddesign_certified                       -- relaxed solve passed both checks
  fixeddesign_certified_integral              -- integral re-solve reached its MIPGap target
  fixeddesign_UNCERTIFIED_integral_timelimit  -- integral re-solve hit its time limit; the
                                                 exported NPC is an UPPER bound, not certified
  fixeddesign_UNCERTIFIED_integral_memlimit   -- the same, stopped by LINOPY_SOFTMEMLIMIT
                                                 (Gurobi status 17, incumbent recovered; 2026-09-18)
  (any of the above + "_parallel" when LINOPY_GEN_PARALLEL=1 was used for the integral re-solve)
  fixeddesign_UNCERTIFIED                     -- relaxed only (LINOPY_INTEGRAL=0, or the
                                                 integral re-solve found no solution); the
                                                 exported NPC is a LOWER bound
so an exported xlsx can never be mistaken for a certified result just because the export ran.

USAGE: LINOPY_CONFIRM_REAL_SCALE=1 LINOPY_DESIGN=capexp5_9sc python linopy_fixed_design_export.py [tag]
(same env vars as linopy_fixed_design_crosscheck.py -- importing that module IS how this script
triggers the actual solve; [tag] is an optional extra Results/<RUN_ID> folder-name suffix,
appended after the automatic certified/UNCERTIFIED tag.)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "Code", "Model")))

if __name__ == "__main__":
    # Importing this SOLVES the model (same LINOPY_CONFIRM_REAL_SCALE/LINOPY_SITE/LINOPY_DESIGN
    # env-var gate as running it directly) -- deferred to here, same reason as
    # linopy_results_export.py's own identical deferred import.
    import linopy_fixed_design_crosscheck as solved

    if solved.result["status"] != "ok":
        raise SystemExit(f"Solve did not reach status 'ok' (got {solved.result['status']}/{solved.result['condition']}) -- not exporting Results.")

    from linopy_results_export import build_adapter, export_adapter_to_results  # noqa: E402

    adapter = build_adapter(solved)

    certified = solved.certified
    verdict_tag = solved.verdict_tag
    print(f"\nExporting the {solved.result['mode'].upper()} solve -- verdict: {verdict_tag}")
    if not certified:
        bound = "UPPER" if solved.result["mode"] == "integral" else "LOWER"
        print("\n" + "!" * 78)
        print(f"WARNING: NOT a certified fixed-design result -- the exported NPC is only a {bound}")
        print("bound on this design's true cost (see the SUMMARY block above). Exporting anyway for")
        print(f"inspection; the Results/<RUN_ID> folder below is tagged '{verdict_tag}' so this can")
        print("never be mistaken for a certified number.")
        print("!" * 78)

    # Tag Results/<RUN_ID> with the verdict (+ any user-supplied extra tag), same sys.argv[1]
    # convention run_id.py already uses -- Results.py imports run_id at the top of
    # export_adapter_to_results(), so sys.argv must be set before that call, not after.
    user_tag = sys.argv[1] if len(sys.argv) > 1 else None
    sys.argv = [sys.argv[0], f"{verdict_tag}_{user_tag}" if user_tag else verdict_tag]

    export_adapter_to_results(adapter)
