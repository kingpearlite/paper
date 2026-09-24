"""Fetches NASA POWER renewable-resource data and writes RES_Time_Series_<site>.csv, without
requiring Pyomo or MicroGrids.py at all -- the missing piece that made the linopy/VS Code
workflow (linopy_free_sizing_crosscheck.py etc.) still depend on one Pyomo bootstrap run.

WHY THIS WAS POSSIBLE WITHOUT TOUCHING RE_calculation.py: `RE_supply()` in
Code/Model/RE_calculation.py already has zero Pyomo dependency -- it reads
`params_path.PARAMS_PATH` (a plain env-var-resolved file path, itself Pyomo-free) and parses the
`.dat` file's raw text lines directly with regex, the same way Initialize.py does before it ever
builds a Pyomo model. It even already has `if __name__ == "__main__": RE_supply()`, so it was
already runnable standalone -- the only real gaps were (1) its output is hardcoded to the generic
`Code/Inputs/RES_Time_Series.csv`, not the per-site name the linopy DESIGNS dicts expect, and
(2) there was no entry point living in Code/linopy_port/ alongside the rest of that workflow.
This script is a thin wrapper closing both gaps; RE_calculation.py itself is unmodified.

USAGE (mirrors the MGPY_PARAMS convention already used throughout this project):
    MGPY_PARAMS=Parameters_newsite.dat LINOPY_RES_OUTPUT=RES_Time_Series_newsite.csv \
        python linopy_res_fetch.py

LINOPY_RES_OUTPUT is required (no silent default) -- this writes into Code/Inputs/, which is
shared mutable state across every design in this project (see params_path.py's own documented
race-condition incidents); an explicit, unambiguous target name is the same discipline
params_path.py already enforces for MGPY_PARAMS/MGPY_DEMAND.

LIVE-TESTED end-to-end (2026-09-01, myenv conda env): a real NASA POWER fetch against
Parameters_6sc_5y_MILP.dat's coordinates completed in ~10s and produced a correctly-named,
8760-row RES_Time_Series_<name>.csv, with zero Pyomo import anywhere in the call chain.
"""
import os
import shutil
import sys

sys.path.insert(0, os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "Model")))

OUTPUT_NAME = os.environ.get("LINOPY_RES_OUTPUT")
if not OUTPUT_NAME:
    raise SystemExit(
        "LINOPY_RES_OUTPUT is not set -- refusing to guess. Set it to the exact "
        "RES_Time_Series_<site>.csv filename this design's DESIGNS entry (res_csv=...) expects, "
        "e.g.: LINOPY_RES_OUTPUT=RES_Time_Series_newsite.csv"
    )

import params_path  # noqa: E402  (import after sys.path insert; also validates MGPY_PARAMS exists)
import RE_calculation  # noqa: E402

INPUTS_DIR = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "Inputs"))
GENERIC_OUTPUT = os.path.join(INPUTS_DIR, "RES_Time_Series.csv")
target_path = OUTPUT_NAME if os.path.isabs(OUTPUT_NAME) else os.path.join(INPUTS_DIR, OUTPUT_NAME)

if os.path.exists(target_path):
    raise SystemExit(
        f"{target_path} already exists -- refusing to overwrite silently. Remove it first if "
        "you intend to re-fetch."
    )

print(f"Fetching NASA POWER data for params file: {params_path.PARAMS_PATH}")
print(f"Will write result to: {target_path}")

# RE_supply() writes to params_path.RES_PATH, which is the tracked, shared
# Code/Inputs/RES_Time_Series.csv unless MGPY_RES is set -- and MGPY_RES cannot point at the
# not-yet-existing target, because params_path.resolve_input() refuses missing files. So the
# fetch necessarily overwrites that shared file. Before 2026-09-11 this script then MOVED it to
# the target, leaving the tracked generic file deleted from the working tree after every fetch.
# Keep a copy of the shared file and put it back afterwards instead.
fetch_output = params_path.RES_PATH
backup_path = None
if os.path.exists(fetch_output):
    backup_path = fetch_output + ".pre_linopy_res_fetch"
    shutil.copy2(fetch_output, backup_path)
try:
    RE_calculation.RE_supply()  # writes to params_path.RES_PATH (see note above)
    if not os.path.exists(fetch_output):
        raise SystemExit(f"RE_supply() completed but {fetch_output} was not created -- something went wrong upstream.")
    shutil.copy2(fetch_output, target_path)
finally:
    if backup_path is not None:
        shutil.move(backup_path, fetch_output)  # restore the shared file exactly as it was
print(f"Done. Wrote {target_path} (shared {os.path.basename(fetch_output)} restored unchanged)")
