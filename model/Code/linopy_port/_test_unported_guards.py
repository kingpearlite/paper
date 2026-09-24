"""Regression test for DesignInputs' unported-feature guards (added 2026-09-11).

Model_Components, RE_Supply_Calculation and Demand_Profile_Generation must each make
DesignInputs raise NotImplementedError when non-zero -- before these guards, linopy never read
them and would silently solve a .dat that set them as if they were 0. No solve is run; this only
builds DesignInputs, so it takes seconds.

    cd Code/linopy_port && conda activate mgpy
    python _test_unported_guards.py        # exit code 0 = all passed
"""
import glob
import os
import re
import shutil
import sys
import tempfile

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from linopy_core_model_builder import DesignInputs, load_dat_scalar  # noqa: E402

INPUTS = os.path.join(HERE, "..", "Inputs")
DAT = os.path.join(INPUTS, "Parameters_1sc_2y_gili_ketapang_scpos2_MILP.dat")
DEM = os.path.join(INPUTS, "Demand_1sc_2y_gili_ketapang_scpos2.csv")
RES = os.path.join(INPUTS, "RES_Time_Series_1sc_gili_ketapang_scpos2.csv")
FLAGS = ["Model_Components", "RE_Supply_Calculation", "Demand_Profile_Generation"]


def main():
    fails = 0

    # 1. The parser reads all three flags from every .dat -- a ValueError here would mean the
    #    guard itself crashes on a real file instead of passing it through.
    dats = sorted(glob.glob(os.path.join(INPUTS, "*.dat")))
    nonzero = []
    for d in dats:
        vals = {f: load_dat_scalar(d, f) for f in FLAGS}
        if any(vals.values()):
            nonzero.append(os.path.basename(d))
    print(f"[1] parsed all 3 flags from {len(dats)} .dat files; non-zero in: {nonzero}")

    # 2. The unmodified smoketest design still builds.
    inp = DesignInputs(DAT, DEM, RES)
    print(f"[2] baseline OK: S={inp.S} YEARS={inp.YEARS} MODEL_COMPONENTS={inp.MODEL_COMPONENTS}")

    # 3. Each flag switched on must raise NotImplementedError.
    tmpdir = tempfile.mkdtemp()
    try:
        text = open(DAT, encoding="utf-8").read()
        for flag, val in [("Model_Components", 1), ("Model_Components", 2),
                          ("RE_Supply_Calculation", 1), ("Demand_Profile_Generation", 1)]:
            patched, n = re.subn(rf"(param: {flag} := )0;", rf"\g<1>{val};", text)
            assert n == 1, f"could not patch {flag} in {DAT}"
            path = os.path.join(tmpdir, f"{flag}_{val}.dat")
            open(path, "w", encoding="utf-8").write(patched)
            try:
                DesignInputs(path, DEM, RES)
                print(f"[3] FAIL: {flag}={val} did NOT raise")
                fails += 1
            except NotImplementedError as e:
                print(f"[3] OK: {flag}={val} -> NotImplementedError: {str(e)[:70]}...")
    finally:
        shutil.rmtree(tmpdir)

    print("ALL PASSED" if fails == 0 else f"{fails} FAILURE(S)")
    return fails


if __name__ == "__main__":
    sys.exit(main())
