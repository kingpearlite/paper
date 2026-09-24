"""Audit Code/Results/*/Results_Summary.xlsx for the relaxed-salvage bug (found 2026-09-11).

THE BUG. From the salvage floor-at-zero patch (2026-08-28) until the fix on 2026-09-11,
MGPY_FIX_RELAX_BINARIES=1 also relaxed the scalar binary Salvage_Positive_Flag, which sits in
the big-M pair  S <= S_raw + M*(1-f),  S <= M*f.  Relaxed, the solver books
    S = (S_raw + M) / 2
i.e. roughly M/2 of fictitious salvage, subtracted from the NPC. See
docs/microgridspy_pipeline_usage.tex, Known gaps, and KEEP_INTEGRAL_WHEN_RELAXING in
Model_Resolution.py.

THE TEST. Genuine salvage in this project is tens to a few hundred kUSD; a bugged run carries at
least M/2 = 1,000 kUSD (M = BIG_M_SALVAGE, which was 2e7, then 5e6, then 2e6 USD over
2026-08-28/29). So a run is flagged SUSPECT when |Salvage value| >= 900 kUSD. For each M the
bug could have used, the raw salvage is recovered as S_raw = 2*S - M, and the corrected NPC as
    NPC_true ~= NPC_reported + S - max(S_raw, 0)     (= NPC + (M - S) when S_raw >= 0)
S_raw may be negative -- that is the very case the floor was added for (short-lifetime gensets
installed early), where the true salvage is 0. Only an M giving -1,500 <= S_raw <= 1,500 kUSD
is reported as consistent. With M = 2e6 the threshold catches S_raw down to -200 kUSD; the
most negative raw salvage ever observed here was -8.5 kUSD. Checked against the
2026-09-11 smoketest_6sc_2y_dieselops runs: the bugged relaxed run (NPC 1,992.187, salvage
1,198.217) gives S_raw 396.43 and NPC_true 2,793.970 at M = 2,000 -- the re-run after the fix
reports 2,793.972 and salvage 396.433.

Verdicts:
  SUSPECT  dated 2026-08-28 or later, |salvage| >= 900 kUSD, some BIG_M gives a plausible S_raw,
           AND the folder tag contains "fix" (every fixed-design tag in this project does:
           pyomo_fix_..., ..._fixdesign). Confirm it was MGPY_FIX_RELAX_BINARIES=1 and re-run it.
  CHECK    large salvage that either fits no BIG_M, or fits one but the tag has no "fix". The
           numbers alone are ambiguous: genuine salvage can be large -- a 6sc/4y design from
           2026-08-16 books 2,280 kUSD, and a 9sc/20y free-sizing run from 2026-08-30 books
           936 kUSD because it installs 21,698 kW of PV at its final step -- so confirm by hand.
  ok       below the threshold, or dated before 2026-08-28 ("pre-patch": the bug did not exist).
The script only reads files; it changes nothing.

    python _audit_salvage_relax_bug.py [RESULTS_DIR ...]      # default: ../Results
    e.g. on HPC:
    python _audit_salvage_relax_bug.py /scratch/<project>/<user>/MicroGridsPy-SESAM_old_20260911/Code/Results
"""
import glob
import os
import re
import sys

import pandas as pd

SUSPECT_KUSD = 900.0                        # |salvage| at or above this is flagged
BIG_M_KUSD = (20000.0, 5000.0, 2000.0)      # BIG_M_SALVAGE values in force 2026-08-28/29, kUSD
MAX_PLAUSIBLE_RAW_KUSD = 1500.0
PATCH_DATE = "20260828"


def read_summary(xlsx):
    """Return (npc_kusd, salvage_kusd) from the Cost sheet, or raise."""
    cost = pd.read_excel(xlsx, sheet_name="Cost", header=None)
    # The value sits in the column headed "Total" -- NOT the last column: multi-step designs
    # append "Step 1 .. Step n" columns after it, filled with '-' on these rows.
    header = [str(v).strip() for v in cost.iloc[0].tolist()]
    if "Total" not in header:
        raise ValueError(f"Cost sheet header has no 'Total' column: {header}")
    col = header.index("Total")
    npc = salvage = None
    for _, row in cost.iloc[1:].iterrows():
        item = str(row.iloc[0]).strip().lower()
        if item == "weighted net present cost":
            npc = float(row.iloc[col])
        elif item == "salvage value":
            salvage = float(row.iloc[col])
    if npc is None or salvage is None:
        raise ValueError("Cost sheet has no 'Weighted Net present cost' / 'Salvage value' row")
    return npc, salvage


def classify(folder_name, npc, salvage):
    s = abs(salvage)
    date = (re.match(r"(\d{8})_", folder_name) or [None, ""])[1]
    if date and date < PATCH_DATE:
        return "ok (pre-patch)", ""
    if s < SUSPECT_KUSD:
        return "ok", ""
    fits = []
    for m in BIG_M_KUSD:
        raw = 2 * s - m
        # raw may be NEGATIVE: that is exactly the case the floor exists for (short-lifetime
        # gensets installed early), and the relaxed flag still books (raw + M)/2 > 0 there. The
        # true (floored) salvage is max(raw, 0), so NPC_true = NPC + S - max(raw, 0).
        if -MAX_PLAUSIBLE_RAW_KUSD <= raw <= MAX_PLAUSIBLE_RAW_KUSD:
            fits.append(f"M={m:,.0f}: raw salvage {raw:,.2f}, NPC_true ~{npc + s - max(raw, 0.0):,.3f}")
    if fits and "fix" in folder_name.lower():
        return "SUSPECT", "; ".join(fits)
    if fits:
        # The numbers alone cannot separate a bugged run with a slightly negative raw salvage from
        # a genuine one (e.g. 20260830_083505_capexp4_9sc_nobinary_dm0_coldstart: a FREE-SIZING
        # run whose 936 kUSD salvage is real -- 21,698 kW of PV installed at the final step keeps
        # ~84% of its value). The run type is what decides, and only the folder tag hints at it.
        return "CHECK", ("fits the bug pattern, but the tag has no 'fix' -- the bug only touches relaxed "
                         "fixed-design runs (MGPY_FIX_RELAX_BINARIES=1); confirm the run type. " + "; ".join(fits))
    return "CHECK", "large salvage but no BIG_M value fits -- probably genuine (short horizon?); inspect by hand"


def main(dirs):
    rows = []
    for d in dirs:
        for xlsx in sorted(glob.glob(os.path.join(d, "*", "Results_Summary.xlsx"))):
            name = os.path.basename(os.path.dirname(xlsx))
            try:
                npc, salvage = read_summary(xlsx)
                verdict, detail = classify(name, npc, salvage)
            except Exception as e:  # noqa: BLE001 -- report and keep going
                npc = salvage = float("nan")
                verdict, detail = "UNREADABLE", str(e)[:80]
            rows.append((name, npc, salvage, verdict, detail))
    if not rows:
        print(f"No */Results_Summary.xlsx found under: {', '.join(dirs)}")
        return 1
    w = max(len(r[0]) for r in rows)
    print(f"{'run folder':<{w}}  {'NPC kUSD':>11}  {'salvage kUSD':>12}  verdict")
    for name, npc, salvage, verdict, detail in rows:
        print(f"{name:<{w}}  {npc:>11,.3f}  {salvage:>12,.3f}  {verdict}" + (f"  [{detail}]" if detail else ""))
    n_sus = sum(r[3] == "SUSPECT" for r in rows)
    n_chk = sum(r[3] == "CHECK" for r in rows)
    n_bad = sum(r[3] == "UNREADABLE" for r in rows)
    n_ok = len(rows) - n_sus - n_chk - n_bad
    print(f"\n{len(rows)} runs scanned: {n_sus} SUSPECT, {n_chk} CHECK, {n_bad} unreadable, {n_ok} ok.")
    if n_sus:
        print("SUSPECT: if the run was a relaxed fixed-design run (MGPY_FIX_RELAX_BINARIES=1), its NPC is "
              "too LOW by about S - max(S_raw, 0). Re-run it with the fixed Model_Resolution.py (git pull) "
              "before using the number; NPC_true above is only an estimate.")
    if n_chk:
        print("CHECK: large salvage the numbers cannot settle -- either no BIG_M fits (most likely genuine), "
              "or one fits but the tag does not mark a fixed-design run. Confirm the run type by hand.")
    return 0


if __name__ == "__main__":
    here = os.path.dirname(os.path.abspath(__file__))
    sys.exit(main(sys.argv[1:] or [os.path.join(here, "..", "Results")]))
