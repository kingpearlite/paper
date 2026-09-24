import re
import sys
from collections import defaultdict

import gurobipy as gp

lp_path = "/home/<user>/MicroGridsPy-SESAM/Code/Model/Verification_20260828_133139_6sc_20y_capexp5_dieselops.lp"

print(f"Reading {lp_path} ...", flush=True)
m = gp.read(lp_path)
print(f"Read done: {m.NumConstrs} constrs, {m.NumVars} vars", flush=True)

print("Computing IIS...", flush=True)
m.computeIIS()
print("IIS computed.", flush=True)

iis_constrs = [c for c in m.getConstrs() if c.IISConstr]
iis_lb_vars = [v for v in m.getVars() if v.IISLB]
iis_ub_vars = [v for v in m.getVars() if v.IISUB]

print(f"\nIIS: {len(iis_constrs)} constraints, {len(iis_lb_vars)} LB, {len(iis_ub_vars)} UB", flush=True)

def prefix_of(name):
    return re.sub(r"\[.*\]$", "", name)

fam_counts = defaultdict(int)
fam_examples = defaultdict(list)
for c in iis_constrs:
    p = prefix_of(c.ConstrName)
    fam_counts[p] += 1
    if len(fam_examples[p]) < 8:
        fam_examples[p].append(c.ConstrName)

print("\n=== IIS constraint families (by prefix) ===")
for p, cnt in sorted(fam_counts.items(), key=lambda kv: -kv[1]):
    print(f"  {p}: {cnt}")
    for ex in fam_examples[p]:
        print(f"      e.g. {ex}")

print("\n=== IIS variable bounds ===")
lb_fam = defaultdict(list)
for v in iis_lb_vars:
    lb_fam[prefix_of(v.VarName)].append(v.VarName)
ub_fam = defaultdict(list)
for v in iis_ub_vars:
    ub_fam[prefix_of(v.VarName)].append(v.VarName)

print("-- IISLB (lower bound in IIS) --")
for p, names in sorted(lb_fam.items(), key=lambda kv: -len(kv[1])):
    print(f"  {p}: {len(names)}")
    for n in names[:8]:
        print(f"      e.g. {n}")

print("-- IISUB (upper bound in IIS) --")
for p, names in sorted(ub_fam.items(), key=lambda kv: -len(kv[1])):
    print(f"  {p}: {len(names)}")
    for n in names[:8]:
        print(f"      e.g. {n}")

ilp_path = lp_path.replace(".lp", ".ilp")
m.write(ilp_path)
print(f"\nWrote IIS-only model -> {ilp_path}", flush=True)
