"""Solver-free feasibility bound for pinned single-step designs against the 12-scenario ensemble.

Read-only. Added 2026-09-14 to test whether a design with a battery below the 12sc autonomy floor is
physically unable to serve the ensemble, without an HPC solve (a pinned certification on the base 12sc
site fails on Battery_Min_Capacity alone, which says nothing about physics).

The bound relaxes the model: it drops the genset minimum load, integer commitment and the battery mode
gate, and keeps zero lost load, genset output <= min(capacity, demand), PV curtailment, the SOC bounds,
charge/discharge efficiencies, the 0.25C charge and discharge caps, a full battery in the first hour and
SOC carried across the 20 years. Each hour the most the battery can gain is the PV left after the gensets
serve as much demand as they can, c = pv + min(G, d) - d; if c < 0 the battery must deliver -c. Charging
greedily keeps the SOC as high as any feasible dispatch can keep it, so if this best-case path drops below
the depth-of-discharge floor, or needs more than the discharge cap, no dispatch of the real model exists.
A design that passes is NOT proven feasible: certify it.

    python _check_ensemble_soc_bound.py                       # the default designs below
    python _check_ensemble_soc_bound.py --design NAME RES_UNITS BATTERY_KWH GEN_UNITS [--design ...]
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

INP = Path(__file__).resolve().parents[1] / "Inputs"
DEFAULT_DESIGNS = [
    ("control: probabilistic 1-step certified", 7356, 14736, 3),
    ("control: probabilistic zero-autonomy certified", 4620, 3479, 3),
    ("paper deterministic 1-step design", 7388, 14273, 2),
    ("HOMER H1 design", 9749, 14273, 2),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--design", nargs=4, action="append", metavar=("NAME", "RES_UNITS", "BATTERY_KWH", "GEN_UNITS"))
    args = ap.parse_args()
    designs = [(n, int(r), float(b), int(g)) for n, r, b, g in args.design] if args.design else DEFAULT_DESIGNS

    manifest = json.load(open(INP / "merged_scenarios_gili_ketapang.json"))
    unit_kw, inv_eff = manifest["settings"]["res_unit_kw"], manifest["settings"]["res_inverter_eff"]
    dem = pd.read_csv(INP / "Demand_12sc_20y_gili_ketapang_merged.csv", sep=";", decimal=",", index_col=0)
    res = pd.read_csv(INP / "RES_Time_Series_12sc_gili_ketapang_merged.csv", index_col=0)
    T, S, Y = 8760, res.shape[1], dem.shape[1] // res.shape[1]
    D = dem.to_numpy().reshape(T, S, Y)                  # MicroGridsPy column (s-1)*Y + y
    pv_per_kw = res.to_numpy() * inv_eff / unit_kw        # kW AC per kW installed
    dod, eta, gen_kw, t_batt = 0.8, 0.9517, 470.0, 4.0   # from the dieselops_dea2025 .dat files
    print(f"inputs: {S} scenarios x {Y} years x {T} h; unit {unit_kw} kW, inverter {inv_eff}; DOD {dod}, "
          f"eta {eta}, genset unit {gen_kw} kW, {t_batt:.0f} h battery power limits")

    for name, res_u, batt, gen_u in designs:
        pv_kw, G, cap, floor = res_u * unit_kw, gen_u * gen_kw, batt / t_batt, batt * (1 - dod)
        below, over_cap, deepest, where = 0, 0, 0.0, None
        for s in range(S):
            soc, pv = batt, pv_per_kw[:, s] * pv_kw
            for y in range(Y):
                c = pv + np.minimum(G, D[:, s, y]) - D[:, s, y]
                for t, ct in enumerate(c):
                    if ct >= 0:
                        soc = min(batt, soc + eta * min(ct, cap))
                        continue
                    if -ct > cap:
                        over_cap += 1
                    soc += ct / eta
                    if soc < floor:
                        below += 1
                        if floor - soc > deepest:
                            deepest, where = floor - soc, (s + 1, y + 1, t)
                        soc = floor
        head = f"{name}: PV {pv_kw:,.2f} kW, battery {batt:,.0f} kWh, gensets {G:,.0f} kW"
        if below or over_cap:
            print(f"{head} -> INFEASIBLE by relaxation: {below} hours below the SOC floor, {over_cap} hours above "
                  f"the discharge cap; deepest shortfall {deepest:,.0f} kWh (scenario {where[0]}, year {where[1]}, "
                  f"hour {where[2]})")
        else:
            print(f"{head} -> passes (not a proof of feasibility)")


if __name__ == "__main__":
    main()
