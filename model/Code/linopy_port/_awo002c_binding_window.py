"""AWO-002 C (2026-09-25): which low-sun stretch sets the PV capacity of the dea2025k 12sc 1-step designs.

Read-only and solver-free. Workbook item AWO-002 asked why the ensemble design's PV depends on the
upper-tail member (A: P25/P75 members) and on the PV year paired with it (B: five re-pairings). The
certified designs of 2026-09-24/25 measured how much; this looks for why. Two parts per design:

1. Relaxation, identical to _check_ensemble_soc_bound.py (no minimum load, integer commitment or mode
   gate; zero lost load, gensets <= min(capacity, demand), curtailment, SOC floor, efficiencies, 0.25C
   caps, full battery in hour 1, SOC carried over the 20 years): the least PV that keeps the tail
   scenario's best-case SOC at or above the floor, by bisection with the design's own battery and
   gensets. At that PV the binding drawdown -- from the last hour the battery was full to the first
   hour it reaches the floor -- and the PV yield inside it. The same bisection on scenarios 1-11 shows
   whether any other scenario needs more PV.
2. The certified dispatch (Time_Series_SC_12.xlsx of the design's results folder in Code/Results): the
   hours the tail scenario's SOC sits at the floor, by year, and the drawdown ending at the first of
   them in the latest year that has one.

The relaxed PV is a lower bound on what any dispatch needs, not the optimum: the certified PV adds what
the costs justify. Hours are 0-based hours of a 365-day year, dated on 2026.

    python _awo002c_binding_window.py
"""
import glob
import json
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
INP, RESULTS = CODE / "Inputs", CODE / "Results"
SITE = "gili_ketapang_12sc_20y_MILP_1step_dieselops_dea2025k"
DOD, ETA, GEN_KW, T_BATT = 0.8, 0.9517, 470.0, 4.0   # the dieselops_dea2025k .dat files
T, TAIL = 8760, 11                                    # tail stratum = scenario 12

# label, demand/manifest suffix, RES suffix, results-site suffix, certified RES units, battery kWh, gensets
DESIGNS = [
    ("base: member 122, own PV year", "_kwh", "_kwh", "", 12731, 15818, 2),
    ("B: PV year min (member 114)", "_kwh", "_kwh_tailpvmin", "_tailpvmin", 18291, 15818, 2),
    ("B: PV year P25 (member 175)", "_kwh", "_kwh_tailpvp25", "_tailpvp25", 13358, 15818, 2),
    ("B: PV year P50 (member 147)", "_kwh", "_kwh_tailpvp50", "_tailpvp50", 22490, 15818, 2),
    ("B: PV year P75 (member 10)", "_kwh", "_kwh_tailpvp75", "_tailpvp75", 24879, 15818, 2),
    ("B: PV year max (member 185)", "_kwh", "_kwh_tailpvmax", "_tailpvmax", 13200, 15818, 2),
    ("A: tail member P25 (member 9)", "_kwh_tailq25", "_kwh_tailq25", "_tailq25", 8720, 15833, 2),
    ("A: tail member P75 (member 150)", "_kwh_tailq75", "_kwh_tailq75", "_tailq75", 27443, 16160, 2),
]


def when(t):
    return (datetime(2026, 1, 1) + timedelta(hours=int(t))).strftime("%d %b %H:00")


def soc_path(pv_kw, batt, gen_kw, dem, pvk):
    """Best-case SOC (kWh) at the end of every hour over all years; dem is (Y, T), pvk is kW per kW."""
    cap, floor, soc = batt / T_BATT, batt * (1 - DOD), batt
    out, below = np.empty(dem.size), 0
    pv = (pvk * pv_kw).tolist()
    i = 0
    for d_y in dem:
        for p, d in zip(pv, d_y.tolist()):
            c = p + min(gen_kw, d) - d
            if c >= 0:
                soc = min(batt, soc + ETA * min(c, cap))
            else:
                soc += c / ETA
                if soc < floor - 1e-6:
                    below += 1
                    soc = floor
            out[i] = soc
            i += 1
    return out, below


def min_pv(batt, gen_kw, dem, pvk, hi=40000.0):
    """Least PV (kW, to 0.5 kW) at which the relaxed SOC never drops below the floor."""
    if soc_path(0.0, batt, gen_kw, dem, pvk)[1] == 0:
        return 0.0
    lo = 0.0
    while hi - lo > 0.5:
        mid = 0.5 * (lo + hi)
        lo, hi = (lo, mid) if soc_path(mid, batt, gen_kw, dem, pvk)[1] == 0 else (mid, hi)
    return hi


def drawdown(soc, batt, floor, t_end):
    """Last hour at a full battery before t_end (index into the flattened path)."""
    full = np.flatnonzero(soc[:t_end] >= batt - 1e-6)
    return int(full[-1]) if full.size else 0


def main():
    cache = {}
    print(f"Relaxed bound: DOD {DOD}, eta {ETA}, genset unit {GEN_KW} kW, {T_BATT:.0f} h battery limits")
    rows = []
    for label, dsuf, rsuf, site_suf, res_u, batt, gen_u in DESIGNS:
        settings = json.load(open(INP / f"merged_scenarios_gili_ketapang{dsuf}.json"))["settings"]
        unit_kw, inv = settings["res_unit_kw"], settings["res_inverter_eff"]
        if dsuf not in cache:
            dem = pd.read_csv(INP / f"Demand_12sc_20y_gili_ketapang_merged{dsuf}.csv", sep=";", decimal=",",
                              index_col=0).to_numpy()
            cache[dsuf] = dem.reshape(T, 12, -1).transpose(1, 2, 0)      # (S, Y, T)
        dem = cache[dsuf]
        res = pd.read_csv(INP / f"RES_Time_Series_12sc_gili_ketapang_merged{rsuf}.csv", index_col=0).to_numpy()
        pvk = res * inv / unit_kw                                            # kW AC per kW installed
        gen_kw, floor = gen_u * GEN_KW, batt * (1 - DOD)
        need = min_pv(batt, gen_kw, dem[TAIL], pvk[:, TAIL])
        others = max(min_pv(batt, gen_kw, dem[s], pvk[:, s]) for s in range(TAIL))
        soc, _ = soc_path(need, batt, gen_kw, dem[TAIL], pvk[:, TAIL])
        t_bind = int(np.argmin(soc))          # the lowest SOC: at the floor, to the bisection tolerance
        t_start = drawdown(soc, batt, floor, t_bind)
        y, h1, h0 = t_bind // T + 1, t_bind % T, t_start % T
        days = (t_bind - t_start + 1) / 24
        pv_win = pvk[h0:h1 + 1, TAIL].sum() / days if t_start // T + 1 == y else float("nan")
        pv_year = pvk[:, TAIL].sum() / 365
        d_win = dem[TAIL, y - 1, h0:h1 + 1].sum() / days
        print(f"\n{label}\n  annual PV yield {pvk[:, TAIL].sum():,.1f} kWh/kW AC; year-20 tail demand "
              f"{dem[TAIL, -1].sum() / 1e3:,.1f} MWh, peak {dem[TAIL, -1].max():,.1f} kW")
        print(f"  relaxed least PV {need:,.1f} kW (scenarios 1-11 need at most {others:,.1f} kW); "
              f"certified PV {res_u * unit_kw:,.1f} kW ({res_u * unit_kw / need:.2f} x)")
        print(f"  binding drawdown: year {y}, {when(h0)} -> {when(h1)} ({days:.1f} days); PV yield in it "
              f"{pv_win:.2f} kWh/kW/day against {pv_year:.2f} for the year ({100 * pv_win / pv_year:.0f}%); "
              f"demand {d_win:,.0f} kWh/day")
        # The PV years are 24 h blocks drawn from one measured record, so the same cloudy days recur on
        # different dates in different members; March-April (hours 1416-2879) is the tail's high-load season.
        daily = pvk[:, TAIL].reshape(365, 24).sum(axis=1)
        low = daily[59:120]
        print(f"  daily PV yield: lowest {daily.min():.2f} kWh/kW ({when(24 * daily.argmin())[:6]}); March-April "
              f"lowest {low.min():.2f} ({when(24 * (59 + low.argmin()))[:6]}), {(low < 2.0).sum()} days under 2.0")
        rows.append((label, site_suf, batt, floor, t_bind))

    print("\nCertified dispatch, tail scenario (Time_Series_SC_12.xlsx):")
    for label, site_suf, batt, floor, t_rel in rows:
        hits = glob.glob(str(RESULTS / f"*_fixeddesign_certified_integral_{SITE}{site_suf}"))
        if not hits:
            print(f"  {label}: no certified results folder in {RESULTS}")
            continue
        sheets = pd.read_excel(Path(sorted(hits)[-1]) / "Time_Series_SC_12.xlsx", sheet_name=None, header=None)
        # Rows 0-4 are headers, column 11 is "Battery SOC" (kWh, exported to 0.1 kWh).
        soc_all = np.concatenate([pd.to_numeric(sheets[f"Year {y}"].iloc[5:5 + T, 11]).to_numpy()
                                  for y in range(1, 21)])
        # The cost-optimal dispatch runs the battery to the floor whenever diesel is dearer, so floor hours
        # are everywhere; what matters is whether it is at the floor where the relaxation binds.
        hits = np.flatnonzero(soc_all <= floor + 0.5)
        by_year = np.bincount(hits // T, minlength=20)
        near = soc_all[max(0, t_rel - 24):t_rel + 25]
        print(f"  {label}: {hits.size} h at the SOC floor ({floor:,.1f} kWh) over 20 years "
              f"(years 19-20: {by_year[18]} and {by_year[19]} h); lowest SOC within 24 h of the relaxed "
              f"binding hour (year {t_rel // T + 1}, {when(t_rel % T)}): {near.min():,.1f} kWh")


if __name__ == "__main__":
    main()
