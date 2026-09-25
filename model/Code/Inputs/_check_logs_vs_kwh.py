"""Records check for the kWh-level demand (2026-09-25): hourly generator logs against the meter's daily kWh.

Read-only. With MGPY_LEVEL_SOURCE=kwh the demand level is the daily kWh production (stand-meter
differences of Cummins 2 + 3, produksi_kwh_harian.csv), so checking the synthetic profile against that record
is circular; the independent check is between the two records the model uses. Each day with all 24 hours in
the hourly log (the builder's own loader: outages and empty readings dropped, timestamps rounded to the hour)
gives an energy as the sum of its 24 hourly kW readings, compared with the meter's production for that date.

The meter's day may not start at midnight, so the same comparison is repeated for every day-start hour
h0 = 0..23 (log energy over [d h0:00, d+1 h0:00), still needing all 24 of those hours logged); the calendar
day (h0 = 0) is the figure to quote unless the reading time is known.

    python _check_logs_vs_kwh.py
"""
import os
import sys

import numpy as np
import pandas as pd

sys.argv = sys.argv[:1]                  # the builder reads nothing from argv; keep its defaults
import _build_demand_merged as b         # noqa: E402  (reuses load_genset_log_rounded and LF_DIR)


def compare(log_kw, meter, h0):
    """Daily log energy over [d h0, d+1 h0) for days with all 24 hours logged, joined to the meter."""
    shifted = log_kw.copy()
    shifted.index = shifted.index - np.timedelta64(h0, "h")
    day = shifted.index.normalize()
    g = shifted.groupby(day)
    e = g.sum()[g.size() == 24]
    j = pd.concat([e.rename("log_kwh"), meter.rename("meter_kwh")], axis=1, join="inner")
    j["ape"] = (j["log_kwh"] / j["meter_kwh"] - 1).abs()
    return j


def main():
    log_kw = b.load_genset_log_rounded(os.path.join(b.LF_DIR, "load data", "harian_gilket_gabungan.csv"))["load_kw"]
    d = pd.read_csv(os.path.join(b.LF_DIR, "load data", "produksi_kwh_harian.csv"), parse_dates=["date"])
    meter = (d["prod_c2_kwh"].fillna(0) + d["prod_c3_kwh"].fillna(0)).set_axis(d["date"])
    print(f"hourly log: {len(log_kw):,} h, {log_kw.index.min()} to {log_kw.index.max()}; "
          f"meter: {len(meter)} days, {meter.index.min():%Y-%m-%d} to {meter.index.max():%Y-%m-%d}")

    j = compare(log_kw, meter, 0)
    r = j["log_kwh"] / j["meter_kwh"] - 1
    print(f"\ncalendar days (00:00-24:00) with all 24 hours logged and a meter value: {len(j)} "
          f"({j.index.min():%Y-%m-%d} to {j.index.max():%Y-%m-%d})")
    print(f"  daily MAPE {100 * j['ape'].mean():.2f}%, median APE {100 * j['ape'].median():.2f}%, "
          f"max {100 * j['ape'].max():.2f}% ({j['ape'].idxmax():%Y-%m-%d})")
    print(f"  bias: mean daily log/meter - 1 = {100 * r.mean():+.2f}%; total energy {j['log_kwh'].sum() / 1e3:,.1f} "
          f"vs {j['meter_kwh'].sum() / 1e3:,.1f} MWh ({100 * (j['log_kwh'].sum() / j['meter_kwh'].sum() - 1):+.2f}%)")
    print(f"  after removing that bias: MAPE {100 * (j['log_kwh'] / (1 + r.mean()) / j['meter_kwh'] - 1).abs().mean():.2f}%; "
          f"correlation of daily energies {j['log_kwh'].corr(j['meter_kwh']):.3f}")
    print(f"  days within 5%: {int((j['ape'] <= 0.05).sum())}, within 10%: {int((j['ape'] <= 0.10).sum())}, "
          f"above 20%: {int((j['ape'] > 0.20).sum())}")
    m = j.groupby(j.index.to_period("M"))[["log_kwh", "meter_kwh"]].sum()
    m["err"] = 100 * (m["log_kwh"] / m["meter_kwh"] - 1)
    n = j.groupby(j.index.to_period("M")).size()
    print("  by month (complete days only): " + "; ".join(
        f"{p}: {n[p]} days, {v:+.2f}%" for p, v in m["err"].items()))
    print(f"  monthly MAPE (energy over the complete days of each month): {m['err'].abs().mean():.2f}%")

    print("\nday-start hour h0: days, daily MAPE, bias")
    rows = []
    for h0 in range(24):
        k = compare(log_kw, meter, h0)
        rb = (k["log_kwh"] / k["meter_kwh"] - 1).mean()
        rows.append((h0, len(k), 100 * k["ape"].mean(), 100 * rb))
    print("  " + "  ".join(f"{h:02d}h {n} {a:.2f}% {bb:+.2f}%" + ("\n " if (h + 1) % 6 == 0 else "")
                           for h, n, a, bb in rows).rstrip())
    best = min(rows, key=lambda t: t[2])
    print(f"lowest daily MAPE at h0 = {best[0]:02d}:00: {best[2]:.2f}% on {best[1]} days (bias {best[3]:+.2f}%)")


if __name__ == "__main__":
    main()
