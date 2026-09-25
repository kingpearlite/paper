"""Every number the SAC 2027 paper's revision (rev1) takes from the dea2025k designs (2026-09-25).

Read-only: parses the certification logs (laptop 1sc lane, HPC 12sc lanes), two results folders, the
dea2025k inputs and their build manifests, and prints the values for Tables 4-6 and the Section 5 text.
Nothing here is re-solved. The paper's own wording is not reproduced; this only states the quantities.

    python _sac_numbers_dea2025k.py
"""
import glob
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

CODE = Path(__file__).resolve().parents[1]
REPO, INP, RES = CODE.parent, CODE / "Inputs", CODE / "Results"
LOGS = REPO / "logs"
GEN_KW, T, Y = 470.0, 8760, 20
# The 1sc base certification and the 12sc design pinned on the 1sc site share one site name, so their
# results folders are named here (logs/lanes_dea2025k_1sc_laptop and logs/cross_dea2025k).
FOLDER_1SC_BASE = "20260924_121335_fixeddesign_certified_integral_gili_ketapang_1sc_20y_MILP_1step_dieselops_dea2025k"
FOLDER_CROSS = "20260925_030233_fixeddesign_certified_integral_gili_ketapang_1sc_20y_MILP_1step_dieselops_dea2025k"
DESIGNS = [("base", "1step", ""), ("r=10%", "1step", "_dr10"), ("r=8%", "1step", "_dr08"),
           ("D=0 d", "1step", "_aut0"), ("D=2 d", "1step", "_aut2"), ("fuel -25%", "1step", "_fuel064"),
           ("fuel +25%", "1step", "_fuel107"), ("growth 2018-23", "1step", "_g24h"),
           ("3-step", "3step", ""), ("5-step", "5step", "")]


def cert_log(sc, step, suf):
    name = f"gili_ketapang_{sc}_20y_MILP_{step}_dieselops_dea2025k{suf}_cert.log"
    hits = glob.glob(str(LOGS / ("lanes_dea2025k_1sc_laptop" if sc == "1sc" else "lanes_dea2025k_12sc_*") / name))
    t = Path(hits[0]).read_text(encoding="utf-8", errors="replace")
    num = lambda pat: float(re.search(pat, t).group(1).replace(",", ""))
    row = lambda label: [float(x) for x in re.search(rf"^{label}\s+kWh?\s+(.*)$", t, re.M).group(1).split()]
    pv, batt, gen = row("Solar PV"), row("Battery bank"), row("Diesel Genset 1")
    return dict(npc=num(r"CERTIFIED NPC: ([\d,.]+)"), pv=pv, batt=batt, gen=gen[-1], G=round(gen[-1] / GEN_KW),
                lcoe=num(r"LCOE = ([\d.]+)"), re=num(r"renewable penetration per year = ([\d.]+)"),
                curt=num(r"curtailment per year = ([\d.]+)"))


def cost_sheet(folder):
    df = pd.read_excel(RES / folder / "Results_Summary.xlsx", sheet_name="Cost", header=None)
    df[0] = df[0].ffill()
    return {(str(a).strip(), str(b).strip()): float(v) for a, b, v in zip(df[0], df[1], df[4]) if a != "Cost item"}


def demand(suffix, sc):
    d = pd.read_csv(INP / f"Demand_{sc}_20y_gili_ketapang_merged{suffix}.csv", sep=";", decimal=",", index_col=0)
    return d.to_numpy().reshape(T, -1, Y)            # (T, S, Y): MicroGridsPy column (s-1)*Y + y


def manifest(suffix):
    return json.load(open(INP / f"merged_scenarios_gili_ketapang{suffix}.json"))


def main():
    one = {k: cert_log("1sc", st, sf) for k, st, sf in DESIGNS}
    twelve = {k: cert_log("12sc", st, sf) for k, st, sf in DESIGNS}
    pct = lambda a, b: 100 * (a / b - 1)

    print("== Table 5 (base designs): NPC kUSD, PV kWp, battery kWh, genset kW, LCOE USD/kWh, curtailment %, RE %")
    for sc, d in (("deterministic (1sc)", one), ("probabilistic (12sc)", twelve)):
        print(f"  {sc}")
        for k in ("base", "3-step", "5-step"):
            v = d[k]
            print(f"    {k:7s} {v['npc'] / 1e3:10,.1f} {v['pv'][-1]:9,.1f} {v['batt'][-1]:9,.0f} {v['gen']:6,.0f} "
                  f"{v['lcoe']:.4f} {v['curt']:6.2f}% {v['re']:6.2f}%")

    print("== Table 6 (single-step sensitivities): NPC kUSD, PV kWp, G  |  12sc: NPC, PV, G  |  premium")
    for k, _, _ in DESIGNS[:8]:
        a, b = one[k], twelve[k]
        print(f"  {k:15s} {a['npc'] / 1e3:9,.1f} {a['pv'][-1]:8,.1f} {a['G']}  | {b['npc'] / 1e3:9,.1f} {b['pv'][-1]:8,.1f} "
              f"{b['G']}  | {(b['npc'] - a['npc']) / 1e3:+8,.1f} kUSD {pct(b['npc'], a['npc']):+6.2f}%")
    print("  change against each base, 1sc / 12sc: NPC %, PV %:")
    for k, _, _ in DESIGNS[1:8]:
        print(f"    {k:15s} NPC {pct(one[k]['npc'], one['base']['npc']):+6.2f} / {pct(twelve[k]['npc'], twelve['base']['npc']):+6.2f}"
              f"   PV {pct(one[k]['pv'][-1], one['base']['pv'][-1]):+6.1f} / {pct(twelve[k]['pv'][-1], twelve['base']['pv'][-1]):+6.1f}"
              f"   battery {one[k]['batt'][-1]:,.0f} / {twelve[k]['batt'][-1]:,.0f} kWh   RE {one[k]['re']:.2f} / {twelve[k]['re']:.2f}%")

    b1, b12 = one["base"], twelve["base"]
    print("== 5.2 premium of the ensemble plan")
    print(f"  1-step: {(b12['npc'] - b1['npc']) / 1e3:,.1f} kUSD ({pct(b12['npc'], b1['npc']):.2f}%); PV {b12['pv'][-1]:,.1f} vs "
          f"{b1['pv'][-1]:,.1f} kWp (+{b12['pv'][-1] - b1['pv'][-1]:,.1f}, {pct(b12['pv'][-1], b1['pv'][-1]):.1f}%); battery "
          f"+{b12['batt'][-1] - b1['batt'][-1]:,.0f} kWh; gensets {b1['G']} and {b12['G']}")
    for k in ("3-step", "5-step"):
        print(f"  {k}: {pct(twelve[k]['npc'], one[k]['npc']):.2f}%")
    m = manifest("_kwh")
    w = np.array(m["probabilistic"]["scenario_weights"])
    d12 = demand("_kwh", "12sc")
    hours = np.einsum("s,sy->y", w, (d12 > 2 * GEN_KW).sum(axis=0))
    print(f"  weighted hours above {2 * GEN_KW:.0f} kW: mean {hours.mean():,.0f} per year, year 20 {hours[-1]:,.0f}")
    tail = d12[:, 11, :]
    print(f"  tail scenario: year-20 peak {tail[:, -1].max():,.1f} kW; 20-year peak {tail.max():,.1f} kW (year {tail.max(axis=0).argmax() + 1})")
    ts = pd.read_excel(glob.glob(str(RES / "*_fixeddesign_certified_integral_gili_ketapang_12sc_20y_MILP_3step_dieselops_dea2025k"))[0]
                       + "/Time_Series_SC_12.xlsx", sheet_name="Year 20", header=None).iloc[5:5 + T]
    ts = ts.apply(pd.to_numeric, errors="coerce")
    h = int(ts[1].to_numpy().argmax())
    rating = twelve["3-step"]["batt"][-1] / 4
    print(f"  3-step ensemble design at that peak (hour {h}): demand {ts[1].iloc[h]:,.1f}, PV {ts[2].iloc[h]:,.1f}, gensets "
          f"{ts[3].iloc[h]:,.1f}, battery {ts[7].iloc[h]:,.1f} kW ({100 * ts[7].iloc[h] / rating:.0f}% of its {rating:,.0f} kW rating)")

    print("== 5.3 the ensemble plan on the median path, and the reverse")
    base, cross = cost_sheet(FOLDER_1SC_BASE), cost_sheet(FOLDER_CROSS)
    for key in [("Weighted Net present cost", "System"), ("Investment cost", "Solar PV"), ("Investment cost", "Battery bank"),
                ("Fixed cost", "Solar PV"), ("Fixed cost", "Battery bank"), ("Fixed cost", "Diesel Genset 1"),
                ("Total variable O&M cost", "System"), ("Replacement cost", "Battery bank"), ("Fuel cost", "Diesel"),
                ("Salvage value", "System")]:
        print(f"  {key[0]:26s} {key[1]:16s} median-sized {base[key]:10,.3f}  ensemble-sized {cross[key]:10,.3f}  "
              f"difference {cross[key] - base[key]:+10,.3f} kUSD")
    print(f"  RE {b1['re']:.2f}% -> 98.91%, curtailment {b1['curt']:.2f}% -> 26.58% (the cross-check's export)")

    print("== 5.4 staging")
    for sc, d in (("1sc", one), ("12sc", twelve)):
        print(f"  {sc}: 3-step {pct(d['3-step']['npc'], d['base']['npc']):+.2f}%, 5-step {pct(d['5-step']['npc'], d['base']['npc']):+.2f}% "
              f"vs 1-step; 5-step {pct(d['5-step']['npc'], d['3-step']['npc']):+.2f}% vs 3-step")
        cum = lambda x: np.cumsum(x[:-1])
        print(f"    3-step cumulative PV {', '.join(f'{v:,.1f}' for v in cum(d['3-step']['pv']))} kWp; battery "
              f"{', '.join(f'{v:,.0f}' for v in cum(d['3-step']['batt']))} kWh; curtailment 1-step {d['base']['curt']:.2f}%, "
              f"3-step {d['3-step']['curt']:.2f}%")

    print("== 5.5 sensitivities: the median path and the growth calibration")
    d1 = demand("_kwh", "1sc")[:, 0, :]
    yp = int(d1.max(axis=0).argmax())
    print(f"  median path: peak {d1.max():,.1f} kW in year {yp + 1} ({d1[:, yp].sum() / 1e3:,.1f} MWh that year); "
          f"hours in (470, 587.5] kW: {int(((d1 > 470) & (d1 <= 587.5)).sum()):,} of {d1.size:,}; peak - 940 kW = {d1.max() - 940:.1f} kW")
    g = manifest("_kwh_g24h")["probabilistic"]["peak_y20_percentiles_kw"]
    p = m["probabilistic"]["peak_y20_percentiles_kw"]
    tail_g = demand("_kwh_g24h", "12sc")[:, 11, -1].max()
    print(f"  growth 2014-23 -> 2018-23: ensemble year-20 P50 {p['50']['ensemble']:,.0f} -> {g['50']['ensemble']:,.0f} kW, "
          f"P95 {p['95']['ensemble']:,.0f} -> {g['95']['ensemble']:,.0f} kW; tail scenario year-20 peak "
          f"{tail[:, -1].max():,.1f} -> {tail_g:,.1f} kW ({pct(tail_g, tail[:, -1].max()):+.1f}%)")

    print("== Table 4 (reduction checks); in parentheses the 11-scenario set without the tail stratum")
    n = manifest("_kwh_notail")["probabilistic"]
    for q in ("10", "50", "90", "95"):
        print(f"  year-20 peak P{q}: ensemble {p[q]['ensemble']:,.0f}, reduced {p[q]['reduced']:,.0f} ({n['peak_y20_percentiles_kw'][q]['reduced']:,.0f})")
    for label, src in (("12sc", m), ("11sc no tail", manifest("_kwh_notail"))):
        pr, dg = src["probabilistic"], src["diagnostics"]
        print(f"  {label}: max annual-energy error {max(abs(x) for x in pr['energy_error_pct_by_year']):.2f}%; year-1 monthly "
              f"energy vs backbone MAPE {dg['year1_weighted_window_mape_pct']:.2f}%; silhouette {pr['silhouette']:.3f}; "
              f"bootstrap ARI {pr['bootstrap_ari_mean']:.2f}")
    res = pd.read_csv(INP / "RES_Time_Series_12sc_gili_ketapang_merged_kwh.csv", index_col=0).to_numpy()
    y = res.sum(axis=0) * m["settings"]["res_inverter_eff"] / m["settings"]["res_unit_kw"]
    print(f"  PV yield of the 12 scenarios: mean {y.mean():,.0f}, range {y.min():,.0f}-{y.max():,.0f} kWh/kWp/yr "
          f"(ensemble mean {m['diagnostics']['pv_kwh_per_kw_mean']:,.0f})")

    more_numbers(one, twelve, m, w, d12)


# ---------------------------------------------------------------- added 2026-09-25 (second pass)
def solve_stats(path):
    """Solve time, threads and peak RSS of one log: sizing 'Explored ... in X seconds' (last), certification
    '[timing] relaxed/integral: ... solve N s', 'Thread count was N', last 'peak so far N MB'."""
    t = Path(path).read_text(encoding="utf-8", errors="replace")
    grab = lambda pat: [float(x) for x in re.findall(pat, t)]
    return dict(explored=grab(r"Explored \d+ nodes .*? in ([\d.]+) seconds"),
                relaxed=grab(r"\[timing\] relaxed: .*?solve (\d+) s"),
                integral=grab(r"\[timing\] integral: .*?solve (\d+) s"),
                threads=grab(r"Thread count was (\d+)"), rss=grab(r"peak so far (\d+) MB"))


def cert_gap(path):
    """(integral NPC, best bound, relaxed NPC) of a certification log, on the NPC basis."""
    t = Path(path).read_text(encoding="utf-8", errors="replace")
    m = re.search(r"linopy integral NPC: ([\d,.]+)\s+\(\w+, best bound ([\d,.]+)\)", t)
    r = re.search(r"linopy relaxed NPC : ([\d,.]+)", t)
    f = lambda s: float(s.replace(",", ""))
    return f(m.group(1)), f(m.group(2)), f(r.group(1))


def more_numbers(one, twelve, m, w, d12):
    pct = lambda a, b: 100 * (a / b - 1)
    pr, dg = m["probabilistic"], m["diagnostics"]

    print("\n== 3.5 reduction and the median path (Figure 2)")
    order = np.argsort(w)[::-1]
    print(f"  weights: heaviest {w.max():.3f} (scenario {order[0] + 1}); at the 0.01 floor: scenarios "
          f"{', '.join(str(i + 1) for i in np.flatnonzero(w <= 0.01 + 1e-9))}; range {w.min():.3f}-{w.max():.3f}")
    tail = pr["upper_tail"]
    print(f"  tail member {tail['member']} ({tail['rule']}): 20-year peak {tail['peak_20y_kw']:,.1f} kW, "
          f"20-year energy {tail['energy_20y_mwh']:,.0f} MWh")
    e_full = dg["energy_mean_mwh_full"]
    print(f"  ensemble mean annual energy: year 1 {e_full[0]:,.1f}, year 20 {e_full[-1]:,.1f} MWh")
    d1 = demand("_kwh", "1sc")[:, 0, :]
    pk = d1.max(axis=0)
    print(f"  median path: peak year 1 {pk[0]:,.1f} kW, year 20 {pk[-1]:,.1f} kW, horizon maximum {pk.max():,.1f} kW "
          f"(year {pk.argmax() + 1}); year-20 energy {d1[:, -1].sum() / 1e3:,.1f} MWh "
          f"({pct(d1[:, -1].sum() / 1e3, e_full[-1]):+.2f}% vs the ensemble mean)")
    print(f"  median path cumulative growth factor, year 20: {m['deterministic']['growth_factor_p50'][-1]:.3f}")

    print("\n== 3.1 outage hours: backbone demand on the hours dropped in cleaning")
    import sys
    sys.path.insert(0, str(INP))
    saved = sys.argv
    sys.argv = sys.argv[:1]
    import _build_demand_merged as b                     # noqa: E402  (its own loader and backbone)
    sys.argv = saved
    b.LEVEL_SOURCE = "kwh"
    path = Path(b.LF_DIR) / "load data" / "harian_gilket_gabungan.csv"
    raw = pd.read_csv(path, parse_dates=["tanggal"])
    oh = raw.loc[b.rb._flag_outage_rows(raw), "tanggal"].dt.round("h").drop_duplicates()
    shape = b.rb.extract_daily_shape(b.load_genset_log_rounded(str(path)))
    idx = pd.date_range("2026-01-01", periods=T, freq="h")
    mb = pd.Series(b.backbone(idx, shape, b.kwh_levels(str(Path(b.LF_DIR) / "load data" / "produksi_kwh_harian.csv"))),
                   index=idx)
    key = pd.DatetimeIndex(oh).map(lambda t: t.replace(year=2026))
    print(f"  {len(oh)} outage hours ({(oh.dt.year == 2026).sum()} in 2026): {mb.reindex(key).sum():,.1f} kWh of "
          f"backbone demand; backbone year {mb.sum() / 1e3:,.1f} MWh")

    print("\n== 4 autonomy rule coverage (single-step ensemble battery)")
    batt, dod = twelve["base"]["batt"][-1], 0.8
    daily = np.array([w @ d12[:, :, y].sum(axis=0) / 365 for y in range(Y)])
    print(f"  {batt:,.0f} kWh x {dod} covers {batt * dod / daily[0]:.2f} days of year-1 and "
          f"{batt * dod / daily[-1]:.2f} days of year-20 weighted mean daily demand "
          f"(horizon mean {daily.mean():,.0f} kWh/day, rule {daily.mean() / dod:,.0f} kWh)")

    print("\n== 4 certification gaps (NPC basis) and whole-unit corrections, the paper's designs")
    rows = []
    for sc, lanes in (("1sc", "lanes_dea2025k_1sc_laptop"), ("12sc", "lanes_dea2025k_12sc_*")):
        for k, st, sf in DESIGNS:
            f = glob.glob(str(LOGS / lanes / f"gili_ketapang_{sc}_20y_MILP_{st}_dieselops_dea2025k{sf}_cert.log"))[0]
            npc, bound, relaxed = cert_gap(f)
            rows.append((f"{sc} {k}", 100 * (npc - bound) / npc, 100 * (npc - relaxed) / npc))
    for name, g, c in rows:
        print(f"  {name:20s} gap {g:.3f}%  integral - relaxed {c:+.3f}%")
    rest = [r for r in rows if "D=0" not in r[0]]
    print(f"  without zero autonomy: max gap {max(r[1] for r in rest):.3f}%, max correction {max(r[2] for r in rest):.3f}%")

    print("\n== 4 computational setup (12sc logs of the paper's designs)")
    size = [f for f in glob.glob(str(LOGS / "lanes_dea2025k_12sc_*" / "*_size_*.log")) if "_tail" not in f]
    cert = [f for f in glob.glob(str(LOGS / "lanes_dea2025k_12sc_*" / "*_cert.log")) if "_tail" not in f]
    stats = lambda fs: {Path(f).name: solve_stats(f) for f in fs}
    s, c = stats(size), stats(cert)
    one_step = {k: v["explored"][-1] for k, v in s.items() if "_1step_" in k}
    for k, v in sorted(s.items()):
        print(f"  size {k[len('gili_ketapang_12sc_20y_MILP_'):-4]:48s} {v['explored'][-1]:8,.0f} s  "
              f"threads {v['threads'][-1]:.0f}  peak RSS {max(v['rss']) / 1e3:5.1f} GB")
    print(f"  1-step sizing {min(one_step.values()) / 60:.0f}-{max(one_step.values()) / 60:.0f} min per genset count")
    for k, v in sorted(c.items()):
        print(f"  cert {k[len('gili_ketapang_12sc_20y_MILP_'):-4]:48s} relaxed {v['relaxed'][-1]:6,.0f} s  "
              f"integral {v['integral'][-1]:6,.0f} s  threads {v['threads'][-1]:.0f}  peak RSS {max(v['rss']) / 1e3:5.1f} GB")
    rel = [v["relaxed"][-1] for v in c.values()]
    itg = [v["integral"][-1] for v in c.values()]
    print(f"  certification: relaxed {min(rel) / 3600:.1f}-{max(rel) / 3600:.1f} h, integral {min(itg) / 60:.0f}-"
          f"{max(itg) / 60:.0f} min; peak RSS of sizing and certification up to "
          f"{max(max(v['rss']) for v in list(s.values()) + list(c.values())) / 1e3:.1f} GB")
    for f in sorted(glob.glob(str(LOGS / "queue_20260925_*" / "0*.log"))):
        v = solve_stats(f)
        t = Path(f).read_text(encoding="utf-8", errors="replace")
        mf = re.findall(r"Set parameter MIPFocus to value (\d)", t)
        print(f"  queue {Path(f).parent.name}/{Path(f).name[:3]}{Path(f).name[-8:-4]}: {v['explored'][-1]:,.0f} s, "
              f"MIPFocus {mf[-1] if mf else '-'}")

    print("\n== 5.2 tail representation (AWO-002): certified 12sc designs and the least PV of the relaxation")
    import _awo002c_binding_window as a                  # noqa: E402  (same directory)
    base_npc = twelve["base"]["npc"]
    for label, dsuf, rsuf, site_suf, res_u, batt_kwh, gen_u in a.DESIGNS:
        st = m if dsuf == "_kwh" else manifest(dsuf)
        unit_kw, inv = st["settings"]["res_unit_kw"], st["settings"]["res_inverter_eff"]
        dem = demand(dsuf, "12sc").transpose(1, 2, 0)[a.TAIL]           # (Y, T)
        res = pd.read_csv(INP / f"RES_Time_Series_12sc_gili_ketapang_merged{rsuf}.csv", index_col=0).to_numpy()
        pvk = res[:, a.TAIL] * inv / unit_kw
        need = a.min_pv(batt_kwh, gen_u * a.GEN_KW, dem, pvk)
        v = cert_log("12sc", "1step", site_suf)
        print(f"  {label:34s} yield {pvk.sum():7,.1f}  PV {v['pv'][-1]:7,.1f}  least PV {need:7,.1f} "
              f"({v['pv'][-1] / need:.2f}x)  NPC {v['npc'] / 1e3:8,.0f} kUSD ({pct(v['npc'], base_npc):+5.1f}%)")

    print("\n== 5.5 zero autonomy: the 470-587.5 kW band on the median path")
    pv1 = pd.read_csv(INP / "RES_Time_Series_1sc_gili_ketapang_merged_kwh.csv", index_col=0).to_numpy()[:, 0]
    band = (d1 > 470) & (d1 <= 587.5)
    print(f"  hours in the band: {int(band.sum()):,} of {d1.size:,}; with PV at zero (night): "
          f"{int((band & (pv1 == 0)[:, None]).sum()):,}")


if __name__ == "__main__":
    main()
