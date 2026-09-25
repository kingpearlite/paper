"""The `dea2025k` design family (2026-09-24): the `dea2025c` design set on the rebuilt demand.

Each of the 20 `dea2025c` designs (1sc and 12sc; 1-step base, _dr10, _dr08, _aut0, _aut2, _fuel064,
_fuel107, _g24h; 3- and 5-step) is copied as `..._dea2025k[...]` with only its demand inputs changed:
the demand and RES files, the 12sc Scenario_Weight block, and SITE_MAX_BATTERY_KWH (the same multiple,
1x or 2x, of 1.1 x the largest daily demand energy as before). Costs, steps, discount rate, autonomy,
fuel price and solver settings are carried over unchanged. The demand comes from
_build_demand_merged.py with MGPY_LEVEL_SOURCE=kwh MGPY_TAKSR_DEV=abs MGPY_TAIL_RULE=growth_medoid
(suffix _kwh, and _kwh_g24h for the growth variant); the build commands are in its docstring.

Seven more 12sc 1-step designs carry the review's AWO-002 sensitivity on the base design:
  _tailq25, _tailq75           the upper-tail scenario replaced by the stratum's 25th / 75th
                               percentile member by 20-year peak (own demand and weights; A)
  _tailpv{min,p25,p50,p75,max} the base demand, with the tail scenario's PV year replaced by the PV
                               member at that annual-yield quantile (B)

Run from Code/Inputs with any Python 3. An existing output with identical content is skipped; a
different one is never overwritten.
"""
import json
import os
import re
import sys

import _build_dea2025_designs as base

VARIANTS_1STEP = ("", "_dr10", "_dr08", "_aut0", "_aut2", "_fuel064", "_fuel107", "_g24h")
SOURCES = [f"gili_ketapang_{sc}_20y_MILP_1step_dieselops_dea2025c{v}" for sc in ("1sc", "12sc") for v in VARIANTS_1STEP]
SOURCES += [f"gili_ketapang_{sc}_20y_MILP_{n}step_dieselops_dea2025c" for sc in ("1sc", "12sc") for n in (3, 5)]
AWO_BASE = "gili_ketapang_12sc_20y_MILP_1step_dieselops_dea2025c"
PV_REPAIRS = ("min", "p25", "p50", "p75", "max")
NOTE = ("dea2025k: demand level from the daily kWh production, TAKSR mean-absolute-deviation feature, "
        "growth-medoid tail representative")


def summaries(site):
    """(old, new) summary JSONs for a dea2025c source site."""
    g = "_g24h" if site.endswith("_g24h") else ""
    return f"merged_scenarios_gili_ketapang{g}.json", f"merged_scenarios_gili_ketapang_kwh{g}.json"


def scheme_of(summary, label):
    return base.load_schemes(summary)[label]


def conf_value(conf_text, key):
    m = re.search(rf"^{key}=(.*)$", conf_text, flags=re.M)
    if not m:
        sys.exit(f"no {key} line")
    return m.group(1).strip()


def transform(conf_text, dat_text, label, old, new, summary, extra_note="", res=None):
    """Swap the demand inputs of one design from scheme `old` to scheme `new`."""
    res = res or new["res"]
    if conf_value(conf_text, "SITE_DEMAND") != old["demand"] or conf_value(conf_text, "SITE_RES") != old["res"]:
        sys.exit(f"conf does not use the expected source files {old['demand']} / {old['res']}")
    old_ceiling = int(conf_value(conf_text, "SITE_MAX_BATTERY_KWH"))
    factor = old_ceiling / base.battery_ceiling_kwh(old)
    if factor not in (1.0, 2.0):
        sys.exit(f"battery ceiling {old_ceiling} is {factor:.3f} x the base bound, expected 1 or 2")
    ceiling = int(factor) * base.battery_ceiling_kwh(new)

    weight_lines = "\n".join(f"{i}      {v:.6f}" for i, v in enumerate(new["weights"], 1))
    kind = "TAKSR representatives + upper-tail scenario, moment-matched weights" if label != "1sc" else \
        "single deterministic scenario, Monte Carlo P50 growth path"
    dat = base.sub_once(dat_text, r"^param: Scenario_Weight\s*:=[^\n]*\n(?:\d+\s+[0-9.]+\n)*\d+\s+[0-9.]+;",
                        f"param: Scenario_Weight :=  # {kind} ({summary})\n{weight_lines};", "Scenario_Weight")
    for a, b in ((old["demand"], new["demand"]), (old["res"], res)):
        if dat.count(a) != 1:
            sys.exit(f"expected {a} exactly once in the .dat comments, found {dat.count(a)}")
        dat = dat.replace(a, b)

    conf = base.sub_once(conf_text, r"^SITE_DEMAND=.*$", f"SITE_DEMAND={new['demand']}", "SITE_DEMAND")
    conf = base.sub_once(conf, r"^SITE_RES=.*$", f"SITE_RES={res}", "SITE_RES")
    conf = base.sub_once(conf, r"^SITE_MAX_BATTERY_KWH=.*$", f"SITE_MAX_BATTERY_KWH={ceiling}", "SITE_MAX_BATTERY_KWH")
    conf = base.sub_once(conf, r"largest daily demand energy \(\d+ kWh\)",
                         f"largest daily demand energy ({new['max_daily_kwh']:.0f} kWh)", "ceiling comment")
    conf = conf.replace("\n", f"\n# {NOTE} ({summary}){extra_note}\n", 1)
    return conf, dat


def read(path):
    with open(path, encoding="utf-8", newline="") as f:
        return f.read()


def main():
    outputs = []

    def add(site, dat_name, conf, dat):
        outputs.append((os.path.join(base.HERE, dat_name), dat))
        outputs.append((os.path.join(base.SITES, site + ".conf"), conf))

    for site in SOURCES:
        conf_text = read(os.path.join(base.SITES, site + ".conf"))
        src_dat = conf_value(conf_text, "SITE_PARAMS")
        label = "1sc" if "_1sc_" in site else "12sc"
        old_sum, new_sum = summaries(site)
        conf, dat = transform(conf_text, read(os.path.join(base.HERE, src_dat)), label,
                              scheme_of(old_sum, label), scheme_of(new_sum, label), new_sum)
        new_dat = src_dat.replace("dieselops_dea2025c", "dieselops_dea2025k", 1)
        conf = base.sub_once(conf, r"^SITE_PARAMS=.*$", f"SITE_PARAMS={new_dat}", "SITE_PARAMS")
        add(site.replace("dieselops_dea2025c", "dieselops_dea2025k", 1), new_dat, conf, dat)

    # AWO-002 on the 12sc 1-step base design
    conf_text = read(os.path.join(base.SITES, AWO_BASE + ".conf"))
    src_dat = conf_value(conf_text, "SITE_PARAMS")
    dat_text = read(os.path.join(base.HERE, src_dat))
    old = scheme_of("merged_scenarios_gili_ketapang.json", "12sc")
    variants = []
    for q in ("q25", "q75"):
        s = f"merged_scenarios_gili_ketapang_kwh_tail{q}.json"
        ut = json.loads(read(os.path.join(base.HERE, s)))["probabilistic"]["upper_tail"]
        note = (f"; AWO-002 A: tail = stratum {q.upper()} member {ut['member']} by 20-year peak "
                f"({ut['peak_20y_kw']:.0f} kW)")
        variants.append((f"_tail{q}", s, scheme_of(s, "12sc"), None, note))
    s = "merged_scenarios_gili_ketapang_kwh.json"
    repairs = json.loads(read(os.path.join(base.HERE, s)))["probabilistic"]["upper_tail"]["tail_pv_repairs"]
    for k in PV_REPAIRS:
        r = repairs[k]
        note = (f"; AWO-002 B: tail PV re-paired with member {r['pv_member']} "
                f"({r['pv_kwh_per_kw']:.1f} kWh/kW, annual-yield {k})")
        variants.append((f"_tailpv{k}", s, scheme_of(s, "12sc"), r["res"], note))
    for suffix, summary, new, res, note in variants:
        conf, dat = transform(conf_text, dat_text, "12sc", old, new, summary, note, res)
        new_dat = src_dat.replace("dieselops_dea2025c", "dieselops_dea2025k", 1).replace(".dat", f"{suffix}.dat")
        conf = base.sub_once(conf, r"^SITE_PARAMS=.*$", f"SITE_PARAMS={new_dat}", "SITE_PARAMS")
        add(AWO_BASE.replace("dieselops_dea2025c", "dieselops_dea2025k", 1) + suffix, new_dat, conf, dat)

    clash = [p for p, text in outputs if os.path.exists(p) and read(p) != text]
    if clash:
        sys.exit("Refusing to overwrite (content differs):\n  " + "\n  ".join(clash))
    for path, text in outputs:
        rel = os.path.relpath(path, os.path.join(base.HERE, "..", ".."))
        if os.path.exists(path):
            print("unchanged", rel)
            continue
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        print("wrote", rel)
    print(f"{len(outputs) // 2} designs")


if __name__ == "__main__":
    main()
