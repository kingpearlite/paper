"""Builds the paper-update designs (2026-09-11): {1sc deterministic, 12sc probabilistic} demand
schemes from _build_demand_merged.py x {1, 3, 5 investment steps} at the 12% base discount rate,
plus 8%/10% discount-rate sensitivity variants of the 1-step designs (10 designs), 20-year horizon,
dieselops formulation, costs from the DEA/MEMR "Technology Data for the Indonesian Power Sector"
(March 2024) at reference year 2025, fuel from Statistik PLN 2025.

Each .dat is the certified 9sc capexp5 dieselops .dat with ONLY the lines below replaced -- costs,
technology, steps, discount rate, Scenarios/Scenario_Weight and the demand/RES source comments;
every replacement must match exactly once or the script stops. Each sites/*.conf copies the
certified 9sc capexp5 dieselops conf (Gurobi default Crossover=1), with SITE_PARAMS/DEMAND/RES/
SCENARIOS set for the scheme and SITE_MAX_BATTERY_KWH = 1.1 x the scheme's largest daily demand
energy (the old ceilings, 62,031 and 74,675 kWh, were about 1x the old demand files' largest daily
energy). Scenario weights and daily-energy maxima come from merged_scenarios_gili_ketapang.json, so
run _build_demand_merged.py first.

2025 values: v2025 = v2023 + (2/7) * (v2030 - v2023), linear between the catalogue's 2023 and 2030
columns. Catalogue costs are USD, price year 2022 (as stated on each data sheet's intro page).

Number of steps = ceil(Years / Step_Duration) in both Initialize_Upgrades_Number (Pyomo) and
DesignInputs.STEPS_NUMBER (linopy), so 3 steps needs Step_Duration=7 (7+7+6 years); the older
capexp6 designs (Step_Duration=6) have 4 steps (6+6+6+2), not 3.

Usage (from Code/Inputs/):  python _build_dea2025_designs.py
Refuses to overwrite existing outputs.
"""
import json
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
SITES = os.path.normpath(os.path.join(HERE, "..", "..", "sites"))
MERGED_SUMMARY = "merged_scenarios_gili_ketapang.json"
TEMPLATE_DAT = "Parameters_9sc_20y_gili_ketapang_capexp5_MILP_dieselops.dat"
TEMPLATE_CONF = "gili_ketapang_9sc_20y_MILP_capexp5_dieselops.conf"
BATTERY_CEILING_FACTOR = 1.1


def interp2025(v2023, v2030):
    return v2023 + (2.0 / 7.0) * (v2030 - v2023)


# ---- DEA 2024 data sheets (printed page numbers) ----
# PV industrial scale, rooftop or ground mounted, 100 kWe (p.70)
PV_CAPEX = interp2025(1.08, 0.76) * 1000.0          # M$/MWe -> $/kW: 988.57
PV_FOM = interp2025(4700.0, 4200.0)                 # $/MWe/yr: 4557.14
PV_OM_FRAC = PV_FOM / (PV_CAPEX * 1000.0)           # 0.004610
PV_LIFE = round(interp2025(27, 30))                 # 27.86 -> 28 (Pyomo Param is integer)
# Diesel engine using fuel oil (p.225): 3 MWe units, 14 MWe plant
GEN_CAPEX = interp2025(0.91, 0.91) * 1000.0         # 910 $/kW
GEN_FOM = interp2025(9120.0, 9120.0)                # $/MWe/yr
GEN_OM_FRAC = GEN_FOM / (GEN_CAPEX * 1000.0)        # 0.010022
GEN_LIFE = 25
# Batteries, lithium-ion, utility scale, 4 MWh / 4 h (p.265)
BATT_HOURS = 4.0
BATT_TOTAL = interp2025(0.47, 0.33) * 1000.0        # MUSD/MWh -> $/kWh: 430.00 (4 h system)
BATT_POWER_PER_KWH = interp2025(0.33, 0.29) * 1000.0 / BATT_HOURS   # 318.57 $/kW / 4 h = 79.64
BATT_FOM_PER_KWH = interp2025(15000.0, 10500.0) / 1000.0 / BATT_HOURS  # $/MW/yr -> $/kWh/yr: 3.4286
BATT_OM_FRAC = BATT_FOM_PER_KWH / BATT_TOTAL        # 0.007973
BATT_RTE = interp2025(0.90, 0.92)                   # AC round trip: 0.905714
BATT_ONEWAY = BATT_RTE ** 0.5                       # 0.951690, same for charge and discharge
BATT_CYCLES = round(interp2025(7500, 10000))        # 8214
# ---- Fuel: Statistik PLN 2025, Tabel 25, UID Jawa Timur, HSD ----
FUEL_IDR_PER_KL = 14059091.89
IDR_PER_USD = 16400.0                               # BI 2025 average (user decision)
FUEL_USD_PER_L = FUEL_IDR_PER_KL / 1000.0 / IDR_PER_USD   # 0.857262
# ---- Discount rate ----
# Base 12%: the WACC the same catalogue uses to back-calculate CAPEX from ESDM's proposed FIT levels
# (Solar PV chapter cost table footnote, PDF p.61; bioenergy chapter footnote, PDF p.195). The
# catalogue does not say whether it is real or nominal; the model treats it as real.
# Sensitivity 8% and 10% on the 1-step designs only (files suffixed _dr08 / _dr10).
DISCOUNT_BASE = 0.12
DISCOUNT_SENSITIVITY = (0.08, 0.10)
SENSITIVITY_STEPS = (1,)

SRC = "DEA/MEMR Technology Data Indonesia 2024"
SCALAR = {
    "Battery_Specific_Investment_Cost": (f"{BATT_TOTAL:.2f}",
        f"{SRC} p.265 Li-ion 4 h total investment, 2025 interp. of 0.47/0.33 MUSD/MWh (USD2022)"),
    "Battery_Specific_Electronic_Investment_Cost": (f"{BATT_POWER_PER_KWH:.2f}",
        "Non-replaceable part = p.265 power component 318.57 $/kW (2025) / 4 h; excluded from the wear cost"),
    "Battery_Specific_OM_Cost": (f"{BATT_OM_FRAC:.6f}",
        "p.265 fixed O&M 13,714 USD/MW/yr (2025) / 4 h = 3.43 $/kWh/yr, as fraction of 430 $/kWh"),
    "Battery_Discharge_Battery_Efficiency": (f"{BATT_ONEWAY:.4f}",
        "sqrt of p.265 AC round-trip efficiency 90.57% (2025)"),
    "Battery_Charge_Battery_Efficiency": (f"{BATT_ONEWAY:.4f}",
        "sqrt of p.265 AC round-trip efficiency 90.57% (2025)"),
    "Maximum_Battery_Discharge_Time": (f"{BATT_HOURS:.1f}", "p.265 discharge time, matches the 4 h cost basis"),
    "Maximum_Battery_Charge_Time": (f"{BATT_HOURS:.1f}", "p.265 discharge time, matches the 4 h cost basis"),
    "Battery_Cycles": (f"{BATT_CYCLES:.1f}", "p.265 technical lifetime in cycles, 2025 interp. of 7,500/10,000"),
}
INDEXED = {
    "RES_Specific_Investment_Cost": (f"{PV_CAPEX:.2f}",
        f"{SRC} p.70 PV industrial 100 kW, 2025 interp. of 1.08/0.76 M$/MWe (USD2022)"),
    "RES_Specific_OM_Cost": (f"{PV_OM_FRAC:.6f}", "p.70 fixed O&M 4,557 $/MWe/yr (2025) as fraction of capex"),
    "RES_Lifetime": (f"{PV_LIFE}", "p.70 technical lifetime 27.86 y (2025), rounded (integer Param)"),
    "Generator_Specific_Investment_Cost": (f"{GEN_CAPEX:.2f}",
        f"{SRC} p.225 diesel engine (fuel oil), 0.91 M$/MWe 2023 and 2030 (USD2022)"),
    "Generator_Specific_OM_Cost": (f"{GEN_OM_FRAC:.6f}",
        "p.225 fixed O&M 9,120 $/MWe/yr as fraction of capex; variable O&M 7.17 $/MWh has no model field, omitted"),
    "Generator_Lifetime": (f"{GEN_LIFE}", "p.225 technical lifetime"),
    "Fuel_Specific_Start_Cost": (f"{FUEL_USD_PER_L:.4f}",
        "Statistik PLN 2025 Tabel 25, UID Jawa Timur HSD Rp 14,059,091.89/kL (incl. transport) / Rp 16,400 per USD (BI 2025 avg)"),
}
STEPS = {1: 20, 3: 7, 5: 4}  # number of investment steps -> Step_Duration (20-year horizon)


def load_schemes(summary=MERGED_SUMMARY):
    """The two demand schemes, from the summary _build_demand_merged.py writes."""
    with open(os.path.join(HERE, summary), encoding="utf-8") as f:
        s = json.load(f)
    out = s["outputs"]
    prob_weights = s["probabilistic"]["scenario_weights"]
    return {
        "1sc": dict(kind="DETERMINISTIC", demand=out["demand_det"], res=out["res_det"], weights=[1.0],
                    max_daily_kwh=s["deterministic"]["max_daily_energy_kwh"],
                    weight_note="single deterministic scenario, Monte Carlo P50 growth path"),
        f"{len(prob_weights)}sc": dict(
            kind="PROBABILISTIC", demand=out["demand_prob"], res=out["res_prob"], weights=prob_weights,
            max_daily_kwh=s["probabilistic"]["max_daily_energy_kwh"],
            weight_note="TAKSR representatives + upper-tail scenario, moment-matched weights "
                        "(merged_scenarios_gili_ketapang.json)"),
    }


def battery_ceiling_kwh(scheme):
    return int(math.ceil(BATTERY_CEILING_FACTOR * scheme["max_daily_kwh"] / 10.0) * 10)


def sub_once(text, pattern, repl, what):
    new, n = re.subn(pattern, repl, text, flags=re.M)
    if n != 1:
        sys.exit(f"{what}: expected exactly 1 match, found {n} -- template changed, refusing to guess")
    return new


def build_dat(template_text, scheme, n_steps, discount):
    t = template_text
    step_dur = STEPS[n_steps]
    w = scheme["weights"]
    t = sub_once(t, r"^param: Scenarios\s*:=\s*\d+;.*$",
                 f"param: Scenarios := {len(w)};                          # {scheme['kind'].lower()} demand "
                 f"scheme, _build_demand_merged.py (paper update)", "Scenarios")
    weight_lines = "\n".join(f"{i}      {v:.6f}" for i, v in enumerate(w, 1))
    t = sub_once(t, r"^param: Scenario_Weight\s*:=[^\n]*\n(?:\d+\s+[0-9.]+\n)*\d+\s+[0-9.]+;",
                 f"param: Scenario_Weight :=  # {scheme['weight_note']}\n{weight_lines};", "Scenario_Weight")
    t = sub_once(t, r"^param: RE_Supply_Calculation\s*:=\s*0;.*$",
                 f"param: RE_Supply_Calculation := 0;         # Pre-generated {scheme['res']} "
                 f"(_build_demand_merged.py), not a live per-run NASA fetch", "RE_Supply_Calculation")
    t = sub_once(t, r"^param: Demand_Profile_Generation\s*:=\s*0;.*$",
                 f"param: Demand_Profile_Generation := 0;     # Pre-generated {scheme['demand']} "
                 f"(_build_demand_merged.py), not the archetype generator", "Demand_Profile_Generation")
    note = ("DEA 2024 catalogue WACC (CAPEX back-calculation from ESDM FIT, PDF p.61/p.195), used as real rate"
            if discount == DISCOUNT_BASE else
            f"SENSITIVITY variant (base case {DISCOUNT_BASE:.2f}, see _build_dea2025_designs.py)")
    t = sub_once(t, r"^param: Real_Discount_Rate\s*:=\s*[^;]*;.*$",
                 f"param: Real_Discount_Rate := {discount:.2f};         # {note}", "Real_Discount_Rate")
    t = sub_once(t, r"^param: Step_Duration\s*:=\s*[^;]*;.*$",
                 f"param: Step_Duration := {step_dur};                      # {n_steps} investment step(s): "
                 f"number of steps = ceil(Years / Step_Duration) (paper update 2026-09-11)", "Step_Duration")
    for name, (val, note) in SCALAR.items():
        t = sub_once(t, rf"^param: {name}\s*:=\s*[^;]*;.*$",
                     f"param: {name} := {val};    # {note}", name)
    for name, (val, note) in INDEXED.items():
        t = sub_once(t, rf"^param: {name}\s*:=.*\n1\t[^\n]*$",
                     f"param: {name} :=  # {note}\n1\t{val}", name)
    return t


def build_conf(template_conf_text, label, scheme, n_steps, dat_name, discount):
    role = "base case" if discount == DISCOUNT_BASE else "SENSITIVITY variant"
    ceiling = battery_ceiling_kwh(scheme)
    overrides = {
        "SITE_PARAMS": dat_name, "SITE_DEMAND": scheme["demand"], "SITE_RES": scheme["res"],
        "SITE_SCENARIOS": str(len(scheme["weights"])), "SITE_MAX_BATTERY_KWH": str(ceiling),
    }
    header = [
        f"# Site configuration: Gili Ketapang -- {label} x 20y, MILP, {scheme['kind']}, {n_steps} investment step(s)",
        f"# (Step_Duration={STEPS[n_steps]}), dieselops, DEA 2024 costs at 2025 + PLN 2025 UID Jatim fuel.",
        f"# Real_Discount_Rate={discount:.2f} ({role}). Demand/PV: merged Load Forecasting + DataScarce",
        "# scenarios (_build_demand_merged.py). Generated by Code/Inputs/_build_dea2025_designs.py.",
        f"# Solver/resource settings copied from {TEMPLATE_CONF}.",
        f"# SITE_MAX_BATTERY_KWH = {BATTERY_CEILING_FACTOR} x largest daily demand energy "
        f"({scheme['max_daily_kwh']:.0f} kWh). CHECK AFTER THE RUN: installed",
        "# battery below SITE_MAX_BATTERY_KWH and generator below SITE_MAX_GENERATOR_KW.",
        "",
    ]
    body = []
    for line in template_conf_text.splitlines():
        s = line.strip()
        if not s or s.startswith("#"):
            continue
        key = s.split("=", 1)[0]
        if key in overrides:
            s = f"{key}={overrides.pop(key)}"
        body.append(s)
    if overrides:
        sys.exit(f"template conf lacks {sorted(overrides)} -- refusing to guess")
    return "\n".join(header + body) + "\n"


def main():
    outputs = []
    with open(os.path.join(HERE, TEMPLATE_DAT), encoding="utf-8", newline="") as f:
        dat_text = f.read()
    with open(os.path.join(SITES, TEMPLATE_CONF), encoding="utf-8") as f:
        conf_text = f.read()
    schemes = load_schemes()
    for label, scheme in schemes.items():
        variants = [(n, DISCOUNT_BASE, "") for n in STEPS]
        variants += [(n, dr, f"_dr{round(dr * 100):02d}") for n in SENSITIVITY_STEPS for dr in DISCOUNT_SENSITIVITY]
        for n_steps, dr, suffix in variants:
            dat_name = f"Parameters_{label}_20y_gili_ketapang_{n_steps}step_MILP_dieselops_dea2025{suffix}.dat"
            conf_name = f"gili_ketapang_{label}_20y_MILP_{n_steps}step_dieselops_dea2025{suffix}.conf"
            outputs.append((os.path.join(HERE, dat_name), build_dat(dat_text, scheme, n_steps, dr)))
            outputs.append((os.path.join(SITES, conf_name), build_conf(conf_text, label, scheme, n_steps, dat_name, dr)))
    clash = [p for p, _ in outputs if os.path.exists(p)]
    if clash:
        sys.exit("Refusing to overwrite:\n  " + "\n  ".join(clash))
    for path, text in outputs:
        with open(path, "w", encoding="utf-8", newline="") as f:
            f.write(text)
        print("wrote", os.path.relpath(path, os.path.join(HERE, "..", "..")))
    print(f"\nPV {PV_CAPEX:.2f} $/kW, O&M {PV_OM_FRAC:.6f}, life {PV_LIFE}")
    print(f"Genset {GEN_CAPEX:.2f} $/kW, O&M {GEN_OM_FRAC:.6f}, life {GEN_LIFE}")
    print(f"Battery {BATT_TOTAL:.2f} $/kWh (electronic {BATT_POWER_PER_KWH:.2f}), O&M {BATT_OM_FRAC:.6f}, "
          f"eff {BATT_ONEWAY:.4f} each way, {BATT_CYCLES} cycles, {BATT_HOURS} h")
    print(f"Fuel {FUEL_USD_PER_L:.4f} USD/L")
    for label, scheme in schemes.items():
        print(f"{label}: {len(scheme['weights'])} scenario(s), battery ceiling {battery_ceiling_kwh(scheme)} kWh, "
              f"demand {scheme['demand']}, RES {scheme['res']}")


if __name__ == "__main__":
    main()
