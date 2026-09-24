"""The `dea2025c` design family: the conference paper's design set with corrected cost inputs (2026-09-18).

Applies the three corrections of _build_cost_sensitivity_designs.py's `_costall` variant (PV 822.86
USD/kWp; genset fuel curve B at the 470 kW rating, Generator_Efficiency 0.3874 and Generator_pgen
0.1014; variable O&M 0.00717 USD/kWh genset and 0.001943 USD/kWh battery discharge) to every
`dea2025` design the conference paper reports, and writes each as `..._dea2025c[...]`, so the
original family stays reproducible. Everything else in each design (steps, discount rate, autonomy,
fuel price, growth calibration, demand/PV files, weights, battery ceiling) is carried over unchanged.

Run from Code/Inputs with any Python 3. An existing output with identical content is skipped; a
different one is never overwritten.
"""
import os
import sys

import _build_cost_sensitivity_designs as cs
import _build_dea2025_designs as base

VARIANTS_1STEP = ("", "_dr10", "_dr08", "_aut0", "_aut2", "_fuel064", "_fuel107", "_g24h")
SOURCES = [f"gili_ketapang_{sc}_20y_MILP_1step_dieselops_dea2025{v}" for sc in ("1sc", "12sc") for v in VARIANTS_1STEP]
SOURCES += [f"gili_ketapang_{sc}_20y_MILP_{n}step_dieselops_dea2025" for sc in ("1sc", "12sc") for n in (3, 5)]


def dat_of(conf_text):
    for line in conf_text.splitlines():
        if line.startswith("SITE_PARAMS="):
            return line.split("=", 1)[1].strip()
    sys.exit("no SITE_PARAMS line")


def main():
    outputs = []
    for site in SOURCES:
        with open(os.path.join(base.SITES, site + ".conf"), encoding="utf-8", newline="") as f:
            conf_text = f.read()
        src_dat = dat_of(conf_text)
        with open(os.path.join(base.HERE, src_dat), encoding="utf-8", newline="") as f:
            dat_text = f.read()
        dat, note = cs.make_dat(dat_text, cs.VARIANTS["_costall"])
        new_site = site.replace("dieselops_dea2025", "dieselops_dea2025c", 1)
        new_dat = src_dat.replace("dieselops_dea2025", "dieselops_dea2025c", 1)
        conf = base.sub_once(conf_text, r"^SITE_PARAMS=.*$", f"SITE_PARAMS={new_dat}", "SITE_PARAMS")
        conf = conf.replace("\n", f"\n# dea2025c (corrected cost inputs): {note}\n", 1)
        outputs.append((os.path.join(base.HERE, new_dat), dat))
        outputs.append((os.path.join(base.SITES, new_site + ".conf"), conf))

    def existing(path):
        with open(path, encoding="utf-8", newline="") as f:
            return f.read()
    clash = [p for p, text in outputs if os.path.exists(p) and existing(p) != text]
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
    print(f"{len(SOURCES)} designs")


if __name__ == "__main__":
    main()
