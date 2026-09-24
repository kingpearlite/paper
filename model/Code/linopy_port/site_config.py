"""Config-driven site loader for the linopy port, reusing the SAME `sites/<name>.conf`
convention the Pyomo path already uses (`run_site_interactive.sh`/`submit_site.sh`), instead of
requiring a hand-edited `DESIGNS` dict entry for every new site.

WHY THIS EXISTS: `DesignInputs` (linopy_core_model_builder.py) already reads almost every model
parameter straight out of the real `.dat` file -- the hardcoded `DESIGNS` dicts in
linopy_free_sizing_crosscheck.py/linopy_fixed_design_crosscheck.py/linopy_multiobjective_sweep.py
only ever supplied 5 things: `params_dat`/`demand_csv`/`res_csv` (filenames under Code/Inputs/)
and `max_battery_kwh`/`max_generator_kw` (MILP ceilings with no `.dat` counterpart). All five
already have a 1:1 field in `sites/*.conf`: `SITE_PARAMS`/`SITE_DEMAND`/`SITE_RES`/
`SITE_MAX_BATTERY_KWH`/`SITE_MAX_GENERATOR_KW`. This module parses that same `.conf` format in
pure Python (no shelling out to bash, so it also runs on Windows, not just the HPC/Linux side)
and performs the same validation `run_site_interactive.sh` does before a Pyomo run, so a `.conf`
mismatch fails immediately here too, not silently.

USAGE (from any of the three DESIGNS-consuming scripts):
    LINOPY_SITE=<name>  -- looks up sites/<name>.conf instead of DESIGNS[LINOPY_DESIGN].
"""
import os
import re

_THIS_DIR = os.path.dirname(__file__)
REPO_ROOT = os.path.normpath(os.path.join(_THIS_DIR, "..", ".."))
SITES_DIR = os.path.join(REPO_ROOT, "sites")
INPUTS_DIR = os.path.join(REPO_ROOT, "Code", "Inputs")

_KV_RE = re.compile(r"^([A-Za-z_][A-Za-z0-9_]*)=(.*)$")


class SiteConfigError(ValueError):
    """Raised for anything a new-site author needs to fix in their .conf/.dat, mirroring the
    `die()` failure mode in run_site_interactive.sh/submit_site.sh (fail fast and readably,
    not deep inside a solve)."""


def _parse_conf(path):
    values = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.split("#", 1)[0].strip()
            if not line:
                continue
            m = _KV_RE.match(line)
            if m:
                values[m.group(1)] = m.group(2).strip()
    return values


def _read_dat_param(dat_path, name):
    """Same numeric `param: <name> := <value>;` extraction as run_site_interactive.sh's own
    read_param() (a sed one-liner there), reimplemented in Python."""
    pattern = re.compile(rf"^param:\s*{re.escape(name)}\s*:=\s*([0-9.]+)")
    with open(dat_path, encoding="utf-8") as f:
        for line in f:
            m = pattern.match(line.strip())
            if m:
                return m.group(1)
    return None


def _count_csv_columns(csv_path):
    with open(csv_path, encoding="utf-8") as f:
        header = f.readline().rstrip("\n")
    n_semi = header.count(";") + 1
    n_comma = header.count(",") + 1
    n_fields = n_semi if n_semi >= n_comma else n_comma
    return n_fields - 1  # minus the index column, same convention as run_site_interactive.sh


def load_site_design(site_name):
    """Returns a dict shaped exactly like a DESIGNS[...] entry (params_dat/demand_csv/res_csv as
    absolute paths, plus max_battery_kwh/max_generator_kw as floats or None), read from
    sites/<site_name>.conf and validated against the .dat/CSV files it points at."""
    conf_path = os.path.join(SITES_DIR, f"{site_name}.conf")
    if not os.path.isfile(conf_path):
        raise SiteConfigError(f"no config at {conf_path}")
    conf = _parse_conf(conf_path)

    required = ("SITE_PARAMS", "SITE_DEMAND", "SITE_YEARS", "SITE_SCENARIOS", "SITE_MILP")
    missing = [v for v in required if v not in conf]
    if missing:
        raise SiteConfigError(f"{conf_path} is missing required field(s): {', '.join(missing)}")

    params_dat = os.path.join(INPUTS_DIR, conf["SITE_PARAMS"])
    demand_csv = os.path.join(INPUTS_DIR, conf["SITE_DEMAND"])
    if not os.path.isfile(params_dat):
        raise SiteConfigError(f"parameters file not found: {params_dat}")
    if not os.path.isfile(demand_csv):
        raise SiteConfigError(f"demand file not found: {demand_csv}")

    site_years = int(conf["SITE_YEARS"])
    site_scenarios = int(conf["SITE_SCENARIOS"])
    site_milp = int(conf["SITE_MILP"])
    for name, declared in (("Years", site_years), ("Scenarios", site_scenarios),
                            ("MILP_Formulation", site_milp)):
        actual = _read_dat_param(params_dat, name)
        if actual is None:
            raise SiteConfigError(f"could not read 'param: {name}' from {params_dat}")
        if int(float(actual)) != declared:
            raise SiteConfigError(
                f"{conf['SITE_PARAMS']} has {name} = {actual}, but {site_name}.conf declares {declared}")

    demand_cols = _count_csv_columns(demand_csv)
    expected_cols = site_scenarios * site_years
    if demand_cols != expected_cols:
        raise SiteConfigError(
            f"{conf['SITE_DEMAND']} has {demand_cols} data columns, expected {expected_cols} "
            f"(SITE_SCENARIOS={site_scenarios} x SITE_YEARS={site_years})")

    # SITE_RES is optional for Pyomo (RE_Supply_Calculation=1 can fetch live instead), but the
    # linopy port has no live-fetch path (docs Section "Demand/RES data acquisition") -- a
    # pre-generated RES CSV is always required here.
    if not conf.get("SITE_RES"):
        raise SiteConfigError(
            f"{conf_path} has no SITE_RES set -- the linopy port always needs a pre-generated "
            f"RES_Time_Series CSV (it has no live NASA-POWER-fetch path, unlike Pyomo's "
            f"RE_Supply_Calculation=1 branch)")
    res_csv = os.path.join(INPUTS_DIR, conf["SITE_RES"])
    if not os.path.isfile(res_csv):
        raise SiteConfigError(f"RES time series file not found: {res_csv}")
    res_cols = _count_csv_columns(res_csv)
    if res_cols <= 0 or res_cols % site_scenarios != 0:
        raise SiteConfigError(
            f"{conf['SITE_RES']} has {res_cols} data columns, not a positive multiple of "
            f"SITE_SCENARIOS={site_scenarios}")

    max_battery_kwh = float(conf["SITE_MAX_BATTERY_KWH"]) if conf.get("SITE_MAX_BATTERY_KWH") else None
    max_generator_kw = float(conf["SITE_MAX_GENERATOR_KW"]) if conf.get("SITE_MAX_GENERATOR_KW") else None

    return dict(
        params_dat=params_dat, demand_csv=demand_csv, res_csv=res_csv,
        max_battery_kwh=max_battery_kwh, max_generator_kw=max_generator_kw,
    )
