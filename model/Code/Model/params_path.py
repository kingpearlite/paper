"""
Single source of truth for WHICH parameters file a run reads.

Why this exists
---------------
Every module used to hardcode 'Parameters.dat', which made it shared mutable
state across concurrent jobs. A *running* job is immune (it reads the file once
at import), but a *queued or newly submitted* job reads whatever happens to be
on disk when it starts. On 2026-07-25 that cost ~8h of compute: job 58530437 was
submitted from run_microgridspy_6sc_5y_MILP_mipfocus3.sbatch while
Parameters.dat still held Years := 10 from a 10-year run, so it solved the
10-YEAR problem (5,256,181 rows) under a "6sc_5y" label, at memory sized for 5y,
and would have written 10-year results into Results/<ts>_6sc_5y_MILP_mipfocus3/.

Nothing caught it because the CLI tag passed to MicroGrids.py is consumed only by
run_id.py to name the output folder -- it never had to agree with the data.

Usage
-----
Pin each sbatch script to its own committed parameters file:

    export MGPY_PARAMS=Parameters_6sc_5y_MILP.dat

Bare filenames resolve inside Code/Inputs/; absolute paths are used as-is.
Unset -> 'Parameters.dat', so every existing workflow behaves exactly as before.

Optionally assert what the run is supposed to be. These turn a silent
mislabelling into an immediate crash, before Gurobi burns any walltime:

    export MGPY_EXPECT_YEARS=5
    export MGPY_EXPECT_SCENARIOS=6

Import-time side effects are deliberate: resolution, validation and the banner
all happen once, at first import, because Python caches modules. Every consumer
sees the same file for the whole run even if the file is replaced on disk
mid-solve.
"""
import os
import re

INPUTS_DIRECTORY = os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'Inputs')

DEFAULT_PARAMS_FILE = 'Parameters.dat'


def resolve_input(env_var, default_name, example):
    """Return the absolute path of an input file, overridable via `env_var`.

    Bare filenames resolve inside Code/Inputs/; absolute paths are used as-is.
    A missing file is fatal rather than silently falling back to the default --
    silently falling back would recreate the exact class of bug this module
    exists to prevent: a run whose label and data disagree, found hours later.
    """
    name = os.environ.get(env_var, default_name).strip()
    path = name if os.path.isabs(name) else os.path.join(INPUTS_DIRECTORY, name)
    path = os.path.abspath(path)
    if not os.path.isfile(path):
        raise FileNotFoundError(
            "{} points at a file that does not exist: {}\n"
            "Set {} to a filename inside Code/Inputs/ (e.g. {}) or to an "
            "absolute path.".format(env_var, path, env_var, example))
    return path


def _resolve():
    """Return the absolute path of the parameters file this run should read."""
    return resolve_input('MGPY_PARAMS', DEFAULT_PARAMS_FILE,
                         'Parameters_6sc_5y_MILP.dat')


def read_param(lines, name):
    """Extract a numeric 'param: <name> := <value>;' from already-read .dat lines.

    Returns a float, or None when the parameter is absent. Kept tolerant on
    purpose -- this is used for reporting and optional assertions, not for
    building the model, so an unparseable line must not break a solve.
    """
    pattern = re.compile(r'^param:\s*' + re.escape(name) + r'\s*:=\s*(-?[\d.]+)\s*;')
    for line in lines:
        match = pattern.match(line.strip())
        if match:
            try:
                return float(match.group(1))
            except ValueError:
                return None
    return None


def _check_expectations(path):
    """Crash now if the file's contents contradict what the sbatch declared."""
    expectations = (('MGPY_EXPECT_YEARS', 'Years'),
                    ('MGPY_EXPECT_SCENARIOS', 'Scenarios'),
                    ('MGPY_EXPECT_MILP', 'MILP_Formulation'))
    declared = [(env, param) for env, param in expectations if os.environ.get(env)]
    if not declared:
        return
    with open(path) as handle:
        lines = handle.readlines()
    for env, param in declared:
        expected = float(os.environ[env])
        actual = read_param(lines, param)
        if actual is None:
            raise ValueError(
                "{} is set to {:g} but '{}' was not found in {}".format(
                    env, expected, param, path))
        if actual != expected:
            raise ValueError(
                "PARAMETER MISMATCH -- refusing to start.\n"
                "  {} declares {} = {:g}\n"
                "  {} actually has {} = {:g}\n"
                "The sbatch script and the parameters file disagree about what "
                "this run is. Fix one of them before resubmitting.".format(
                    env, param, expected, os.path.basename(path), param, actual))


PARAMS_PATH = _resolve()
_check_expectations(PARAMS_PATH)

# Demand.csv is shared mutable state exactly like Parameters.dat, and is NOT
# covered by MGPY_PARAMS. On 2026-07-25 it was swapped to the 6sc/10y profile for
# a 10y run and never swapped back; every "5y" run afterwards silently used the
# first 5 columns of 10-year demand. Pin it per-config to close that race:
#     export MGPY_DEMAND=Demand_6sc_5y.csv
# Unset -> 'Demand.csv', preserving existing behaviour. The horizon mismatch is
# additionally caught in Initialize.py, which now refuses to trim excess columns.
DEMAND_PATH = resolve_input('MGPY_DEMAND', 'Demand.csv', 'Demand_6sc_5y.csv')

# RES_Time_Series.csv and WT_Power_Curve.csv are shared mutable state exactly
# like Parameters.dat and Demand.csv, but were NOT covered by either safeguard
# above. That gap let two nominally-identical "6sc_5y" submissions (jobs
# 58519303 and 58547964, two days apart) silently solve different problem
# instances -- confirmed via Gurobi's own "Model fingerprint" log line, not
# caught until Gurobi Support flagged the coefficient-statistics mismatch.
# Pin per-config to close the same race:
#     export MGPY_RES=RES_Time_Series_6sc_5y.csv
# Unset -> 'RES_Time_Series.csv', preserving existing behaviour.
RES_PATH = resolve_input('MGPY_RES', 'RES_Time_Series.csv',
                         'RES_Time_Series_6sc_5y.csv')

# Turbine power-curve data. Lower risk than RES_PATH (spec data, not
# scenario/weather data) but hardcoded the same way, so closed the same way.
WT_POWER_CURVE_PATH = resolve_input('MGPY_WT_POWER_CURVE', 'WT_Power_Curve.csv',
                                    'WT_Power_Curve.csv')

# Backwards-compatible alias: modules that previously built their own
# 'data_file_path' can import this name instead.
data_file_path = PARAMS_PATH


def describe():
    """One-line summary of the configuration actually loaded, for the log header."""
    with open(PARAMS_PATH) as handle:
        lines = handle.readlines()
    fields = []
    for label, param in (('Years', 'Years'), ('Scenarios', 'Scenarios'),
                         ('Periods', 'Periods'), ('MILP', 'MILP_Formulation')):
        value = read_param(lines, param)
        if value is not None:
            fields.append('{}={:g}'.format(label, value))
    return '{}  [{}]'.format(os.path.basename(PARAMS_PATH), ', '.join(fields))


print('=' * 78)
print('PARAMETERS FILE: {}'.format(describe()))
print('  full path: {}'.format(PARAMS_PATH))
print('DEMAND FILE:     {}'.format(os.path.basename(DEMAND_PATH)))
print('  full path: {}'.format(DEMAND_PATH))
print('RES TIME SERIES FILE: {}'.format(os.path.basename(RES_PATH)))
print('  full path: {}'.format(RES_PATH))
print('WT POWER CURVE FILE:  {}'.format(os.path.basename(WT_POWER_CURVE_PATH)))
print('  full path: {}'.format(WT_POWER_CURVE_PATH))
_unpinned = [name for name in ('MGPY_PARAMS', 'MGPY_DEMAND', 'MGPY_RES') if not os.environ.get(name)]
if _unpinned:
    print('  ({} unset -- using the shared default(s). Set them in the sbatch'.format(
        ' and '.join(_unpinned)))
    print('   script to pin this run to its own files and avoid racing other jobs.)')
print('=' * 78)
