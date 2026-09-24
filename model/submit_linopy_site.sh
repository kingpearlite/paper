#!/bin/bash
# Submit a linopy run for a named site through SLURM -- the linopy counterpart of submit_site.sh
# (which runs the Pyomo model). Added 2026-09-11 for the paper-update designs.
#
#     ./submit_linopy_site.sh <site-name> [--mode free|cert] [--time D-HH:MM:SS] [--check]
#
#   --mode free   free sizing + Results.py export (linopy_results_export.py)        [default]
#   --mode cert   fixed-design certification of a pinned sizing + export
#                 (linopy_fixed_design_export.py: relaxed solve, integral re-solve if a check
#                 fails, Results folder tagged with the verdict). Needs LINOPY_PINNED_RES,
#                 LINOPY_PINNED_BATTERY and LINOPY_PINNED_GENERATOR exported in this shell:
#                 comma-separated CUMULATIVE UNIT COUNTS per investment step, i.e. the
#                 "cumulative UNIT COUNT per step" lines of the free-sizing log.
#   --time        override the conf's SITE_TIME for this submission
#   --check       validate and print what would be submitted, do not submit
#
# Reads sites/<site-name>.conf for SITE_CPUS/SITE_MEM/SITE_TIME and validates the site with the
# same site_config.load_site_design() the linopy scripts use (file existence, Years/Scenarios/
# MILP against the .dat, demand/RES column counts) before anything is queued. Gurobi gets
# Threads = the allocated cores (the scripts read SLURM_CPUS_PER_TASK), MemLimit = 90% of
# SITE_MEM, and a TimeLimit one hour short of the wall time so the Results export still runs.
# Run from an environment with the project's Python (on HPC: conda activate myenv); override
# the interpreter with PYTHON=/path/to/python.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
SITES_DIR="${SCRIPT_DIR}/sites"
RUNNER="${SCRIPT_DIR}/run_linopy_site.sbatch"
PYTHON="${PYTHON:-python}"

die()  { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }

[ $# -ge 1 ] || die "usage: $0 <site-name> [--mode free|cert] [--time D-HH:MM:SS] [--check]"
SITE="$1"; shift
MODE="free"; CHECK_ONLY=""; TIME_OVERRIDE=""
while [ $# -gt 0 ]; do
    case "$1" in
        --mode)  MODE="${2:-}"; shift 2 ;;
        --time)  TIME_OVERRIDE="${2:-}"; shift 2 ;;
        --check) CHECK_ONLY=1; shift ;;
        *) die "unknown argument: $1" ;;
    esac
done
case "${MODE}" in free|cert) ;; *) die "--mode must be free or cert, got '${MODE}'" ;; esac

CONF="${SITES_DIR}/${SITE}.conf"
[ -f "${CONF}" ] || die "no config at ${CONF}"
# shellcheck source=/dev/null
source "${CONF}"
for v in SITE_CPUS SITE_MEM SITE_TIME; do
    [ -n "${!v:-}" ] || die "${v} is not set in ${CONF}"
done
WALL="${TIME_OVERRIDE:-${SITE_TIME}}"

echo "Site '${SITE}' (${CONF}), mode ${MODE}"
if [ "${MODE}" = "cert" ]; then
    for v in LINOPY_PINNED_RES LINOPY_PINNED_BATTERY LINOPY_PINNED_GENERATOR; do
        [ -n "${!v:-}" ] || die "--mode cert needs ${v} exported (comma-separated cumulative unit counts per step)"
    done
fi
( cd "${SCRIPT_DIR}/Code/linopy_port" && "${PYTHON}" - "${SITE}" "${MODE}" <<'EOF'
import math, os, sys
import site_config
from linopy_core_model_builder import load_dat_scalar
site, mode = sys.argv[1], sys.argv[2]
d = site_config.load_site_design(site)
print("  [ok] site validated by site_config.load_site_design")
for k in ("params_dat", "demand_csv", "res_csv", "max_battery_kwh", "max_generator_kw"):
    print(f"       {k}: {d[k]}")
steps = math.ceil(load_dat_scalar(d["params_dat"], "Years") / load_dat_scalar(d["params_dat"], "Step_Duration"))
print(f"       investment steps: {steps}")
if mode == "cert":
    for v in ("LINOPY_PINNED_RES", "LINOPY_PINNED_BATTERY", "LINOPY_PINNED_GENERATOR"):
        vals = os.environ[v].split(",")
        if len(vals) != steps:
            sys.exit(f"ERROR: {v} has {len(vals)} value(s), the design has {steps} investment step(s)")
        [float(x) for x in vals]
    print("  [ok] pinned sizing has one value per step")
EOF
) || die "site validation failed (see above)"

PINNED_EXPORTS=""
if [ "${MODE}" = "cert" ]; then
    PINNED_EXPORTS=",LINOPY_PINNED_RES=${LINOPY_PINNED_RES},LINOPY_PINNED_BATTERY=${LINOPY_PINNED_BATTERY}"
    PINNED_EXPORTS+=",LINOPY_PINNED_GENERATOR=${LINOPY_PINNED_GENERATOR}"
fi

# Wall time [D-]HH:MM:SS -> seconds; Gurobi stops one hour earlier so the export can finish.
if [[ "${WALL}" =~ ^(([0-9]+)-)?([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
    wall_s=$(( ${BASH_REMATCH[2]:-0} * 86400 + 10#${BASH_REMATCH[3]} * 3600 + 10#${BASH_REMATCH[4]} * 60 + 10#${BASH_REMATCH[5]} ))
else
    die "cannot parse wall time '${WALL}' (expected [D-]HH:MM:SS)"
fi
[ "${wall_s}" -gt 7200 ] || die "wall time ${WALL} too short (need > 2 h)"
TIMELIMIT=$(( wall_s - 3600 ))
[[ "${SITE_MEM}" =~ ^([0-9]+)G$ ]] || die "SITE_MEM='${SITE_MEM}' not in <n>G form"
MEMLIMIT=$(( BASH_REMATCH[1] * 9 / 10 ))

echo "  resources  : ${SITE_CPUS} cores, ${SITE_MEM}, wall ${WALL}"
echo "  Gurobi     : TimeLimit ${TIMELIMIT} s, MemLimit ${MEMLIMIT} GB, Threads = allocated cores"
[ "${MODE}" = "cert" ] && echo "  pinned     : RES ${LINOPY_PINNED_RES} | battery ${LINOPY_PINNED_BATTERY} | generator ${LINOPY_PINNED_GENERATOR}"

if [ -n "${CHECK_ONLY}" ]; then
    echo
    echo "All checks passed. Not submitting (--check)."
    exit 0
fi

[ -f "${RUNNER}" ] || die "runner not found: ${RUNNER}"
mkdir -p "${SCRIPT_DIR}/logs"
# The Gurobi WLS licence refuses a third concurrent session (see submit_site.sh).
running=$(squeue -u "${USER}" -h -t RUNNING,PENDING -o "%i" 2>/dev/null | wc -l | tr -d ' ')
if [ "${running}" -ge 2 ]; then
    warn "you already have ${running} job(s) queued or running; a third concurrent Gurobi session is refused."
fi

sbatch \
    --job-name="linopy_${MODE}_${SITE}" \
    --cpus-per-task="${SITE_CPUS}" \
    --mem="${SITE_MEM}" \
    --time="${WALL}" \
    --output="logs/linopy_${MODE}_${SITE}_%j.out" \
    --error="logs/linopy_${MODE}_${SITE}_%j.err" \
    --export="ALL,LINOPY_SITE=${SITE},LINOPY_RUN_MODE=${MODE},LINOPY_TIMELIMIT=${TIMELIMIT},LINOPY_MEMLIMIT=${MEMLIMIT}${PINNED_EXPORTS}" \
    "${RUNNER}"
