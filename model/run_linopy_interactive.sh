#!/bin/bash
# Run a linopy job for a named site on an already-allocated interactive session (e.g. an HPC
# VS Code Server session with 250 GB / 24 threads, where no SLURM job can be requested).
# The interactive counterpart of submit_linopy_site.sh; added 2026-09-11.
#
#     ./run_linopy_interactive.sh <site-name> [--mode free|cert] [--timelimit SECONDS]
#                                 [--memlimit GB] [--check]
#
#   --mode free    free sizing + Results.py export (linopy_results_export.py)        [default]
#   --mode cert    fixed-design certification + export (linopy_fixed_design_export.py); needs
#                  LINOPY_PINNED_RES/BATTERY/GENERATOR exported (cumulative unit counts per step)
#   --timelimit    Gurobi TimeLimit in seconds. Default: the session's remaining time minus one
#                  hour (so the export still runs), read with squeue from SLURM_JOB_ID, or from
#                  LINOPY_TIME_LEFT ([D-]HH:MM:SS) if set. Required when neither is available.
#   --memlimit     Gurobi MemLimit in GB [default 180 -- leaves room on a 250 GB session for the
#                  linopy model itself, which stays in Python memory during the solve]
#   --check        validate only (same checks as submit_linopy_site.sh --check), do not run
#
# Outside tmux the script starts itself in a new detached tmux session (a solve that runs for
# hours dies with the terminal otherwise) and prints how to attach; inside tmux it runs in place.
# Output is teed to logs/linopy_<mode>_<site>_<YYYYmmdd_HHMM>.log. Gurobi Threads come from
# SLURM_CPUS_PER_TASK if the session sets it, else the scripts' default of 24.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${PYTHON:-python}"

PASS_VARS=(LINOPY_PINNED_RES LINOPY_PINNED_BATTERY LINOPY_PINNED_GENERATOR
           LINOPY_INTEGRAL_TIMELIMIT LINOPY_MIPGAP LINOPY_WARMSTART_MST
           LINOPY_FIX_GENERATOR_UNITS LINOPY_CUTOFF LINOPY_SOFTMEMLIMIT
           LINOPY_NODEFILESTART LINOPY_NODEFILEDIR LINOPY_THREADS
           LINOPY_SOLFILES LINOPY_EXPORT_SUBOPTIMAL LINOPY_INTEGRAL_MIPGAP
           LINOPY_DUALREDUCTIONS LINOPY_GEN_PARALLEL)
die()  { echo "ERROR: $*" >&2; exit 1; }
warn() { echo "WARNING: $*" >&2; }

[ $# -ge 1 ] || die "usage: $0 <site-name> [--mode free|cert] [--timelimit SECONDS] [--memlimit GB] [--check]"
SITE="$1"; shift
MODE="free"; TIMELIMIT=""; MEMLIMIT="180"; CHECK_ONLY=""; IN_TMUX_CHILD=""
while [ $# -gt 0 ]; do
    case "$1" in
        --mode)      MODE="${2:-}"; shift 2 ;;
        --timelimit) TIMELIMIT="${2:-}"; shift 2 ;;
        --memlimit)  MEMLIMIT="${2:-}"; shift 2 ;;
        --check)     CHECK_ONLY=1; shift ;;
        --tmux-child) IN_TMUX_CHILD=1; shift ;;
        *) die "unknown argument: $1" ;;
    esac
done
case "${MODE}" in free|cert) ;; *) die "--mode must be free or cert, got '${MODE}'" ;; esac
[[ "${MEMLIMIT}" =~ ^[0-9]+$ ]] || die "--memlimit must be an integer number of GB"

# Same validation as the SLURM path (site files, Years/Scenarios/MILP, column counts, pins).
PYTHON="${PYTHON}" bash "${SCRIPT_DIR}/submit_linopy_site.sh" "${SITE}" --mode "${MODE}" --check \
    | grep -v -E "^  (resources|Gurobi)|Not submitting" || die "validation failed (see above)"

to_seconds() {  # [D-]HH:MM:SS | MM:SS -> seconds; empty on failure
    if [[ "$1" =~ ^(([0-9]+)-)?([0-9]+):([0-9]{2}):([0-9]{2})$ ]]; then
        echo $(( ${BASH_REMATCH[2]:-0} * 86400 + 10#${BASH_REMATCH[3]} * 3600 + 10#${BASH_REMATCH[4]} * 60 + 10#${BASH_REMATCH[5]} ))
    elif [[ "$1" =~ ^([0-9]+):([0-9]{2})$ ]]; then
        echo $(( 10#${BASH_REMATCH[1]} * 60 + 10#${BASH_REMATCH[2]} ))
    fi
}

if [ -z "${TIMELIMIT}" ]; then
    left="${LINOPY_TIME_LEFT:-}"
    if [ -z "${left}" ] && [ -n "${SLURM_JOB_ID:-}" ] && command -v squeue >/dev/null 2>&1; then
        left="$(squeue -h -j "${SLURM_JOB_ID}" -o %L 2>/dev/null | tr -d ' ')"
    fi
    [ -n "${left}" ] || die "cannot tell how long this session has left -- pass --timelimit SECONDS or set LINOPY_TIME_LEFT=[D-]HH:MM:SS"
    left_s="$(to_seconds "${left}")"
    [ -n "${left_s}" ] || die "cannot parse the session's remaining time '${left}' -- pass --timelimit SECONDS"
    TIMELIMIT=$(( left_s - 3600 ))
    [ "${TIMELIMIT}" -ge 1800 ] || die "session has only ${left} left; not enough for a solve plus export"
    echo "  time       : session has ${left} left -> Gurobi TimeLimit ${TIMELIMIT} s"
else
    [[ "${TIMELIMIT}" =~ ^[0-9]+$ ]] || die "--timelimit must be an integer number of seconds"
    echo "  time       : Gurobi TimeLimit ${TIMELIMIT} s (given)"
fi
echo "  memory     : Gurobi MemLimit ${MEMLIMIT} GB; Threads ${LINOPY_THREADS:-${SLURM_CPUS_PER_TASK:-24}}"

# The session's own memory allocation, not the node's: /proc/meminfo reports the whole node
# (985 GB on m3t106 while the session had 250 GB), so it cannot be used for this check.
if [ -n "${SLURM_MEM_PER_NODE:-}" ]; then
    alloc_gb=$(( SLURM_MEM_PER_NODE / 1024 ))
    echo "  allocation : ${alloc_gb} GB (SLURM_MEM_PER_NODE)"
    [ $(( MEMLIMIT * 10 )) -le $(( alloc_gb * 9 )) ] \
        || warn "--memlimit ${MEMLIMIT} GB is above 90% of this session's ${alloc_gb} GB; the linopy model also needs memory"
else
    echo "  allocation : unknown (SLURM_MEM_PER_NODE unset) -- keep --memlimit well below your session's RAM"
fi
# Gurobi WLS sessions allowed at once, shared by the laptop and HPC: 2 until 2026-09-18, 4 since
# (confirmed by three HPC queue jobs holding sessions together). Override with LINOPY_WLS_SESSIONS.
wls_sessions="${LINOPY_WLS_SESSIONS:-4}"
running=$(pgrep -fc "linopy_(results|fixed_design)_export.py" 2>/dev/null || true)
if [ "${running:-0}" -ge "${wls_sessions}" ]; then
    warn "${running} linopy runs are already going; the Gurobi WLS licence allows ${wls_sessions} sessions"
fi

if [ -n "${CHECK_ONLY}" ]; then
    echo
    echo "All checks passed. Not running (--check)."
    exit 0
fi

if [ -z "${TMUX:-}" ] && [ -z "${IN_TMUX_CHILD}" ]; then
    command -v tmux >/dev/null 2>&1 || die "not inside tmux and tmux is not installed -- start one first"
    name="lp_${MODE}_$(echo "${SITE}" | sed -E 's/^gili_ketapang_//; s/_20y_MILP//; s/_dieselops_dea2025//')"
    # Everything the child needs goes on its command line: a new tmux session takes its
    # environment from the tmux server, not from this shell.
    env_args=()
    for v in "${PASS_VARS[@]}"; do env_args+=("${v}=${!v:-}"); done
    cmd="$(printf '%q ' env "PYTHON=${PYTHON}" "${env_args[@]}" \
        bash "${SCRIPT_DIR}/run_linopy_interactive.sh" "${SITE}" --mode "${MODE}" \
        --timelimit "${TIMELIMIT}" --memlimit "${MEMLIMIT}" --tmux-child)"
    tmux new-session -d -s "${name}" "${cmd}; echo; echo '[run finished -- press Enter to close]'; read -r"
    echo
    echo "Started in tmux session '${name}'.  Attach: tmux attach -t ${name}   (detach: Ctrl-b d)"
    exit 0
fi

mkdir -p "${SCRIPT_DIR}/logs"
# Seconds and the pinned genset count are in the name, so runs of one site started in the same
# minute (e.g. a LINOPY_FIX_GENERATOR_UNITS enumeration) do not write into the same file.
GEN_TAG=""
[ -n "${LINOPY_FIX_GENERATOR_UNITS:-}" ] && GEN_TAG="_gen$(echo "${LINOPY_FIX_GENERATOR_UNITS}" | tr -d ' ' | tr ',' '-')"
LOG="${SCRIPT_DIR}/logs/linopy_${MODE}_${SITE}${GEN_TAG}_$(date +%Y%m%d_%H%M%S).log"
# run_linopy_queue.sh chooses the log path itself so it can read the result back.
[ -n "${LINOPY_LOG_FILE:-}" ] && LOG="${LINOPY_LOG_FILE}" && mkdir -p "$(dirname "${LOG}")"
echo "  log        : ${LOG}"
# Gurobi and linopy resolve these against the working directory, which is Code/linopy_port
# below -- so a path given relative to the repository root (logs/sol_x) would land in the
# wrong place. Make them absolute before the cd (2026-09-12: LINOPY_SOLFILES silently wrote
# into Code/linopy_port/logs/ and the expected files appeared to be missing).
for v in LINOPY_SOLFILES LINOPY_WARMSTART_MST LINOPY_NODEFILEDIR; do
    val="${!v:-}"
    case "${val}" in
        ""|/*) ;;
        *) export "${v}=${SCRIPT_DIR}/${val}" ;;
    esac
done
cd "${SCRIPT_DIR}/Code/linopy_port"
export LINOPY_CONFIRM_REAL_SCALE=1 LINOPY_SITE="${SITE}" LINOPY_TIMELIMIT="${TIMELIMIT}" LINOPY_MEMLIMIT="${MEMLIMIT}"
for v in "${PASS_VARS[@]}"; do
    [ -n "${!v:-}" ] || unset "${v}"   # passed empty through tmux when not set
done
if [ "${MODE}" = "free" ]; then
    unset LINOPY_PINNED_RES LINOPY_PINNED_BATTERY LINOPY_PINNED_GENERATOR
    script=linopy_results_export.py
else
    script=linopy_fixed_design_export.py
fi
{
    echo "LINOPY ${MODE}: ${SITE}  (git $(git -C "${SCRIPT_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown), $(hostname), $(date))"
    echo "  TimeLimit=${TIMELIMIT} s MemLimit=${MEMLIMIT} GB Threads=${SLURM_CPUS_PER_TASK:-24} MIPGap=${LINOPY_MIPGAP:-script default}"
    [ "${MODE}" = "cert" ] && echo "  pinned: RES ${LINOPY_PINNED_RES} | battery ${LINOPY_PINNED_BATTERY} | generator ${LINOPY_PINNED_GENERATOR}"
    [ "${MODE}" = "cert" ] && [ "${LINOPY_GEN_PARALLEL:-0}" = "1" ] && echo "  genset commitment: parallel load sharing (LINOPY_GEN_PARALLEL=1)"
    [ "${MODE}" = "free" ] && [ -n "${LINOPY_FIX_GENERATOR_UNITS:-}" ] && echo "  generator units fixed: ${LINOPY_FIX_GENERATOR_UNITS} (PV and battery free)"
    [ -n "${LINOPY_CUTOFF:-}" ] && echo "  cutoff: ${LINOPY_CUTOFF} (objective; a worse design ends as infeasible/cutoff)"
    [ -n "${LINOPY_SOLFILES:-}" ] && echo "  solfiles: ${LINOPY_SOLFILES}_<n>.sol written on every improved incumbent"
    "${PYTHON}" -u "${script}" "${SITE}"
} 2>&1 | tee "${LOG}"
