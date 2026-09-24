#!/bin/bash
# Run a queue of linopy free-sizing jobs on an interactive session, at most --jobs at a time,
# and print a summary table at the end. Added 2026-09-12 for the 12sc genset-count enumeration
# (LINOPY_FIX_GENERATOR_UNITS), where each design needs several solves that differ only in the
# pinned genset count.
#
#     ./run_linopy_queue.sh <queue-file> [--jobs N] [--memlimit GB] [--timelimit S] [--foreground]
#
# Queue file: one job per line, "<site> [<genset pattern>] [<cutoff>]"; '#' starts a comment. The
# pattern is LINOPY_FIX_GENERATOR_UNITS (cumulative units per step, e.g. 2,2,3); leave it out or
# write '-' for plain free sizing. The optional cutoff is LINOPY_CUTOFF: the best Gurobi OBJECTIVE
# known for that design, so a pattern that cannot beat it ends quickly as infeasible/cutoff (a job
# that ends that way proves only that it is not better; it has no NPC of its own). Example:
#
#     gili_ketapang_12sc_20y_MILP_3step_dieselops_dea2025  2,2,2
#     gili_ketapang_12sc_20y_MILP_3step_dieselops_dea2025  2,2,3  11361137
#
#   --jobs        runs at once [default 2; the WLS licence allows 4 sessions since 2026-09-18, but
#                 memory is the limit: a 12sc 1-step sizing needs more than 50 GB in presolve]
#   --memlimit    Gurobi MemLimit per run in GB [default 150; a 12sc 5-step job died at 100 and
#                 another at 60 -- MemLimit counts memory Gurobi has allocated, above its RSS]
#   --timelimit   Gurobi TimeLimit per run in seconds [default: the session's time left minus
#                 one hour, read by run_linopy_interactive.sh at each run's start]
#   --foreground  do not start a tmux session (for testing where tmux is missing)
#
# Every job is checked first (site validation, pattern length = investment steps), so a typo
# fails before hours of solving. Each job runs through run_linopy_interactive.sh (in place, no
# tmux of its own) with LINOPY_MIPGAP from the environment [default 0.005 here]. Logs go to
# logs/queue_<stamp>/, one per job, plus summary.txt. Outside tmux the queue starts itself in a
# detached tmux session named lp_queue_<stamp>.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_linopy_interactive.sh"
PYTHON="${PYTHON:-python}"

die()  { echo "ERROR: $*" >&2; exit 1; }

[ $# -ge 1 ] || die "usage: $0 <queue-file> [--jobs N] [--memlimit GB] [--timelimit S] [--foreground]"
QUEUE="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; shift
[ -f "${QUEUE}" ] || die "no queue file at ${QUEUE}"
JOBS=2; MEMLIMIT=150; TIMELIMIT=""; FOREGROUND=""; IN_TMUX_CHILD=""; STAMP=""
while [ $# -gt 0 ]; do
    case "$1" in
        --jobs)       JOBS="${2:-}"; shift 2 ;;
        --memlimit)   MEMLIMIT="${2:-}"; shift 2 ;;
        --timelimit)  TIMELIMIT="${2:-}"; shift 2 ;;
        --foreground) FOREGROUND=1; shift ;;
        --tmux-child) IN_TMUX_CHILD=1; STAMP="${2:-}"; shift 2 ;;
        *) die "unknown argument: $1" ;;
    esac
done
[[ "${JOBS}" =~ ^[1-9][0-9]*$ ]] || die "--jobs must be a positive integer"
[[ "${MEMLIMIT}" =~ ^[0-9]+$ ]] || die "--memlimit must be an integer number of GB"
[ -z "${TIMELIMIT}" ] || [[ "${TIMELIMIT}" =~ ^[0-9]+$ ]] || die "--timelimit must be seconds"
export LINOPY_MIPGAP="${LINOPY_MIPGAP:-0.005}"
STAMP="${STAMP:-$(date +%Y%m%d_%H%M%S)}"

# ---- read and check the queue -------------------------------------------------------------
SITES=(); PATTERNS=(); CUTOFFS=()
while IFS= read -r line || [ -n "${line}" ]; do
    line="${line%%#*}"
    read -r site pat cut _ <<< "${line}" || true
    [ -n "${site:-}" ] || continue
    SITES+=("${site}"); PATTERNS+=("${pat:--}"); CUTOFFS+=("${cut:--}")
done < "${QUEUE}"
[ "${#SITES[@]}" -gt 0 ] || die "queue ${QUEUE} has no jobs"

if [ -z "${IN_TMUX_CHILD}" ]; then
    echo "Queue ${QUEUE}: ${#SITES[@]} job(s), ${JOBS} at a time, MemLimit ${MEMLIMIT} GB, MIPGap ${LINOPY_MIPGAP}"
    declare -A STEPS=()
    for i in "${!SITES[@]}"; do
        site="${SITES[$i]}"; pat="${PATTERNS[$i]}"
        if [ -z "${STEPS[$site]:-}" ]; then
            out="$(PYTHON="${PYTHON}" bash "${SCRIPT_DIR}/submit_linopy_site.sh" "${site}" --check 2>&1)" \
                || { echo "${out}" >&2; die "site '${site}' failed validation"; }
            STEPS[$site]="$(echo "${out}" | sed -n 's/^ *investment steps: *\([0-9]*\).*/\1/p')"
        fi
        if [ "${pat}" != "-" ]; then
            [[ "${pat}" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "job $((i + 1)): bad genset pattern '${pat}'"
            n=$(( $(echo "${pat}" | tr -cd ',' | wc -c) + 1 ))
            [ "${n}" -eq "${STEPS[$site]}" ] \
                || die "job $((i + 1)): pattern '${pat}' has ${n} value(s), '${site}' has ${STEPS[$site]} step(s)"
        fi
        cut="${CUTOFFS[$i]}"
        [ "${cut}" = "-" ] || [[ "${cut}" =~ ^[0-9]+(\.[0-9]+)?([eE][+-]?[0-9]+)?$ ]] \
            || die "job $((i + 1)): bad cutoff '${cut}' (a Gurobi objective value, or - for none)"
        printf '  %2d  %-60s gensets %-11s cutoff %s\n' "$((i + 1))" "${site}" "${pat}" "${cut}"
    done
fi

if [ -z "${TMUX:-}" ] && [ -z "${IN_TMUX_CHILD}" ] && [ -z "${FOREGROUND}" ]; then
    command -v tmux >/dev/null 2>&1 || die "not inside tmux and tmux is not installed -- use --foreground"
    name="lp_queue_${STAMP}"
    cmd="$(printf '%q ' env "PYTHON=${PYTHON}" "LINOPY_MIPGAP=${LINOPY_MIPGAP}" \
        bash "${BASH_SOURCE[0]}" "${QUEUE}" --jobs "${JOBS}" --memlimit "${MEMLIMIT}" \
        ${TIMELIMIT:+--timelimit "${TIMELIMIT}"} --tmux-child "${STAMP}")"
    tmux new-session -d -s "${name}" "${cmd}; echo; echo '[queue finished -- press Enter to close]'; read -r"
    echo
    echo "Started in tmux session '${name}'.  Attach: tmux attach -t ${name}   (detach: Ctrl-b d)"
    echo "Logs and summary: logs/queue_${STAMP}/"
    exit 0
fi

# ---- run ----------------------------------------------------------------------------------
QDIR="${SCRIPT_DIR}/logs/queue_${STAMP}"
mkdir -p "${QDIR}"
cp "${QUEUE}" "${QDIR}/queue.txt"
LOGS=()
for i in "${!SITES[@]}"; do
    tag="$(echo "${PATTERNS[$i]}" | tr ',' '-')"; [ "${tag}" = "-" ] && tag="free"
    LOGS+=("${QDIR}/$(printf '%02d' $((i + 1)))_${SITES[$i]}_gen${tag}.log")
done

run_job() {  # index
    local i="$1" pat="${PATTERNS[$1]}" cut="${CUTOFFS[$1]}"
    [ "${pat}" = "-" ] && pat=""
    [ "${cut}" = "-" ] && cut=""
    # Each job writes its own incumbents next to its log, so a run cut short later still
    # leaves its best design on disk (feed one back with LINOPY_WARMSTART_MST).
    LINOPY_FIX_GENERATOR_UNITS="${pat}" LINOPY_SOLFILES="${LOGS[$i]%.log}_sol" LINOPY_CUTOFF="${cut}" LINOPY_LOG_FILE="${LOGS[$i]}" PYTHON="${PYTHON}" \
        bash "${RUNNER}" "${SITES[$i]}" --memlimit "${MEMLIMIT}" \
        ${TIMELIMIT:+--timelimit "${TIMELIMIT}"} --tmux-child > /dev/null 2> "${LOGS[$i]%.log}.err"
    # stdout is already teed into the log; stderr keeps the runner's own errors (e.g. a failed
    # validation before the log exists). Drop the .err file when it stayed empty.
    [ -s "${LOGS[$i]%.log}.err" ] || rm -f "${LOGS[$i]%.log}.err"
}

echo "Queue started $(date) -- ${#SITES[@]} job(s), ${JOBS} at a time; logs in ${QDIR}"
running=0
for i in "${!SITES[@]}"; do
    if [ "${running}" -ge "${JOBS}" ]; then
        wait -n || true
        running=$((running - 1))
    fi
    echo "  $(date +%H:%M:%S) start job $((i + 1)): ${SITES[$i]} gensets ${PATTERNS[$i]}"
    run_job "${i}" &
    running=$((running + 1))
    sleep 5   # distinct start times; the Gurobi licence check does not like simultaneous logins
done
wait || true
echo "Queue finished $(date)"

# ---- summary ------------------------------------------------------------------------------
{
    printf '%-3s %-55s %-11s %-8s %-18s %-9s %-16s %-16s\n' "#" "site" "gensets" "status" "NPC true (USD)" "gap" "RES units" "battery units"
    for i in "${!SITES[@]}"; do
        log="${LOGS[$i]}"
        npc="-"; gap="-"; st="FAILED"; res="-"; bat="-"
        if [ -f "${log}" ]; then
            l="$(grep -m1 "NPC true:" "${log}" || true)"
            if [ -n "${l}" ]; then
                st="$(echo "${l}" | awk '{print $3}')"
                npc="$(echo "${l}" | sed -E 's/.*NPC true: ([0-9.]+).*/\1/' | awk '{printf "%.2f", $1}')"
            fi
            # Only a real "gap 0.0123%" counts; a cutoff run prints "gap -" and no percentage.
            gap="$(grep -m1 -o -E "gap [0-9.]+%" "${log}" | tail -n 1 | cut -d' ' -f2 || true)"
            res="$(grep -m1 "RES_Units  cumulative UNIT COUNT" "${log}" | sed -E 's/.*\[(.*)\].*/\1/' | tr -s ' .' ' ' | xargs | tr ' ' ',' || true)"
            bat="$(grep -m1 "Battery_Units cumulative UNIT COUNT" "${log}" | sed -E 's/.*\[(.*)\].*/\1/' | tr -s ' .' ' ' | xargs | tr ' ' ',' || true)"
            grep -q -E "Traceback|GurobiError" "${log}" && st="ERROR"
            # A job stopped by its cutoff is not a failure: it proves the pattern is not better.
            grep -q "Model objective exceeds cutoff" "${log}" && { st="CUTOFF"; npc="not better"; }
        fi
        printf '%-3s %-55s %-11s %-8s %-18s %-9s %-16s %-16s\n' "$((i + 1))" "${SITES[$i]}" "${PATTERNS[$i]}" "${st}" "${npc}" "${gap:--}" "${res:--}" "${bat:--}"
    done
    echo
    echo "Lowest NPC per site:"
    for site in $(printf '%s\n' "${SITES[@]}" | sort -u); do
        best=""; bestnpc=""
        for i in "${!SITES[@]}"; do
            [ "${SITES[$i]}" = "${site}" ] || continue
            l="$(grep -m1 "NPC true:" "${LOGS[$i]}" 2>/dev/null || true)"
            [ -n "${l}" ] || continue
            v="$(echo "${l}" | sed -E 's/.*NPC true: ([0-9.]+).*/\1/')"
            if [ -z "${bestnpc}" ] || awk -v a="${v}" -v b="${bestnpc}" 'BEGIN{exit !(a < b)}'; then
                bestnpc="${v}"; best="${PATTERNS[$i]}"
            fi
        done
        printf '  %-55s gensets %-11s NPC %s\n' "${site}" "${best:--}" "${bestnpc:--}"
    done
    echo "(Check each gap: a run stopped by its TimeLimit reports a feasible, not optimal, NPC.)"
} | tee "${QDIR}/summary.txt"
