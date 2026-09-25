#!/bin/bash
# Size and certify a set of designs in parallel lanes on an interactive session: every job is
# free-sized (one run per genset pattern, the cheapest kept), its unit counts are read from the
# sizing log, and the design is certified with them pinned -- run_linopy_queue.sh's sizing plus the
# certification step that used to be started by hand. Added 2026-09-24 for the dea2025k re-run.
#
#     ./run_linopy_lanes.sh <lanes-file> [--memlimit GB] [--softmem GB] [--threads N]
#                           [--size-timelimit S] [--cert-timelimit S] [--foreground] [--summary]
#
# Lanes file: one job per line, "<lane> <site> <genset patterns> [VAR=value ...]"; '#' starts a
# comment. Jobs with the same lane run one after another, different lanes at the same time. The
# patterns are LINOPY_FIX_GENERATOR_UNITS values (cumulative units per step) separated by '|', e.g.
# "2" or "2|3" (both sized, the lower NPC certified) or "2,2,2"; "-" sizes with the genset count
# free, as the 1sc designs are. VAR=value pairs (LINOPY_* only)
# apply to the certification, e.g. LINOPY_GEN_PARALLEL=1. Example:
#
#     A  gili_ketapang_12sc_20y_MILP_1step_dieselops_dea2025k           2
#     B  gili_ketapang_12sc_20y_MILP_1step_dieselops_dea2025k_fuel064   2|3
#     B  gili_ketapang_12sc_20y_MILP_1step_dieselops_dea2025k_aut0      3    LINOPY_GEN_PARALLEL=1
#
#   --memlimit        Gurobi MemLimit per run in GB [default 100]; lanes x memlimit must fit the session
#   --softmem         LINOPY_SOFTMEMLIMIT in GB, below --memlimit [default memlimit - 10]: a sizing run
#                     that reaches it stops and exports its incumbent instead of dying on the hard limit
#   --threads         LINOPY_THREADS per run [default 24 / number of lanes]
#   --size-timelimit  Gurobi TimeLimit of each sizing run in s [default 7200]
#   --cert-timelimit  TimeLimit of the relaxed and of the integral certification solve in s [default 14400]
#   --foreground      do not start a tmux session
#   --summary         print the summary table of what has finished so far and exit
#
# Output: logs/lanes_<lanes-file name>/ -- per job <site>_size_gen<pattern>.log (or _size_free.log), <site>.pins
# ("pattern RES BATTERY GENERATOR sizing-NPC"), <site>_cert.log and <site>.done, plus summary.txt.
# Resumable: the same command in a later session skips jobs with a .done file, re-uses finished
# sizing logs and certifies straight away when the .pins file exists. Delete a job's files to
# redo it. Sizing uses LINOPY_MIPGAP [default 0.005], LINOPY_EXPORT_SUBOPTIMAL=1 and a
# LINOPY_SOLFILES prefix; certification uses LINOPY_DUALREDUCTIONS=0 and the script's own gaps.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUNNER="${SCRIPT_DIR}/run_linopy_interactive.sh"
PYTHON="${PYTHON:-python}"

die() { echo "ERROR: $*" >&2; exit 1; }

[ $# -ge 1 ] || die "usage: $0 <lanes-file> [--memlimit GB] [--softmem GB] [--threads N] [--size-timelimit S] [--cert-timelimit S] [--foreground] [--summary]"
LANES_FILE="$(cd "$(dirname "$1")" && pwd)/$(basename "$1")"; shift
[ -f "${LANES_FILE}" ] || die "no lanes file at ${LANES_FILE}"
MEMLIMIT=100; SOFTMEM=""; THREADS=""; SIZE_TL=7200; CERT_TL=14400; FOREGROUND=""; IN_TMUX_CHILD=""; SUMMARY_ONLY=""
while [ $# -gt 0 ]; do
    case "$1" in
        --memlimit)       MEMLIMIT="${2:-}"; shift 2 ;;
        --softmem)        SOFTMEM="${2:-}"; shift 2 ;;
        --threads)        THREADS="${2:-}"; shift 2 ;;
        --size-timelimit) SIZE_TL="${2:-}"; shift 2 ;;
        --cert-timelimit) CERT_TL="${2:-}"; shift 2 ;;
        --foreground)     FOREGROUND=1; shift ;;
        --summary)        SUMMARY_ONLY=1; shift ;;
        --tmux-child)     IN_TMUX_CHILD=1; shift ;;
        *) die "unknown argument: $1" ;;
    esac
done
for v in MEMLIMIT SIZE_TL CERT_TL; do [[ "${!v}" =~ ^[0-9]+$ ]] || die "${v} must be a whole number"; done
SOFTMEM="${SOFTMEM:-$(( MEMLIMIT - 10 ))}"
[[ "${SOFTMEM}" =~ ^[0-9]+$ ]] && [ "${SOFTMEM}" -lt "${MEMLIMIT}" ] \
    || die "--softmem must be below --memlimit (a soft limit above the hard one never fires)"
SIZE_MIPGAP="${LINOPY_MIPGAP:-0.005}"

NAME="$(basename "${LANES_FILE}" .txt)"
OUTDIR="${SCRIPT_DIR}/logs/lanes_${NAME}"

# ---- read the lanes file ------------------------------------------------------------------
LANE=(); SITE=(); PATS=(); EXTRA=()
while IFS= read -r line || [ -n "${line}" ]; do
    line="${line%%#*}"
    read -r lane site pats extra <<< "${line}" || true
    [ -n "${lane:-}" ] || continue
    [ -n "${site:-}" ] && [ -n "${pats:-}" ] || die "line '${line}': need <lane> <site> <patterns>"
    LANE+=("${lane}"); SITE+=("${site}"); PATS+=("${pats}"); EXTRA+=("${extra:-}")
done < "${LANES_FILE}"
[ "${#SITE[@]}" -gt 0 ] || die "${LANES_FILE} has no jobs"
mapfile -t LANE_IDS < <(printf '%s\n' "${LANE[@]}" | awk '!seen[$0]++')
THREADS="${THREADS:-$(( 24 / ${#LANE_IDS[@]} ))}"
[[ "${THREADS}" =~ ^[1-9][0-9]*$ ]] || die "--threads must be a positive integer"

summary() {
    printf '%-4s %-58s %-9s %-15s %-24s %-17s %s\n' "lane" "site" "gensets" "sizing NPC" "pins RES|BAT|GEN" "certified NPC" "verdict"
    for i in "${!SITE[@]}"; do
        local s="${SITE[$i]}" pat="-" npc="-" pins="-" cnpc="-" verdict="not run"
        if [ -f "${OUTDIR}/${s}.pins" ]; then
            read -r pat res bat gen npc < "${OUTDIR}/${s}.pins"
            pins="${res}|${bat}|${gen}"; npc="$(printf '%.2f' "${npc}")"; verdict="sized"
        fi
        [ -f "${OUTDIR}/${s}.failed" ] && verdict="FAILED: $(cat "${OUTDIR}/${s}.failed")"
        if [ -f "${OUTDIR}/${s}_cert.log" ]; then
            cnpc="$(grep -m1 -oE "^CERTIFIED NPC: [0-9,.]+" "${OUTDIR}/${s}_cert.log" | cut -d' ' -f3 || true)"
            grep -q "^NOT CERTIFIED" "${OUTDIR}/${s}_cert.log" && cnpc="not certified"
            verdict="$(grep -oE "verdict: [A-Za-z0-9_]+" "${OUTDIR}/${s}_cert.log" | tail -n 1 | cut -d' ' -f2 || true)"
            [ -n "${verdict}" ] || verdict="certifying"
            cnpc="${cnpc:--}"
        fi
        printf '%-4s %-58s %-9s %-15s %-24s %-17s %s\n' "${LANE[$i]}" "${s}" "${pat}" "${npc}" "${pins}" "${cnpc}" "${verdict}"
    done
    echo "(Sizing NPC is the free-sizing incumbent, relaxed commitment; only a CERTIFIED NPC goes into the paper."
    echo " A 'time_limit' sizing log means the pins are the best design found, not a proven optimum.)"
}

if [ -n "${SUMMARY_ONLY}" ]; then
    summary
    exit 0
fi

# ---- check every job before any solve -----------------------------------------------------
if [ -z "${IN_TMUX_CHILD}" ]; then
    echo "Lanes ${LANES_FILE}: ${#SITE[@]} job(s) in ${#LANE_IDS[@]} lane(s) (${LANE_IDS[*]}), MemLimit ${MEMLIMIT} GB" \
         "(soft ${SOFTMEM}), ${THREADS} threads each, sizing ${SIZE_TL} s, certification ${CERT_TL} s"
    echo "  memory if every lane peaks together: $(( ${#LANE_IDS[@]} * MEMLIMIT )) GB"
    declare -A STEPS=()
    for i in "${!SITE[@]}"; do
        s="${SITE[$i]}"
        if [ -z "${STEPS[$s]:-}" ]; then
            out="$(PYTHON="${PYTHON}" bash "${SCRIPT_DIR}/submit_linopy_site.sh" "${s}" --check 2>&1)" \
                || { echo "${out}" >&2; die "site '${s}' failed validation"; }
            STEPS[$s]="$(echo "${out}" | sed -n 's/^ *investment steps: *\([0-9]*\).*/\1/p')"
        fi
        for pat in ${PATS[$i]//|/ }; do
            [ "${pat}" = "-" ] && continue
            [[ "${pat}" =~ ^[0-9]+(,[0-9]+)*$ ]] || die "job $((i + 1)): bad genset pattern '${pat}'"
            n=$(( $(echo "${pat}" | tr -cd ',' | wc -c) + 1 ))
            [ "${n}" -eq "${STEPS[$s]}" ] || die "job $((i + 1)): pattern '${pat}' has ${n} value(s), '${s}' has ${STEPS[$s]} step(s)"
        done
        for kv in ${EXTRA[$i]}; do
            [[ "${kv}" =~ ^LINOPY_[A-Z_]+=[^[:space:]]*$ ]] || die "job $((i + 1)): '${kv}' is not a LINOPY_*=value setting"
        done
        state=""
        [ -f "${OUTDIR}/${s}.done" ] && state="(done, skipped)"
        [ -z "${state}" ] && [ -f "${OUTDIR}/${s}.pins" ] && state="(sized, certification next)"
        printf '  %-3s %-60s gensets %-9s %s %s\n' "${LANE[$i]}" "${s}" "${PATS[$i]}" "${EXTRA[$i]}" "${state}"
    done
fi

if [ -z "${TMUX:-}" ] && [ -z "${IN_TMUX_CHILD}" ] && [ -z "${FOREGROUND}" ]; then
    command -v tmux >/dev/null 2>&1 || die "not inside tmux and tmux is not installed -- use --foreground"
    tname="lp_lanes_${NAME}"
    cmd="$(printf '%q ' env "PYTHON=${PYTHON}" "LINOPY_MIPGAP=${SIZE_MIPGAP}" bash "${BASH_SOURCE[0]}" "${LANES_FILE}" \
        --memlimit "${MEMLIMIT}" --softmem "${SOFTMEM}" --threads "${THREADS}" \
        --size-timelimit "${SIZE_TL}" --cert-timelimit "${CERT_TL}" --tmux-child)"
    tmux new-session -d -s "${tname}" "${cmd}; echo; echo '[lanes finished -- press Enter to close]'; read -r"
    echo
    echo "Started in tmux session '${tname}'.  Attach: tmux attach -t ${tname}   (detach: Ctrl-b d)"
    echo "Progress: bash $0 ${LANES_FILE} --summary      Logs: ${OUTDIR}/"
    exit 0
fi

# ---- run ------------------------------------------------------------------------------------
mkdir -p "${OUTDIR}"
cp "${LANES_FILE}" "${OUTDIR}/lanes.txt"

npc_of() {  # sizing log -> NPC true of its last solve, empty if none
    grep -oE "NPC true: [0-9.eE+-]+" "$1" 2>/dev/null | tail -n 1 | cut -d' ' -f3 || true
}

pins_of() {  # sizing log -> "RES BATTERY GENERATOR" as comma-separated cumulative unit counts
    "${PYTHON}" - "$1" <<'EOF'
import re, sys
t = open(sys.argv[1], encoding="utf-8", errors="replace").read()
def grab(label):
    m = re.findall(label + r" cumulative UNIT COUNT per step: \[([^\]]*)\]", t)
    return ",".join(str(int(round(float(x)))) for x in m[-1].split()) if m else ""
print(grab("RES_Units "), grab("Battery_Units"), grab("Generator_Units"))
EOF
}

run_job() {  # index
    local i="$1" s="${SITE[$1]}" best="" bestnpc="" bestpins=""
    local base="${OUTDIR}/${s}"
    if [ -f "${base}.done" ]; then
        echo "  $(date +%H:%M:%S) [${LANE[$i]}] ${s}: done earlier, skipped"
        return 0
    fi
    rm -f "${base}.failed"
    if [ ! -f "${base}.pins" ]; then
        for pat in ${PATS[$i]//|/ }; do
            local tag="gen${pat//,/-}" fix="${pat}" npc pins
            [ "${pat}" = "-" ] && tag="free" && fix=""
            local log="${base}_size_${tag}.log"
            npc="$(npc_of "${log}")"
            if [ -z "${npc}" ] || ! grep -q "Generator_Units cumulative UNIT COUNT" "${log}" 2>/dev/null; then
                echo "  $(date +%H:%M:%S) [${LANE[$i]}] ${s}: sizing, gensets ${pat}"
                env -u LINOPY_PINNED_RES -u LINOPY_PINNED_BATTERY -u LINOPY_PINNED_GENERATOR -u LINOPY_CUTOFF \
                    LINOPY_FIX_GENERATOR_UNITS="${fix}" LINOPY_MIPGAP="${SIZE_MIPGAP}" LINOPY_EXPORT_SUBOPTIMAL=1 \
                    LINOPY_SOLFILES="${log%.log}_sol" LINOPY_SOFTMEMLIMIT="${SOFTMEM}" LINOPY_THREADS="${THREADS}" \
                    LINOPY_LOG_FILE="${log}" PYTHON="${PYTHON}" \
                    bash "${RUNNER}" "${s}" --timelimit "${SIZE_TL}" --memlimit "${MEMLIMIT}" --tmux-child \
                    > /dev/null 2>> "${base}.err" || true
                npc="$(npc_of "${log}")"
            fi
            pins="$(pins_of "${log}" 2>/dev/null || true)"
            if [ -z "${npc}" ] || [ "$(echo "${pins}" | wc -w)" -ne 3 ]; then
                echo "  $(date +%H:%M:%S) [${LANE[$i]}] ${s}: sizing with gensets ${pat} gave no design (see ${log})"
                continue
            fi
            echo "  $(date +%H:%M:%S) [${LANE[$i]}] ${s}: gensets ${pat} -> NPC ${npc}, $(grep -oE "status: [a-z]+ [a-z_]+" "${log}" | tail -n 1)"
            if [ -z "${bestnpc}" ] || awk -v a="${npc}" -v b="${bestnpc}" 'BEGIN{exit !(a < b)}'; then
                best="${pat}"; bestnpc="${npc}"; bestpins="${pins}"
            fi
        done
        if [ -z "${best}" ]; then
            echo "no sizing run produced a design" > "${base}.failed"
            echo "  $(date +%H:%M:%S) [${LANE[$i]}] ${s}: FAILED, no design to certify"
            return 1
        fi
        echo "${best} ${bestpins} ${bestnpc}" > "${base}.pins"
    fi
    local pat res bat gen npc
    read -r pat res bat gen npc < "${base}.pins"
    echo "  $(date +%H:%M:%S) [${LANE[$i]}] ${s}: certifying gensets ${pat} (RES ${res} | battery ${bat} | generator ${gen})"
    # shellcheck disable=SC2086  # EXTRA holds VAR=value words on purpose
    env -u LINOPY_FIX_GENERATOR_UNITS -u LINOPY_MIPGAP -u LINOPY_EXPORT_SUBOPTIMAL -u LINOPY_SOLFILES -u LINOPY_CUTOFF \
        LINOPY_PINNED_RES="${res}" LINOPY_PINNED_BATTERY="${bat}" LINOPY_PINNED_GENERATOR="${gen}" \
        LINOPY_DUALREDUCTIONS=0 LINOPY_INTEGRAL_TIMELIMIT="${CERT_TL}" LINOPY_SOFTMEMLIMIT="${SOFTMEM}" \
        LINOPY_THREADS="${THREADS}" LINOPY_LOG_FILE="${base}_cert.log" PYTHON="${PYTHON}" ${EXTRA[$i]} \
        bash "${RUNNER}" "${s}" --mode cert --timelimit "${CERT_TL}" --memlimit "${MEMLIMIT}" --tmux-child \
        > /dev/null 2>> "${base}.err" || true
    local verdict
    verdict="$(grep -oE "verdict: [A-Za-z0-9_]+" "${base}_cert.log" 2>/dev/null | tail -n 1 | cut -d' ' -f2 || true)"
    if [ -z "${verdict}" ]; then
        echo "certification ended without a verdict" > "${base}.failed"
        echo "  $(date +%H:%M:%S) [${LANE[$i]}] ${s}: certification FAILED (see ${base}_cert.log / .err)"
        return 1
    fi
    { grep -E "^CERTIFIED NPC|^NOT CERTIFIED" "${base}_cert.log" || true; echo "verdict: ${verdict}"; } > "${base}.done"
    echo "  $(date +%H:%M:%S) [${LANE[$i]}] ${s}: ${verdict} -- $(head -n 1 "${base}.done")"
    [ -s "${base}.err" ] || rm -f "${base}.err"
}

run_lane() {  # lane id
    for i in "${!SITE[@]}"; do
        [ "${LANE[$i]}" = "$1" ] || continue
        run_job "${i}" || true
    done
}

echo "Lanes started $(date) -- ${#SITE[@]} job(s) in ${#LANE_IDS[@]} lane(s); logs in ${OUTDIR}"
for lane in "${LANE_IDS[@]}"; do
    run_lane "${lane}" &
    sleep 5   # distinct start times; the Gurobi licence check does not like simultaneous logins
done
wait || true
echo "Lanes finished $(date)"
summary | tee "${OUTDIR}/summary.txt"
