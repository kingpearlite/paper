#!/bin/bash
# Generic linopy runner (2026-09-11). Do NOT submit this directly with sbatch -- resources and
# the site come from sites/<site>.conf through:
#
#     ./submit_linopy_site.sh <site-name> [--mode free|cert]
#
# which validates the site first and passes LINOPY_SITE, LINOPY_RUN_MODE, LINOPY_TIMELIMIT,
# LINOPY_MEMLIMIT (and, for cert, LINOPY_PINNED_*) through --export.

#SBATCH --partition=comp
##SBATCH --account=<project>

set -euo pipefail

source "$HOME/miniconda3/etc/profile.d/conda.sh"
conda activate myenv

if [ -z "${LINOPY_SITE:-}" ] || [ -z "${LINOPY_RUN_MODE:-}" ]; then
    echo "ERROR: LINOPY_SITE/LINOPY_RUN_MODE not set -- submit through ./submit_linopy_site.sh" >&2
    exit 1
fi

echo "=============================================================================="
echo "LINOPY ${LINOPY_RUN_MODE}: ${LINOPY_SITE}"
echo "  Gurobi     : TimeLimit=${LINOPY_TIMELIMIT:-default} s, MemLimit=${LINOPY_MEMLIMIT:-default} GB"
echo "  slurm      : ${SLURM_CPUS_PER_TASK:-?} cores, job ${SLURM_JOB_ID:-?}, $(hostname)"
echo "  git        : $(git -C "${SLURM_SUBMIT_DIR}" rev-parse --short HEAD 2>/dev/null || echo unknown)"
[ "${LINOPY_RUN_MODE}" = "cert" ] && echo "  pinned     : RES ${LINOPY_PINNED_RES} | battery ${LINOPY_PINNED_BATTERY} | generator ${LINOPY_PINNED_GENERATOR}"
echo "=============================================================================="

cd "${SLURM_SUBMIT_DIR}/Code/linopy_port"
export LINOPY_CONFIRM_REAL_SCALE=1
case "${LINOPY_RUN_MODE}" in
    free) python -u linopy_results_export.py "${LINOPY_SITE}" ;;
    cert) python -u linopy_fixed_design_export.py "${LINOPY_SITE}" ;;
    *) echo "ERROR: LINOPY_RUN_MODE must be free or cert, got '${LINOPY_RUN_MODE}'" >&2; exit 1 ;;
esac
