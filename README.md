# Code and model inputs

**From Operating Records to Investment Plans: Synthetic Load Ensembles for Island Microgrid Capacity Expansion**
(submission to ACM SAC 2027)

This repository holds the code and model inputs behind the paper:
- **the synthetic load and PV pipeline:** reference-year profile, residual block bootstrap, Monte Carlo growth, scenario reduction with an upper-tail stratum, and the deterministic median path;
- **the capacity-expansion model:** a linopy reimplementation of MicroGridsPy-SESAM solved with Gurobi, plus the parameter files of every design reported in the paper;
- **the public data** the pipeline uses.

## What is withheld, and why

The load data are **operating records of the island's utility**: the hourly genset log (Dec 2024; Apr–Aug 2026) and the daily peak and off-peak readings (Aug 2025–Jul 2026). They are not public and are not included.

For the same reason, this repository also omits everything derived from those records:
- the 200-member demand ensemble;
- the twelve reduced scenarios and the median path (`Demand_*.csv`, `merged_scenarios_*.json`);
- the hourly model results;
- the HOMER Pro files, which contain the synthetic load.

The code that builds these files is included, so the method can be inspected in full.

## Contents

| Folder | What it holds |
|---|---|
| `synthetic_data_code/` | `rekonstruksi_beban.py`: daily shape, monthly window targets, growth rates. `block_bootstrap.py`: hour-aligned moving-block bootstrap. `_build_demand_merged.py`: the 200-member ensemble, tail stratum, k-means reduction, mean-matched weights and median path (paper, Sections 3.1–3.5). `pv_simulation_multi_month.py`: the PV reference array simulated from NASA POWER weather (Section 3.4). |
| `model/` | `Code/linopy_port/`: the linopy model, sizing and certification runners. `Code/Model/`: the original Pyomo MicroGridsPy-SESAM model, whose data handling the linopy port reuses. Also run scripts, genset-count queue files (`queues/`) and conda environments (`environment.yml` for a workstation, `environment_hpc.yml` for the cluster runs). |
| `model_inputs/` | The MicroGridsPy parameter file (`.dat`) of every design in the paper's Tables 5 and 6 (1- and 12-scenario; 1, 3 and 5 steps; the seven sensitivities). Also their site files (`sites/*.conf`), the template `.dat` and `_build_dea2025c_family.py`, which generates the family. |
| `data_public/` | NASA POWER hourly weather at 7.68°S, 113.25°E (`nasa_power_full_year_2024.json` and three monthly `POWER_Point_Hourly_*.csv` files). `annual_peaks_2014_2023.csv`: annual peak demand from the master's thesis on the same island cited in the paper. `simulated_pv_full_year_2024.csv`: the PV year simulated from the public weather data. |

## What can be reproduced

**A. PV year (public data only).** The script reads and writes in the current folder, so run it in a scratch folder:

```bash
mkdir -p work/pv
cp data_public/nasa_power_full_year_2024.json data_public/POWER_Point_Hourly_*.csv work/pv/
cd work/pv && python ../../synthetic_data_code/pv_simulation_multi_month.py
```

The resulting `simulated_pv_full_year_2024.csv` is identical to the file in `data_public/`. The script needs Python ≥ 3.10 with pandas and NumPy.

**B. Synthetic demand.** `_build_demand_merged.py` needs the withheld operating records. Its docstring and `rekonstruksi_beban.py` document the expected files and columns. The builder is seeded and refuses to overwrite its outputs.

**C. Capacity-expansion runs.** These need demand and PV files with the names each site file gives (`SITE_DEMAND`, `SITE_RES`):
- **Demand:** semicolon-separated, 8,760 hourly rows, one column per scenario-year. The columns run scenario by scenario, with years 1–20 inside each, so an S-scenario file has 20·S columns.
- **PV:** comma-separated, 8,760 rows, one column per scenario, giving the output per installed PV unit.

To set up a run:

```bash
conda env create -f model/environment.yml          # or environment_hpc.yml; Gurobi licence required
cp -r model run && mkdir -p run/sites
cp model_inputs/*.dat <your demand and PV csv files> run/Code/Inputs/
cp model_inputs/sites/*.conf run/sites/
cd run
export PYTHON=/path/to/envs/mgpy/python
bash submit_linopy_site.sh gili_ketapang_1sc_20y_MILP_1step_dieselops_dea2025c --check   # pre-flight
```

Each design is **sized** first, with hourly commitment relaxed and the genset count fixed and enumerated. It is then **certified**: re-solved with its capacities pinned and every commitment variable integral (paper, Section 4). `run_linopy_interactive.sh` and `run_linopy_queue.sh` document the flags used. The twelve-scenario designs need about 150 GB of memory.

## Licence and credits

- The model code derives from MicroGridsPy-SESAM (Politecnico di Milano) and is distributed under the **European Union Public Licence v1.2** (`LICENSE`); its original authors are listed in `AUTHORS`. The pipeline code in `synthetic_data_code/` is released under the same licence.
- NASA POWER data: courtesy of the NASA Langley Research Center POWER Project.
- Techno-economic inputs in the `.dat` files come from the Indonesian technology catalogue and published utility fuel-price statistics, as cited in the paper.
