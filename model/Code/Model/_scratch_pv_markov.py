"""
Step 2: Markov-chain PV scenario generator, built on the 10 real years produced by
_scratch_pv_fetch_test.py (Code/Inputs/_scratch_pv_multiyear_raw.csv).

Architecture mirrors notebooks/synthetic_load_single_draw.ipynb's validated demand-side
Markov chain (quantile-binned levels, hour-conditioned transition counts, value-pool
resampling to avoid bin-midpoint flattening) with one deliberate difference: conditioned
on (month, hour) instead of hour alone. Demand's chain only had 1 real month of training
data, so it needed a *separate* ERA5-CDH seasonal-magnitude-scaling step bolted on after
the walk. PV has 10 real years pooled here -- enough samples per (month, hour) cell
(~300 before binning) to let the transition matrix itself carry seasonal structure
(monsoon cloud cover, day-length) directly, with no separate scaling step needed. Night
hours fall out naturally: real PV output is already ~0 then, so the learned bins/pools for
those (month, hour) cells are ~0 too.
"""
import numpy as np
import pandas as pd
from collections import defaultdict

RAW_PATH = "../Inputs/_scratch_pv_multiyear_raw.csv"
N_BINS = 8
N_SCENARIOS_9SC = 9
N_SCENARIOS_6SC = 6
MASTER_SEED = 2026  # matches this project's demand-pipeline convention (see site conf headers)

raw = pd.read_csv(RAW_PATH)
years = list(raw.columns)
print(f"Loaded {len(years)} real years, {len(raw)} rows each: {years}")

# Build a long (month, hour, value) pool from all 10 real years, tagging each hour with its
# calendar month/hour via a synthetic 2025 (non-leap) index -- matches every leap year down
# to 8760 by dropping Dec 31 for 366-day years (consistent with this project's own
# TMY code, RE_calculation.py:474-477, which drops Feb 29 the same way for the same reason).
records = []
for yr in years:
    col = raw[yr].dropna().to_numpy()
    if len(col) == 8784:
        # drop last day (Dec 31) to normalize every year to 8760 hours before tagging
        col = col[:8760]
    idx = pd.date_range("2025-01-01", periods=8760, freq="h")
    records.append(pd.DataFrame({"month": idx.month, "hour": idx.hour, "value": col}))
pool = pd.concat(records, ignore_index=True)
print(f"Pooled {len(pool)} real hourly observations across {len(years)} years")

# Quantile-bin per (month, hour) cell -- daytime cells get real spread-based bins, nighttime
# cells (all-zero) collapse to a single bin automatically since qcut on constant data is
# handled by the dedup-drop below.
pool["bin"] = -1
for (m, h), grp in pool.groupby(["month", "hour"]):
    vals = grp["value"]
    if vals.nunique() <= 1:
        pool.loc[grp.index, "bin"] = 0  # night / constant-zero cell -> single bin
    else:
        try:
            pool.loc[grp.index, "bin"] = pd.qcut(vals, N_BINS, labels=False, duplicates="drop")
        except ValueError:
            pool.loc[grp.index, "bin"] = 0

# Transition counts and value pools, keyed by (month, hour, bin_from) -- direct analogue of
# the demand notebook's (hour, bin_from) keying, with month added.
transition_counts = defaultdict(lambda: defaultdict(int))
value_pool = defaultdict(list)

pool_sorted = pool.reset_index(drop=True)
# Consecutive-row transitions within each real year (10 separate 8760-hour walks, not one
# continuous 87600-hour walk -- avoids fabricating a fake Dec31->Jan1 cross-year transition).
for yr_i in range(len(years)):
    start = yr_i * 8760
    seg = pool_sorted.iloc[start:start + 8760].reset_index(drop=True)
    for i in range(len(seg) - 1):
        m_now, h_now, b_now = seg.loc[i, ["month", "hour", "bin"]]
        b_next = seg.loc[i + 1, "bin"]
        transition_counts[(m_now, h_now, b_now)][b_next] += 1

for _, r in pool_sorted.iterrows():
    value_pool[(r["month"], r["hour"], r["bin"])].append(r["value"])

marginal_by_month_hour = {}
for (m, h), grp in pool_sorted.groupby(["month", "hour"]):
    counts = grp["bin"].value_counts()
    marginal_by_month_hour[(m, h)] = (counts.index.to_numpy(), (counts / counts.sum()).to_numpy())

n_states_learned = len(transition_counts)
n_states_possible = 12 * 24 * N_BINS
print(f"Learned transitions for {n_states_learned} (month,hour,bin) states out of {n_states_possible} possible")


def simulate_year(rng):
    """Walk the (month,hour)-conditioned Markov chain for 8760 hours."""
    idx = pd.date_range("2025-01-01", periods=8760, freq="h")
    months = idx.month.to_numpy()
    hours = idx.hour.to_numpy()

    choices, weights = marginal_by_month_hour[(months[0], hours[0])]
    current_bin = rng.choice(choices, p=weights)
    bin_seq = [current_bin]

    for t in range(1, 8760):
        m, h = months[t], hours[t]
        m_prev, h_prev = months[t - 1], hours[t - 1]
        key = (m_prev, h_prev, current_bin)
        if key in transition_counts and len(transition_counts[key]) > 0:
            next_bins = list(transition_counts[key].keys())
            next_weights = np.array(list(transition_counts[key].values()), dtype=float)
            next_weights /= next_weights.sum()
            current_bin = rng.choice(next_bins, p=next_weights)
        else:
            choices, weights = marginal_by_month_hour[(m, h)]
            current_bin = rng.choice(choices, p=weights)
        bin_seq.append(current_bin)

    values = np.empty(8760)
    for t in range(8760):
        m, h, b = months[t], hours[t], bin_seq[t]
        p = value_pool.get((m, h, b))
        if not p:
            p = value_pool.get((m, h, 0), [0.0])
        values[t] = rng.choice(p)
    return np.maximum(values, 0.0)  # physical floor, matches step 1's own clipping


def build_scenarios(n_scenarios, seed_offset=0):
    scenarios = []
    for s in range(n_scenarios):
        rng = np.random.default_rng(MASTER_SEED + seed_offset + s)
        scenarios.append(simulate_year(rng))
    return np.array(scenarios)  # [scenario, hour]


if __name__ == "__main__":
    real_annual_kwh = [raw[yr].dropna().sum() / 1000 for yr in years]
    print(f"\nReal years annual energy range: {min(real_annual_kwh):.1f} - {max(real_annual_kwh):.1f} kWh/module")

    for label, n in [("9sc", N_SCENARIOS_9SC), ("6sc", N_SCENARIOS_6SC)]:
        scen = build_scenarios(n, seed_offset=0 if label == "9sc" else 1000)
        annual = scen.sum(axis=1) / 1000
        print(f"\n{label}: {n} synthetic scenarios, annual energy (kWh/module): "
              f"min={annual.min():.1f} max={annual.max():.1f} mean={annual.mean():.1f} "
              f"(real range was {min(real_annual_kwh):.1f}-{max(real_annual_kwh):.1f})")
        # scenario columns must NOT be identical -- the whole point of this exercise
        n_unique = len(np.unique(annual.round(1)))
        print(f"  {n_unique} of {n} scenarios have distinct annual energy (want: all {n})")

    # --- Write candidate RES_Time_Series files, matching the existing production format
    # exactly (kWh/module/hour, comma-separated, blank index column 0, integer scenario
    # headers 1..N) -- written as NEW candidate files, not overwriting the certified
    # production ones, pending review/validation (this project's own established practice
    # after the 2026-07-28 shared-RES-file incident).
    for label, n, offset in [("9sc", N_SCENARIOS_9SC, 0), ("6sc", N_SCENARIOS_6SC, 1000)]:
        scen = build_scenarios(n, seed_offset=offset) / 1000.0  # Wh -> kWh
        out = pd.DataFrame(scen.T, columns=[str(i + 1) for i in range(n)])
        out.index = out.index + 1
        out_path = f"../Inputs/RES_Time_Series_{label}_gili_ketapang_stochasticpv.csv"
        out.to_csv(out_path)
        print(f"Wrote {out_path}: shape={out.shape}, col sums (kWh)={out.sum().round(2).tolist()}")
