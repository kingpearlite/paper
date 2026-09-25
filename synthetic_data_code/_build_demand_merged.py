"""Builds the merged Gili Ketapang demand/PV scenario inputs for the paper update (2026-09-11),
combining two companion studies instead of the paper's earlier one-month Markov-chain demand:

  * Load Forecasting study (<workdir>/Load Forecasting, metodologi_rekonstruksi_beban.tex):
    daily-shape extraction p(h,tau) from the genset log (Dec 2024 + Apr-Aug 2026, outage hours
    excluded), monthly WBP/LWBP targets from the operator's daily readings (Aug 2025 - Jul 2026),
    and the Monte Carlo growth process g_t ~ N(mu_g, sigma_g^2), i.i.d. per year and trial,
    fitted to the 2014-2023 annual peaks (mu_g = 3.01%, sigma_g = 9.66%).
  * DataScarceMicrogrid study (<workdir>/DataScarceMicrogrid,
    paper_forecast_to_dispatch_scenarios.tex): hour-of-day-aligned moving-block bootstrap
    (block_bootstrap.py), month-and-hour PV climatology from the NASA POWER 2024 simulation, and
    temporal-aware K-means scenario reduction (TAKSR, Osifeko & Munda 2025).

Method (one ensemble member i = 1..N_MEMBERS, one reference calendar year 2026, 8760 h):
  1. Backbone m(t) = p(h,tau) * target[month][window] / pbar[window](tau)  (Load Forecasting,
     Eq. 4), for every hour of 2026. The seasonal cycle comes from the monthly targets, so the
     DataScarce study's residual inflation factor (kappa = 1.189, a PV-derived proxy for the six
     unmeasured months) is not needed.
  2. Within-year variability: multiplicative residuals r = load / m(t) on the 2026 measured hours
     only (the targets are 2025/26 readings; Dec 2024 would mix in a different year's level).
     Each window's residuals are normalised to mean 1, so the synthetic window means stay anchored
     to the targets as in the Load Forecasting study; the raw bias is printed as a diagnostic.
     Missing/outage hours get r = 1. Residuals are resampled in 168 h blocks aligned to hour of
     day, drawn afresh for every project year (per-year regeneration, Mandelli et al. 2017).
  3. Growth: year 1 = 2026 level; years 2..20 scale by cumprod(1 + g_{i,t}), one path per member.
  4. PV: DataScarce month-and-hour climatology + 24 h within-month aligned blocks, one year per
     member (MicroGridsPy uses one RES year per scenario), divided by the simulated plant's
     798 kW DC rating to a per-kW AC profile. PV capacity itself is a decision variable of the
     sizing model; 798 kW only normalises the simulation.
  5. Joint TAKSR, K = 11 fixed: load member i is paired with PV member i (their lag-0 residual
     correlation is 0.033, so independent pairing is supported). Features: the six TAKSR features
     of the year-1 load and of the PV year (constant features dropped), plus the growth group --
     the member's 20 annual energies and its highest hourly load over 20 years -- so the growth
     uncertainty drives the clustering, not only within-year texture. The three feature groups
     (load texture, PV texture, growth) are balanced to equal total variance. Representatives are
     the members nearest each centroid. Their cluster-size weights are then calibrated by
     least-squares moment matching (Hoyland & Wallace 2001) to the ensemble's mean annual-energy
     path and mean 20-year peak, staying near the cluster-size weights, every weight >= W_MIN.
     Both weight sets and the remaining per-year errors are recorded in the summary JSON.
     Upper tail: before the clustering, the members whose 20-year peak is at or above the
     ensemble's P90 (TAIL_QUANTILE) are set aside as a stratum represented by one extra scenario
     (the stratum member with the median 20-year peak) with its weight fixed at the stratum's probability
     (0.10); TAKSR then reduces the other members to K = 11, giving K + 1 = 12 scenarios. Without
     it the reduced year-20 P90 peak was 12% below the ensemble's.
  6. Deterministic scheme (1 scenario): the per-year median of the members' growth factors
     (the Monte Carlo P50 path) on one fixed-seed load draw, with the PV member of median annual
     energy.

MicroGridsPy RES units: RES_Energy_Production = cf * RES_Inverter_Efficiency * RES_Units, one unit
= RES_Nominal_Capacity kW, so cf = per-kW AC output * RES_Nominal_Capacity / RES_Inverter_Efficiency
(the DataScarce simulation is already AC). Both constants are read from the template .dat.

Run with a Python >= 3.10 environment that has pandas, scipy and scikit-learn (the Load
Forecasting modules use `X | None` annotations; the mgpy env is 3.9 and cannot import them):
    C:/Users/<user>/miniconda3/python.exe _build_demand_merged.py      (from Code/Inputs/)
Source locations can be overridden with MGPY_LF_DIR / MGPY_DS_DIR. Refuses to overwrite outputs.

Growth-calibration sensitivity (2026-09-17): MGPY_GROWTH_FIRST_YEAR=<year> fits mu_g/sigma_g only to the
annual peaks from that year on (unset: the whole record), and MGPY_MERGED_SUFFIX=<suffix> is appended to
every output name so the base files are never touched. The present two-unit plant entered service in
May 2017 (Falfi 2025, ITB thesis), so the variant fitted to its full years (2018 on) is built with
    MGPY_GROWTH_FIRST_YEAR=2018 MGPY_MERGED_SUFFIX=_g24h C:/Users/<user>/miniconda3/python.exe _build_demand_merged.py
(The suffix is named after a 12 h -> 24 h supply change reported informally; that claim could not be
verified and is no longer used as the justification.)

Tail-stratum diagnostic (2026-09-21): MGPY_TAIL_STRATUM=0 skips the upper-tail stratum and reduces all
N_MEMBERS members to K scenarios, reproducing the pre-stratum build whose year-20 P90 peak sat 12% below
the ensemble's. It exists to quantify what the stratum costs and buys across the whole year-20 peak
distribution; the summary JSON's peak_y20_percentiles_kw then describes a K-scenario set. Outputs are
named K instead of K+1, so pair it with a suffix to leave the shipped files alone:
    MGPY_TAIL_STRATUM=0 MGPY_MERGED_SUFFIX=_notail C:/Users/<user>/miniconda3/python.exe _build_demand_merged.py

Demand level, tail rule and feature fix (2026-09-24, the dea2025k re-run). Four switches, each defaulting
to the behaviour above, so every earlier build still reproduces:
  * MGPY_LEVEL_SOURCE=kwh -- the level comes from the daily kWh production of Cummins 2 + 3 in the
    monthly PLTD reports (LF_DIR/load data/produksi_kwh_harian.csv, meter-consistent on all 365 days),
    not from the daily Siang/Malam kW readings, 91 of whose 365 days are template copies and whose
    level sat 6.7% below the recorded production. Backbone m(t) = p(h,tau) * Ebar(month, tau) / 24,
    with Ebar the mean daily kWh of that month and day type; the residual pool is normalised once
    (per-window normalisation would move the daily energy off the record), and the year-1 check is
    the weighted monthly energy against the backbone's.
  * MGPY_TAKSR_DEV -- the sixth TAKSR feature: signed (default, np.mean(x - ens_mean), which is
    x.mean() minus a constant and so counts the mean twice after z-scoring), abs (the mean ABSOLUTE
    deviation from the ensemble mean, the intended feature; used for dea2025k) or drop (left out).
  * MGPY_TAIL_RULE -- the upper-tail representative: peak_median (default: median 20-year peak),
    growth_medoid (nearest the stratum's mean on the moments the weights are calibrated to -- the
    20 annual energies and the 20-year peak, z-scored within the stratum and balanced like the TAKSR
    groups; used for dea2025k), peak_energy (nearest the stratum's median 20-year peak and median
    20-year energy; tried and not used), or peak_q25 / peak_q75 (the stratum members at those
    20-year-peak quantiles, the tail-representative sensitivity of the review's AWO-002 A). Why: the
    median-peak member can sit far out in energy, and a 7% level change swapped it (member 173 ->
    122) for one costing 19% more; peak_energy also swapped (188 -> 150), growth_medoid picks 122
    under both level sources and both growth fits.
  * MGPY_TAIL_PV_REPAIR=1 -- also writes five RES files in which the tail scenario's PV year is
    replaced by the PV member at the min / P25 / P50 / P75 / max annual yield (AWO-002 B). Demand
    and weights are unchanged, since the weights are fitted to demand moments only.
The dea2025k inputs (Code/Inputs/_build_dea2025k_family.py reads them) are built with
    S="MGPY_LEVEL_SOURCE=kwh MGPY_TAKSR_DEV=abs"; PY=C:/Users/<user>/miniconda3/python.exe
    env $S MGPY_TAIL_RULE=growth_medoid MGPY_TAIL_PV_REPAIR=1 MGPY_MERGED_SUFFIX=_kwh $PY _build_demand_merged.py
    env $S MGPY_TAIL_RULE=growth_medoid MGPY_GROWTH_FIRST_YEAR=2018 MGPY_MERGED_SUFFIX=_kwh_g24h $PY _build_demand_merged.py
    env $S MGPY_TAIL_RULE=peak_q25 MGPY_MERGED_SUFFIX=_kwh_tailq25 $PY _build_demand_merged.py
    env $S MGPY_TAIL_RULE=peak_q75 MGPY_MERGED_SUFFIX=_kwh_tailq75 $PY _build_demand_merged.py
"""
import json
import os
import re
import subprocess
import sys

import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
LF_DIR = os.environ.get("MGPY_LF_DIR", r"<workdir>/Load Forecasting")
DS_DIR = os.environ.get("MGPY_DS_DIR", r"<workdir>/DataScarceMicrogrid")
sys.path.insert(0, LF_DIR)
sys.path.insert(0, DS_DIR)

import rekonstruksi_beban as rb  # noqa: E402  (Load Forecasting study)
from block_bootstrap import regular_hourly_grid, aligned_start_pools, aligned_block_bootstrap  # noqa: E402
from scipy.optimize import minimize  # noqa: E402
from sklearn.cluster import KMeans  # noqa: E402
from sklearn.metrics import adjusted_rand_score, silhouette_score  # noqa: E402
from sklearn.preprocessing import StandardScaler  # noqa: E402

N_MEMBERS = 200
N_YEARS = 20
K = 11
YEAR = 2026                 # reference calendar year (non-leap, 8760 h), as in the Load Forecasting study
LOAD_BLOCK = 168
PV_BLOCK = 24
PV_PDC_KW = 798.0           # DataScarce pv_simulation*.py PDC0_W, used only to normalise to per kW
N_BOOT_ARI = 50
GROWTH_YEARS = tuple(range(1, 21))  # the whole annual-energy path is the TAKSR growth feature group
W_MIN = 0.01                        # floor on a calibrated scenario weight
CAL_LAMBDA = 0.1                    # pull of the calibrated weights towards the cluster-size weights
TAIL_QUANTILE = 0.90                # members with a 20-year peak >= this ensemble quantile form the upper-tail stratum
TAIL_STRATUM = os.environ.get("MGPY_TAIL_STRATUM", "1") != "0"  # 0 reproduces the pre-stratum K-scenario build
N_PROB = K + (1 if TAIL_STRATUM else 0)  # K TAKSR scenarios for the body + 1 upper-tail scenario
SEED = 20260911
TEMPLATE_DAT = "Parameters_9sc_20y_gili_ketapang_capexp5_MILP_dieselops.dat"
GROWTH_FIRST_YEAR = os.environ.get("MGPY_GROWTH_FIRST_YEAR")
SUFFIX = os.environ.get("MGPY_MERGED_SUFFIX", "")
LEVEL_SOURCE = os.environ.get("MGPY_LEVEL_SOURCE", "readings")   # readings | kwh (see the docstring)
TAKSR_DEV = os.environ.get("MGPY_TAKSR_DEV", "signed")          # signed | abs | drop (see the docstring)
TAIL_RULE = os.environ.get("MGPY_TAIL_RULE", "peak_median")
TAIL_PV_REPAIR = os.environ.get("MGPY_TAIL_PV_REPAIR", "0") == "1"
TAIL_RULES = ("peak_median", "peak_energy", "growth_medoid", "peak_q25", "peak_q75")
PV_REPAIR_QUANTILES = {"min": 0.0, "p25": 0.25, "p50": 0.5, "p75": 0.75, "max": 1.0}  # of annual PV yield
if TAKSR_DEV not in ("signed", "abs", "drop"):
    sys.exit(f"MGPY_TAKSR_DEV must be signed, abs or drop, got {TAKSR_DEV!r}")
if LEVEL_SOURCE not in ("readings", "kwh"):
    sys.exit(f"MGPY_LEVEL_SOURCE must be readings or kwh, got {LEVEL_SOURCE!r}")
if TAIL_RULE not in TAIL_RULES:
    sys.exit(f"MGPY_TAIL_RULE must be one of {TAIL_RULES}, got {TAIL_RULE!r}")
if (TAIL_PV_REPAIR or TAIL_RULE != "peak_median") and not TAIL_STRATUM:
    sys.exit("MGPY_TAIL_RULE / MGPY_TAIL_PV_REPAIR need the upper-tail stratum (MGPY_TAIL_STRATUM=1)")

OUT = {
    "demand_prob": f"Demand_{N_PROB}sc_20y_gili_ketapang_merged{SUFFIX}.csv",
    "res_prob": f"RES_Time_Series_{N_PROB}sc_gili_ketapang_merged{SUFFIX}.csv",
    "demand_det": f"Demand_1sc_20y_gili_ketapang_merged{SUFFIX}.csv",
    "res_det": f"RES_Time_Series_1sc_gili_ketapang_merged{SUFFIX}.csv",
    "summary": f"merged_scenarios_gili_ketapang{SUFFIX}.json",
}
if TAIL_PV_REPAIR:
    OUT.update({f"res_tailpv_{k}": f"RES_Time_Series_{N_PROB}sc_gili_ketapang_merged{SUFFIX}_tailpv{k}.csv"
                for k in PV_REPAIR_QUANTILES})


def dat_indexed_value(path, name):
    with open(path, encoding="utf-8") as f:
        text = f.read()
    m = re.search(rf"^param: {name}\s*:=.*\n1\t([0-9.eE+-]+)", text, flags=re.M)
    if not m:
        sys.exit(f"could not read indexed param {name} from {path}")
    return float(m.group(1))


def git_head(path):
    try:
        head = subprocess.run(["git", "-C", path, "rev-parse", "--short", "HEAD"],
                              capture_output=True, text=True, check=True).stdout.strip()
        dirty = subprocess.run(["git", "-C", path, "status", "--porcelain"],
                               capture_output=True, text=True, check=True).stdout.strip()
        return head + ("-dirty" if dirty else "")
    except (OSError, subprocess.CalledProcessError):
        return "unknown"


def load_genset_log_rounded(path):
    """Same cleaning as rekonstruksi_beban.load_genset_log(path, treat_outage_as="missing") --
    outage rows dropped, empty readings dropped, duplicate rows of simultaneously running units
    averaged -- except that timestamps are ROUNDED to the nearest hour instead of floored.
    The logger records with -1 s jitter ('10:59:59' is the 11:00 reading); flooring maps it onto
    10:00, where it collides with the real 10:00 reading, so 881 hours become the average of two
    different readings (spread up to 203 kW) and those hours' neighbours vanish (2,624 hours kept
    instead of 3,528). Rounding gives 3,528 hours with no collisions, identical to the
    DataScarce study's load_clean_2026.py series on every common hour."""
    df = pd.read_csv(path, parse_dates=["tanggal"])
    df.loc[rb._flag_outage_rows(df), "beban_total_kw"] = np.nan
    df["beban_total_kw"] = pd.to_numeric(df["beban_total_kw"], errors="coerce")
    df = df.dropna(subset=["beban_total_kw"])
    load_kw = df.groupby(df["tanggal"].dt.round("h"))["beban_total_kw"].mean()
    load_kw.index.name = None
    return load_kw.to_frame("load_kw")


def kwh_levels(path):
    """Mean daily kWh (Cummins 2 + 3) per calendar month and day type, from ekstrak_produksi_harian.py."""
    d = pd.read_csv(path, parse_dates=["date"])
    d["kwh"] = d["prod_c2_kwh"].fillna(0) + d["prod_c3_kwh"].fillna(0)
    d["tau"] = np.where(d["date"].dt.dayofweek >= 5, "weekend", "weekday")
    lv = d.groupby([d["date"].dt.month, "tau"])["kwh"].mean()
    return {(int(m), t): float(v) for (m, t), v in lv.items()}


def backbone(index, shape, targets):
    """Load Forecasting Eq. 4, vectorised: p(h,tau) * target / pbar(window, tau). With
    MGPY_LEVEL_SOURCE=kwh the targets are mean daily kWh per (month, day type) and the backbone is
    p(h,tau) * level / 24 (p has a daily mean of 1, so each day carries the recorded mean energy)."""
    tau = np.where(index.dayofweek >= 5, "weekend", "weekday")
    hour = index.hour.values
    p = np.array([shape.at[h, t] for h, t in zip(hour, tau)])
    if LEVEL_SOURCE == "kwh":
        lvl = np.array([targets[(m, t)] for m, t in zip(index.month, tau)])
        return p * lvl / 24.0
    wbp = np.isin(hour, rb.WBP_HOURS)
    pbar = {"WBP": shape.loc[rb.WBP_HOURS].mean(), "LWBP": shape.loc[rb.LWBP_HOURS].mean()}
    tgt = np.array([targets[m]["WBP" if w else "LWBP"] for m, w in zip(index.month, wbp)])
    pb = np.array([pbar["WBP" if w else "LWBP"][t] for w, t in zip(wbp, tau)])
    return p * tgt / pb


def taksr_features(x, day_mask, ens_mean):
    # np.mean(x - ens_mean) is x.mean() minus a constant, a duplicate of the first feature after
    # z-scoring; MGPY_TAKSR_DEV=abs uses the mean absolute deviation instead, =drop leaves it out.
    f = [x.mean(), x.std(), x.min(), x.max() - x.min(), x[day_mask].mean() - x[~day_mask].mean()]
    if TAKSR_DEV == "drop":
        return f
    return f + [np.mean(np.abs(x - ens_mean)) if TAKSR_DEV == "abs" else np.mean(x - ens_mean)]


def pick_tail(members, peak20, energy_y):
    """Upper-tail representative under TAIL_RULE; members are the stratum's indices."""
    by_peak = members[np.argsort(peak20[members])]
    if TAIL_RULE == "peak_median":
        return int(by_peak[(len(by_peak) - 1) // 2])
    if TAIL_RULE in ("peak_q25", "peak_q75"):
        q = 0.25 if TAIL_RULE == "peak_q25" else 0.75
        return int(by_peak[int(round(q * (len(by_peak) - 1)))])
    if TAIL_RULE == "growth_medoid":
        # Nearest the stratum's MEAN on the moments the weights are calibrated to (the 20 annual
        # energies and the 20-year peak), z-scored within the stratum and balanced like the TAKSR
        # groups (energy block / sqrt(20), peak / 1), so the fixed-weight tail carries its stratum's
        # mean demand path into the weight calibration.
        e = energy_y[members]
        ze = (e - e.mean(axis=0)) / e.std(axis=0) / np.sqrt(e.shape[1])
        zp = (peak20[members] - peak20[members].mean()) / peak20[members].std()
        return int(members[np.argmin(np.hypot(np.linalg.norm(ze, axis=1), zp))])
    pk, en = peak20[members], energy_y[members].sum(axis=1)
    dist = np.hypot((pk - np.median(pk)) / pk.std(), (en - np.median(en)) / en.std())
    return int(members[np.argmin(dist)])


def window_means(series_kw, index):
    df = pd.DataFrame({"kw": series_kw, "month": index.month,
                       "w": np.where(np.isin(index.hour, rb.WBP_HOURS), "WBP", "LWBP")})
    return df.groupby(["month", "w"])["kw"].mean()


def calibrate_weights(w0, moments, targets, w_min, lam, fixed=()):
    """Moment-matching weight calibration (Hoyland & Wallace 2001), least-squares form: minimises
    the squared relative error of every weighted moment against the full ensemble's mean, plus
    lam * ||w - w0||^2 to stay near the TAKSR cluster-size weights w0, subject to sum(w) = 1 and
    w >= w_min. With 11 weights and 21 moments (20 annual energies + the 20-year peak) an exact
    match is not generally possible; the residual errors are reported. Representatives unchanged."""
    def objective(w):
        return (((moments.T @ w) / targets - 1.0) ** 2).sum() + lam * ((w - w0) ** 2).sum()
    cons = [{"type": "eq", "fun": lambda w: w.sum() - 1.0}]
    # Positions in `fixed` keep their w0 weight (the upper-tail stratum's probability).
    bounds = [(w0[i], w0[i]) if i in fixed else (w_min, 1.0) for i in range(len(w0))]
    res = minimize(objective, w0, method="SLSQP", bounds=bounds, constraints=cons,
                   options={"ftol": 1e-14, "maxiter": 1000})
    if not res.success:
        sys.exit(f"weight calibration failed: {res.message}")
    w = np.round(res.x, 6)
    free = [i for i in range(len(w)) if i not in fixed]
    w[max(free, key=lambda i: w[i])] += round(1.0 - w.sum(), 6)  # exact sum of 1 at 6 decimals
    return w


def weighted_quantile(values, weights, q):
    order = np.argsort(values)
    cum = np.cumsum(weights[order])
    # The tolerance keeps a cumulative weight that reaches q exactly (the body's 0.90 below the 0.10
    # tail) on the lower value; without it 6-decimal rounding decided which side P90 fell on.
    return float(values[order][np.searchsorted(cum, q * cum[-1] - 1e-9)])


def main():
    paths = {k: os.path.join(HERE, v) for k, v in OUT.items()}
    clash = [p for p in paths.values() if os.path.exists(p)]
    if clash:
        sys.exit("Refusing to overwrite:\n  " + "\n  ".join(clash))

    template = os.path.join(HERE, TEMPLATE_DAT)
    res_unit_kw = dat_indexed_value(template, "RES_Nominal_Capacity")
    res_inv_eff = dat_indexed_value(template, "RES_Inverter_Efficiency")

    # ---- 1. backbone (Load Forecasting study) ----
    df_all = load_genset_log_rounded(os.path.join(LF_DIR, "load data", "harian_gilket_gabungan.csv"))
    shape = rb.extract_daily_shape(df_all)
    if LEVEL_SOURCE == "kwh":
        targets = kwh_levels(os.path.join(LF_DIR, "load data", "produksi_kwh_harian.csv"))
        print(f"Level: daily kWh production (Cummins 2 + 3), mean {np.mean(list(targets.values())):.0f} kWh/day "
              f"over the 24 month x day-type cells")
    else:
        targets = rb.build_monthly_targets_from_daily(os.path.join(LF_DIR, "load data", "beban_puncak_harian.csv"))
    peaks = (rb.ANNUAL_PEAK if GROWTH_FIRST_YEAR is None
             else {y: v for y, v in rb.ANNUAL_PEAK.items() if y >= int(GROWTH_FIRST_YEAR)})
    g_yoy = rb.compute_yoy_growth_rates(peaks)
    mu_g, sigma_g = float(g_yoy.mean()), float(g_yoy.std(ddof=1))
    print(f"Growth fitted to annual peaks {min(peaks)}-{max(peaks)}: {len(g_yoy)} year-on-year changes "
          f"{[round(100 * float(v), 2) for v in g_yoy]} %")
    idx = pd.date_range(f"{YEAR}-01-01", periods=8760, freq="h")
    m_year = backbone(idx, shape, targets)
    hour_of_day, month_of_hour = idx.hour.values, idx.month.values
    day_mask = (hour_of_day >= 6) & (hour_of_day < 18)
    print(f"Backbone {YEAR}: energy {m_year.sum() / 1e3:.1f} MWh, peak {m_year.max():.1f} kW; "
          f"mu_g={mu_g:.4f}, sigma_g={sigma_g:.4f}")

    # ---- 2. multiplicative residual pool, 2026 measured hours only ----
    obs = df_all[df_all.index.year == YEAR]
    grid = regular_hourly_grid(obs.rename_axis("timestamp").reset_index(), "load_kw")
    gidx = pd.DatetimeIndex(grid["timestamp"])
    ratio = grid["load_kw"].values / backbone(gidx, shape, targets)
    gwbp = np.isin(gidx.hour, rb.WBP_HOURS)
    raw_bias = {}
    for name, mask in (("WBP", gwbp), ("LWBP", ~gwbp)):
        valid = mask & ~np.isnan(ratio)
        raw_bias[name] = float(np.nanmean(ratio[valid]))
        if LEVEL_SOURCE == "readings":
            ratio[valid] = ratio[valid] / raw_bias[name]
    if LEVEL_SOURCE == "kwh":
        # One normalisation over the whole pool keeps the daily-energy anchor; per window it would
        # move the energy off the record. The per-window biases stay as diagnostics.
        raw_bias["all"] = float(np.nanmean(ratio))
        ratio = ratio / raw_bias["all"]
    n_missing = int(np.isnan(ratio).sum())
    ratio = np.where(np.isnan(ratio), 1.0, ratio)
    pools = aligned_start_pools(gidx.hour.values, grid["segment"].values, LOAD_BLOCK)
    norm = (f"all {raw_bias['all']:.3f} (normalised out once)" if LEVEL_SOURCE == "kwh" else "(normalised out)")
    print(f"Residual pool: {len(grid)} h in {grid['segment'].nunique()} segment(s), {n_missing} missing -> r=1; "
          f"raw measured/target bias WBP {raw_bias['WBP']:.3f}, LWBP {raw_bias['LWBP']:.3f} {norm}")

    # ---- 3. growth paths + per-year regenerated load ----
    rng_g = np.random.default_rng([SEED, 1])
    rng_l = np.random.default_rng([SEED, 2])
    g_paths = rng_g.normal(mu_g, sigma_g, size=(N_MEMBERS, N_YEARS - 1))
    factor = np.ones((N_MEMBERS, N_YEARS))
    factor[:, 1:] = np.cumprod(1.0 + g_paths, axis=1)
    load = np.empty((N_MEMBERS, N_YEARS, 8760), dtype=np.float64)
    for i in range(N_MEMBERS):
        for y in range(N_YEARS):
            r = aligned_block_bootstrap(ratio, pools, LOAD_BLOCK, 8760, hour_of_day, rng_l)
            load[i, y] = np.clip(m_year * factor[i, y] * r, 0.0, None)

    # ---- 4. PV (DataScarce study), per kW AC ----
    pv_df = pd.read_csv(os.path.join(DS_DIR, "simulated_pv_full_year_2024.csv"), parse_dates=["timestamp"])
    pv_df["hour"], pv_df["month"] = pv_df["timestamp"].dt.hour, pv_df["timestamp"].dt.month
    clim = pv_df.groupby(["month", "hour"])["pv_ac_kw"].mean()
    pv_resid = (pv_df["pv_ac_kw"] - pv_df.set_index(["month", "hour"]).index.map(clim)).values
    pv_pools = aligned_start_pools(pv_df["hour"].values, (pv_df["timestamp"].dt.year * 100 + pv_df["month"]).values,
                                   PV_BLOCK, group=pv_df["month"].values)
    clim_year = np.array([clim.get((mm, hh), 0.0) for mm, hh in zip(month_of_hour, hour_of_day)])
    rng_p = np.random.default_rng([SEED, 3])
    pv_kw = np.empty((N_MEMBERS, 8760))
    for i in range(N_MEMBERS):
        pv_kw[i] = np.clip(clim_year + aligned_block_bootstrap(pv_resid, pv_pools, PV_BLOCK, 8760, hour_of_day,
                                                               rng_p, target_group=month_of_hour), 0.0, None)
    pv_per_kw = pv_kw / PV_PDC_KW
    night = clim_year == 0.0
    print(f"PV: {pv_per_kw.sum(axis=1).mean():.1f} kWh/kW-yr AC (sd {pv_per_kw.sum(axis=1).std():.1f}), "
          f"max at night hours {pv_per_kw[:, night].max():.3g}")

    # ---- 5. joint TAKSR, K fixed ----
    y1 = load[:, 0, :]
    ens_load, ens_pv = y1.mean(axis=0), pv_per_kw.mean(axis=0)
    energy_y = load.sum(axis=2) / 1e3  # MWh, (member, year)
    base = ("mean", "std", "min", "range", "day_night") + (() if TAKSR_DEV == "drop" else ("dev",))
    groups = {
        "load_texture": ([f"load_{n}" for n in base],
                         np.array([taksr_features(y1[i], day_mask, ens_load) for i in range(N_MEMBERS)])),
        "pv_texture": ([f"pv_{n}" for n in base],
                       np.array([taksr_features(pv_per_kw[i], day_mask, ens_pv) for i in range(N_MEMBERS)])),
        "growth": ([f"energy_y{y}" for y in GROWTH_YEARS] + ["peak_20y"],
                   np.column_stack([energy_y[:, y - 1] for y in GROWTH_YEARS] + [load.max(axis=(1, 2))])),
    }
    # Each group is z-scored, constant columns dropped, then scaled by 1/sqrt(columns) so every
    # group carries the same total variance. Without this the eleven within-year texture features
    # outvote the growth features and the reduced set misses the growth tails (first build:
    # reduced year-20 energy +4.8% vs the ensemble, P10 year-20 peak 884 vs 582 kW).
    blocks, dropped = [], []
    for gname, (cols, mat) in groups.items():
        keep = mat.std(axis=0) > 1e-12
        dropped += [c for c, k in zip(cols, keep) if not k]
        blocks.append(StandardScaler().fit_transform(mat[:, keep]) / np.sqrt(keep.sum()))
    z = np.hstack(blocks)
    # Upper-tail stratum (added after the fourth build, whose 11 TAKSR scenarios put the year-20
    # P90 peak 12% below the ensemble's): the members whose 20-year peak is at or above the
    # ensemble's TAIL_QUANTILE are set aside and represented by one scenario whose weight is fixed
    # at the stratum's probability (the representative rule is stated below). TAKSR reduces the
    # remaining members to K scenarios as before. MGPY_TAIL_STRATUM=0 skips the split entirely.
    peak20 = load.max(axis=(1, 2))
    tail = (peak20 >= np.quantile(peak20, TAIL_QUANTILE)) if TAIL_STRATUM else np.zeros(N_MEMBERS, bool)
    body, tail_members = np.flatnonzero(~tail), np.flatnonzero(tail)
    zb = z[body]
    km = KMeans(n_clusters=K, n_init=10, random_state=0).fit(zb)
    labels = km.labels_
    sil = float(silhouette_score(zb, labels))
    rng_b = np.random.default_rng([SEED, 4])
    aris = []
    for _ in range(N_BOOT_ARI):
        b = rng_b.integers(0, len(body), len(body))
        kb = KMeans(n_clusters=K, n_init=10, random_state=0).fit(zb[b])
        aris.append(adjusted_rand_score(labels, kb.predict(zb)))
    reps, weights = [], []
    for k in range(K):
        members = body[labels == k]
        reps.append(int(members[np.argmin(np.linalg.norm(z[members] - km.cluster_centers_[k], axis=1))]))
        weights.append(len(members) / N_MEMBERS)
    # The tail is represented by one stratum member chosen by TAIL_RULE (pick_tail). The default is
    # its conditional median on the quantity that defines it, the median 20-year peak.
    # (Nearest-centroid in the full feature space picked the ensemble's most extreme member,
    # 2,329 kW, in the fifth build.)
    if TAIL_STRATUM:
        energy20 = energy_y.sum(axis=1)
        tail_rep = pick_tail(tail_members, peak20, energy_y)
        tail_peak_rank = int((peak20[tail_members] < peak20[tail_rep]).sum()) + 1     # 1 = lowest in the stratum
        tail_energy_rank = int((energy20[tail_members] < energy20[tail_rep]).sum()) + 1
        reps.append(tail_rep)
        weights.append(len(tail_members) / N_MEMBERS)
    order = np.argsort([energy_y[r, -1] for r in reps])  # scenario 1 = lowest year-20 energy
    reps = [reps[j] for j in order]
    weights_cluster = np.array([weights[j] for j in order])
    tail_pos = reps.index(tail_rep) if TAIL_STRATUM else None
    # Cluster-size weights alone do not reproduce the ensemble's mean demand path: in the third
    # build the reduced set sat 6-9% low in years 2-3 and 3-5% high in years 11-17, and matching
    # only lifetime and year-20 energy left that shape error in place. The weights are therefore
    # fitted to the whole mean annual-energy path plus the 20-year hourly peak.
    moment_cols = np.column_stack([energy_y, load.max(axis=(1, 2))])
    weights = calibrate_weights(weights_cluster, moment_cols[reps], moment_cols.mean(axis=0), W_MIN, CAL_LAMBDA,
                                fixed=(tail_pos,) if TAIL_STRATUM else ())
    print(f"TAKSR K={K} on {len(body)} members: silhouette {sil:.3f}, bootstrap ARI {np.mean(aris):.2f} +- "
          f"{np.std(aris):.2f}; dropped constant features {dropped}")
    if TAIL_STRATUM:
        print(f"Upper tail: {len(tail_members)} members with 20-year peak >= {np.quantile(peak20, TAIL_QUANTILE):.0f} kW "
              f"-> scenario {tail_pos + 1} (member {tail_rep}, peak {peak20[tail_rep]:.0f} kW, "
              f"weight {weights[tail_pos]:.3f}); rule {TAIL_RULE}: rank {tail_peak_rank} by 20-year peak and "
              f"{tail_energy_rank} by 20-year energy ({energy20[tail_rep]:.0f} MWh) of {len(tail_members)}")
    else:
        print(f"Upper tail: disabled (MGPY_TAIL_STRATUM=0); all {N_MEMBERS} members reduced to K={K} scenarios")

    # ---- 6. deterministic scheme ----
    factor_p50 = np.median(factor, axis=0)
    rng_d = np.random.default_rng([SEED, 5])
    det_load = np.array([np.clip(m_year * factor_p50[y] * aligned_block_bootstrap(
        ratio, pools, LOAD_BLOCK, 8760, hour_of_day, rng_d), 0.0, None) for y in range(N_YEARS)])
    pv_energy = pv_per_kw.sum(axis=1)
    det_pv_member = int(np.argsort(pv_energy)[N_MEMBERS // 2])

    # ---- verification ----
    if LEVEL_SOURCE == "kwh":  # the level is energy, so compare monthly energy with the backbone's
        tgt = pd.Series(m_year, index=idx).groupby(idx.month).sum()
        wm_y1 = sum(w * pd.Series(load[r, 0], index=idx).groupby(idx.month).sum() for r, w in zip(reps, weights))
        mape_y1 = float((abs(wm_y1 - tgt) / tgt).mean() * 100)
        mape_basis = "monthly energy vs backbone"
    else:
        tgt = pd.Series({(m, w): targets[m][w] for m in range(1, 13) for w in ("WBP", "LWBP")})
        wm_y1 = sum(w * window_means(load[r, 0], idx) for r, w in zip(reps, weights))
        mape_y1 = float((abs(wm_y1 - tgt.reindex(wm_y1.index)) / tgt.reindex(wm_y1.index)).mean() * 100)
        mape_basis = "window means vs targets"
    peak_y = load.max(axis=2)
    peak_pct = {q: {"ensemble": float(np.percentile(peak_y[:, -1], q)),
                    "reduced": weighted_quantile(peak_y[reps, -1], weights, q / 100)} for q in (10, 50, 90, 95)}
    e_full, e_red = energy_y.mean(axis=0), np.average(energy_y[reps], axis=0, weights=weights)
    e_red_cluster = np.average(energy_y[reps], axis=0, weights=weights_cluster)
    e_err = 100 * (e_red / e_full - 1)
    print(f"Weights: cluster {np.round(weights_cluster, 3).tolist()} -> calibrated {weights.tolist()}")
    print(f"Annual-energy error vs ensemble, all 20 years: calibrated max |{np.abs(e_err).max():.2f}|%, "
          f"cluster-size max |{np.abs(100 * (e_red_cluster / e_full - 1)).max():.2f}|%")
    print(f"Year-1 weighted {mape_basis}: MAPE {mape_y1:.2f}%")
    print("Year-20 peak kW (ensemble / reduced): " + ", ".join(
        f"P{q} {v['ensemble']:.0f}/{v['reduced']:.0f}" for q, v in peak_pct.items()))
    print(f"Mean annual energy, reduced vs full: y1 {e_red[0]:.1f}/{e_full[0]:.1f} MWh, "
          f"y20 {e_red[-1]:.1f}/{e_full[-1]:.1f} MWh")
    # The 21st calibrated moment (2026-09-25; diagnostic only, nothing it prints feeds an output file).
    p_full, p_red = float(peak20.mean()), float(np.average(peak20[reps], weights=weights))
    print(f"Mean 20-year peak, reduced vs full: {p_red:.1f}/{p_full:.1f} kW ({100 * (p_red / p_full - 1):+.2f}%)")
    print(f"Deterministic: peak y1 {det_load[0].max():.0f} kW, y20 {det_load[-1].max():.0f} kW, "
          f"energy y1 {det_load[0].sum() / 1e3:.1f} MWh, y20 {det_load[-1].sum() / 1e3:.1f} MWh; "
          f"PV member {det_pv_member} ({pv_energy[det_pv_member]:.1f} kWh/kW)")

    # ---- outputs ----
    def demand_frame(arr):  # arr: (S, years, 8760) -> MicroGridsPy column (s-1)*years + y
        cols = {str(s * N_YEARS + y + 1): arr[s, y] for s in range(arr.shape[0]) for y in range(N_YEARS)}
        return pd.DataFrame(cols, index=range(8760))

    def res_frame(arr):  # arr: (S, 8760) per kW AC -> per MicroGridsPy RES unit, before its inverter factor
        return pd.DataFrame({s + 1: arr[s] * res_unit_kw / res_inv_eff for s in range(arr.shape[0])},
                            index=range(1, 8761))

    demand_frame(load[reps]).to_csv(paths["demand_prob"], sep=";", decimal=",", index=True, index_label="")
    res_frame(pv_per_kw[reps]).to_csv(paths["res_prob"])
    demand_frame(det_load[None]).to_csv(paths["demand_det"], sep=";", decimal=",", index=True, index_label="")
    res_frame(pv_per_kw[[det_pv_member]]).to_csv(paths["res_det"])
    pv_repairs = {}
    if TAIL_PV_REPAIR:  # AWO-002 B: the tail scenario's load with another member's PV year
        by_yield = np.argsort(pv_energy)
        for k, q in PV_REPAIR_QUANTILES.items():
            m = int(by_yield[int(round(q * (N_MEMBERS - 1)))])
            pv = pv_per_kw[reps].copy()
            pv[tail_pos] = pv_per_kw[m]
            res_frame(pv).to_csv(paths[f"res_tailpv_{k}"])
            pv_repairs[k] = {"pv_member": m, "pv_kwh_per_kw": float(pv_energy[m]), "res": OUT[f"res_tailpv_{k}"]}
        print(f"Tail PV re-pairings (tail's own PV member {tail_rep}, {pv_energy[tail_rep]:.1f} kWh/kW): " + ", ".join(
            f"{k} member {v['pv_member']} ({v['pv_kwh_per_kw']:.1f})" for k, v in pv_repairs.items()))

    summary = {
        "built": pd.Timestamp.now().isoformat(timespec="seconds"),
        "sources": {"load_forecasting": {"dir": LF_DIR, "commit": git_head(LF_DIR)},
                    "datascarce": {"dir": DS_DIR, "commit": git_head(DS_DIR)}},
        "settings": {"n_members": N_MEMBERS, "n_years": N_YEARS, "K": K, "year": YEAR, "load_block_h": LOAD_BLOCK,
                     "pv_block_h": PV_BLOCK, "pv_pdc_kw": PV_PDC_KW, "seed": SEED, "mu_g": mu_g, "sigma_g": sigma_g,
                     "growth_peak_years": [min(peaks), max(peaks)], "growth_yoy": [float(v) for v in g_yoy],
                     "res_unit_kw": res_unit_kw, "res_inverter_eff": res_inv_eff,
                     "level_source": LEVEL_SOURCE, "taksr_dev": TAKSR_DEV, "tail_rule": TAIL_RULE},
        "probabilistic": {"scenario_weights": [round(float(w), 6) for w in weights],
                          "cluster_size_weights": [round(float(w), 6) for w in weights_cluster],
                          "calibrated_moments": [f"energy_y{y}" for y in range(1, N_YEARS + 1)] + ["peak_20y"],
                          "calibration_lambda": CAL_LAMBDA,
                          "energy_error_pct_by_year": [round(float(v), 3) for v in e_err],
                          "representative_members": reps,
                          "silhouette": sil, "bootstrap_ari_mean": float(np.mean(aris)),
                          "bootstrap_ari_sd": float(np.std(aris)), "dropped_constant_features": dropped,
                          "peak_y20_percentiles_kw": peak_pct,
                          "max_hourly_load_kw": float(load[reps].max()),
                          "max_daily_energy_kwh": float(load[reps].reshape(len(reps), N_YEARS, 365, 24)
                                                        .sum(axis=3).max()),
                          "upper_tail": ({"quantile": TAIL_QUANTILE, "n_members": int(len(tail_members)),
                                          "scenario": tail_pos + 1, "member": tail_rep,
                                          "peak_20y_kw": float(peak20[tail_rep]),
                                          "weight": float(weights[tail_pos]), "rule": TAIL_RULE,
                                          "energy_20y_mwh": float(energy20[tail_rep]),
                                          "peak_rank": tail_peak_rank, "energy_rank": tail_energy_rank,
                                          "tail_pv_repairs": pv_repairs} if TAIL_STRATUM
                                         else {"enabled": False})},
        "deterministic": {"growth_factor_p50": [float(v) for v in factor_p50], "pv_member": det_pv_member,
                          "max_hourly_load_kw": float(det_load.max()),
                          "max_daily_energy_kwh": float(det_load.reshape(N_YEARS, 365, 24).sum(axis=2).max())},
        "diagnostics": {"raw_measured_over_target_bias": raw_bias, "year1_weighted_window_mape_pct": mape_y1,
                        "year1_mape_basis": mape_basis,
                        "pv_kwh_per_kw_mean": float(pv_energy.mean()),
                        "energy_mean_mwh_full": [float(v) for v in e_full],
                        "energy_mean_mwh_reduced": [float(v) for v in e_red],
                        "peak_20y_mean_kw": {"full": p_full, "reduced": p_red}},
        "outputs": OUT,
    }
    with open(paths["summary"], "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    for key in OUT:
        print("wrote", OUT[key])
    print("weights:", summary["probabilistic"]["scenario_weights"])


if __name__ == "__main__":
    main()
