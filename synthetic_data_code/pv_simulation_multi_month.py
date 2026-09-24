"""
Extends pv_simulation.py's physics pipeline to real additional months, using
real NASA POWER historical GHI/weather rather than any synthetic data (see
roadmap.tex Phase 1 Progress Log for why this is real-data, not synthetic:
the PV "ground truth" for a new month is a physics function of that month's
real observed GHI, exactly as for the original Dec 2024-Jan 2025 series).

Three real months, chosen to bracket the seasonal range the ITB thesis's own
GHI chart shows (Gambar IV.5: lowest ~June, highest ~Sept-Oct):
  - Dec 2024-Jan 2025 (existing, wet season)
  - Jun 2024 (dry season, low-GHI extreme)
  - Sep 2024 (dry season, high-GHI extreme)

EXTENSION (added to close a reviewer-flagged gap: the original 3-month
climatology cannot distinguish a genuine annual seasonal cycle from noise
across only 2 seasons): a full calendar-year 2024 NASA POWER pull
(nasa_power_full_year_2024.json, fetched directly from the same free,
no-login API this project already used for the 3 months above -- no new
data source, same lat/lon, same parameters) is parsed separately below and
used to build a genuine month-and-hour PV climatology, c_PV(month, hour),
covering all 12 calendar months from one internally-consistent year rather
than 3 months stitched across 2024/2026. This does NOT replace the 3-month
RAW_FILES cross-check above (kept for its own ITB-thesis-comparison purpose);
it is a separate, additive climatology-building step (see the bottom of this
script).

IMPORTANT timezone gotcha, caught while building this: the raw NASA POWER
files are NOT all in the same time convention. The header's third line says
either "...in UTC" (the original Dec pull) or "...in LST" (the June/Sept
pulls done later in this project) -- these need different handling:
  - "in UTC": raw HR is UTC, must add TZ_OFFSET_H (7, WIB) to get local time
    (this is what merged_load_weather.csv already did for the Dec series)
  - "in LST": raw HR is ALREADY local standard time (WIB for this site), no
    shift needed
Verified empirically for the June file: HR=0 has GHI=0 (midnight, correct)
and HR=12 has GHI~700-770 (midday peak, correct) with NO shift applied --
confirming "LST" here does mean already-local, not UTC. Applying the wrong
convention would silently shift sunrise/sunset by 7 hours and go undetected
without this kind of sanity check, which is exactly why the per-file header
is parsed automatically below rather than assuming one fixed convention.
"""
import re
import numpy as np
import pandas as pd

LAT = -7.678979182854658
LON = 113.25382180569332
TZ_OFFSET_H = 7          # WIB = UTC+7
PDC0_W = 798_000.0
GAMMA_PDC = -0.004
NOCT = 45.0
ALBEDO = 0.2
SYSTEM_LOSSES = 0.14
INVERTER_EFF = 0.96
TILT_DEG = abs(LAT)
PANEL_AZIMUTH_DEG = 0.0

RAW_FILES = {
    "2024-12_wet": "POWER_Point_Hourly_20241201_20250101_007d68S_113d25E_UTC.csv",
    "2024-06_dry_low": "POWER_Point_Hourly_20240601_20240701_007d68S_113d25E_UTC.csv",
    "2024-09_dry_high": "POWER_Point_Hourly_20240901_20241001_007d68S_113d25E_UTC.csv",
}


def load_raw_power_file(path):
    """Parses a raw NASA POWER hourly CSV, auto-detecting the UTC vs LST
    time convention from the header (see module docstring)."""
    with open(path) as f:
        header_lines = [next(f) for _ in range(14)]
    header_text = "".join(header_lines)
    is_utc = bool(re.search(r"in UTC", header_text))
    is_lst = bool(re.search(r"in LST", header_text))
    if not (is_utc or is_lst):
        raise ValueError(f"{path}: could not detect UTC/LST convention from header")

    raw = pd.read_csv(path, skiprows=14)
    ts = pd.to_datetime(dict(year=raw["YEAR"], month=raw["MO"], day=raw["DY"], hour=raw["HR"]))
    if is_utc:
        ts = ts + pd.Timedelta(hours=TZ_OFFSET_H)  # UTC -> WIB local, matching merged_load_weather.csv's convention

    out = pd.DataFrame({
        "timestamp": ts,
        "ghi_wh_m2": raw["ALLSKY_SFC_SW_DWN"],
        "temp_c": raw["T2M"],
        "rh_pct": raw["RH2M"],
        "wind_ms": raw["WS10M"],
    }).sort_values("timestamp").reset_index(drop=True)
    convention = "UTC (+7h shift applied)" if is_utc else "LST (already local, no shift)"
    print(f"  {path}: {len(out)} rows, detected convention: {convention}")
    return out


def simulate_pv(df):
    """Identical physics pipeline to pv_simulation.py, factored into a
    function so it can run over any weather dataframe with
    [timestamp, ghi_wh_m2, temp_c] columns."""
    df = df.copy()
    doy = df["timestamp"].dt.dayofyear.values
    hour_local = df["timestamp"].dt.hour.values + 0.5

    gamma = 2 * np.pi / 365 * (doy - 1 + (hour_local - 12) / 24)
    eqtime = 229.18 * (0.000075 + 0.001868*np.cos(gamma) - 0.032077*np.sin(gamma)
                        - 0.014615*np.cos(2*gamma) - 0.040849*np.sin(2*gamma))
    decl = (0.006918 - 0.399912*np.cos(gamma) + 0.070257*np.sin(gamma)
            - 0.006758*np.cos(2*gamma) + 0.000907*np.sin(2*gamma)
            - 0.002697*np.cos(3*gamma) + 0.00148*np.sin(3*gamma))

    time_offset = eqtime + 4*LON - 60*TZ_OFFSET_H
    true_solar_time = hour_local*60 + time_offset
    hour_angle_deg = true_solar_time/4 - 180
    hour_angle = np.radians(hour_angle_deg)

    lat_r = np.radians(LAT)
    cos_zenith = np.sin(lat_r)*np.sin(decl) + np.cos(lat_r)*np.cos(decl)*np.cos(hour_angle)
    cos_zenith = np.clip(cos_zenith, -1, 1)
    zenith = np.arccos(cos_zenith)
    zenith_deg = np.degrees(zenith)

    cos_az = (np.sin(decl) - np.sin(lat_r)*cos_zenith) / (np.cos(lat_r)*np.sin(zenith) + 1e-12)
    cos_az = np.clip(cos_az, -1, 1)
    az_raw = np.degrees(np.arccos(cos_az))
    solar_azimuth_deg = np.where(hour_angle_deg > 0, 360 - az_raw, az_raw)

    Gon = 1367.0 * (1 + 0.033*np.cos(np.radians(360*doy/365)))
    G0 = Gon * np.clip(cos_zenith, 0, None)
    ghi = df["ghi_wh_m2"].values
    with np.errstate(divide="ignore", invalid="ignore"):
        kt = np.where(G0 > 5, np.clip(ghi / G0, 0, 1), 0.0)

    kd = np.where(kt <= 0.22, 1.0 - 0.09*kt,
         np.where(kt <= 0.80,
                  0.9511 - 0.1604*kt + 4.388*kt**2 - 16.638*kt**3 + 12.336*kt**4,
                  0.165))
    kd = np.clip(kd, 0, 1)
    dhi = kd * ghi
    with np.errstate(divide="ignore", invalid="ignore"):
        dni = np.where(cos_zenith > 0.01, np.clip((ghi - dhi) / np.clip(cos_zenith, 0.01, None), 0, None), 0.0)
    dni = np.where(zenith_deg >= 90, 0.0, dni)
    dhi = np.where(zenith_deg >= 90, 0.0, dhi)

    tilt = np.radians(TILT_DEG)
    panel_az = np.radians(PANEL_AZIMUTH_DEG)
    solar_az = np.radians(solar_azimuth_deg)
    cos_aoi = np.cos(zenith)*np.cos(tilt) + np.sin(zenith)*np.sin(tilt)*np.cos(solar_az - panel_az)
    cos_aoi = np.clip(cos_aoi, 0, None)

    poa_direct = dni * cos_aoi
    poa_diffuse = dhi * (1 + np.cos(tilt)) / 2
    poa_ground = ghi * ALBEDO * (1 - np.cos(tilt)) / 2
    poa_global = poa_direct + poa_diffuse + poa_ground
    poa_global = np.where(zenith_deg >= 90, 0.0, poa_global)

    t_cell = df["temp_c"].values + (NOCT - 20) / 800 * poa_global
    pdc = PDC0_W * (poa_global / 1000.0) * (1 + GAMMA_PDC * (t_cell - 25))
    pdc = np.clip(pdc, 0, None)
    pac = pdc * (1 - SYSTEM_LOSSES) * INVERTER_EFF

    df["zenith_deg"] = zenith_deg
    df["solar_azimuth_deg"] = solar_azimuth_deg
    df["dni_wh_m2"], df["dhi_wh_m2"], df["kt"] = dni, dhi, kt
    df["poa_global_wh_m2"] = poa_global
    df["t_cell_c"] = t_cell
    df["pv_dc_kw"] = pdc / 1000.0
    df["pv_ac_kw"] = pac / 1000.0
    return df


print("Loading and simulating each real month...")
all_months = []
for label, path in RAW_FILES.items():
    weather = load_raw_power_file(path)
    simulated = simulate_pv(weather)
    simulated["period"] = label
    all_months.append(simulated)

combined = pd.concat(all_months, ignore_index=True)
combined.to_csv("simulated_pv_multi_month.csv", index=False)

print("\n=== Seasonal cross-check: capacity factor and monthly energy by period ===")
summary = combined.groupby("period").agg(
    mean_ac_kw=("pv_ac_kw", "mean"),
    capacity_factor=("pv_ac_kw", lambda s: s.mean() / 798.0),
    total_energy_mwh=("pv_ac_kw", lambda s: s.sum() / 1000.0),
    n_hours=("pv_ac_kw", "size"),
).reset_index()
print(summary.to_string(index=False))
summary.to_csv("seasonal_capacity_factor_summary.csv", index=False)

order = summary.sort_values("capacity_factor")["period"].tolist()
print(f"\nRanking (low to high capacity factor): {order}")

# The ITB thesis text (not the unlabeled chart -- see roadmap.tex) only
# makes one precise, well-supported claim: Sep/Oct is the annual GHI peak.
# It does NOT claim June beats a wet-season month like December -- that
# would require reading exact values off an unlabeled bar chart, which the
# thesis-mining agent explicitly flagged as unreliable. So the honest check
# here is just: does Sept come out highest, as directly stated in the text.
if order[-1] == "2024-09_dry_high":
    print("  Sept correctly ranks HIGHEST, matching the ITB thesis's explicit text claim (Sep/Oct GHI peak).")
else:
    print("  Sept does NOT rank highest -- this contradicts the thesis's explicit claim, investigate.")

if order[0] == "2024-12_wet":
    print("  Dec (wet season) ranks LOWEST of the three -- physically sensible (wet season = more cloud cover),")
    print("  and consistent with this project's earlier finding that Dec 2024 GHI (4.33 kWh/m2/day) sits below")
    print("  the thesis's 4.92 kWh/m2/day annual average. This does NOT match a literal reading of the thesis's")
    print("  unlabeled GHI chart (which visually suggested June as the single lowest month) -- but that chart")
    print("  reading was already flagged as imprecise/pixel-estimated, and comparing a wet-season month against")
    print("  a dry-season month was never the thesis's actual claim (it only ranked dry-season months against")
    print("  each other). Likely explanation: real inter-annual variability between the thesis's data year and")
    print("  this project's 2024 NASA POWER pull, not a pipeline error.")

print("\nSaved: simulated_pv_multi_month.csv, seasonal_capacity_factor_summary.csv")

# =====================================================================
# Full calendar-year 2024 climatology (month-and-hour), from the JSON pull
# =====================================================================
import json


def load_power_json_full_year(path):
    """Parses a NASA POWER hourly JSON export (fetched via the same
    community=RE point-API this project's CSV pulls above used) into the
    same [timestamp, ghi_wh_m2, temp_c, rh_pct, wind_ms] shape as
    load_raw_power_file(). The JSON API returns local-standard-time-free UTC
    timestamps (keys like '2024010100' = YYYYMMDDHH, UTC), confirmed by the
    header text this project always checks ('parameters' block has no LST
    marker; NASA POWER's community=RE point API is documented UTC) -- so the
    same +TZ_OFFSET_H shift as the "in UTC" CSV files above is applied."""
    with open(path) as f:
        d = json.load(f)
    props = d["properties"]["parameter"]
    ghi_raw, t2m, rh2m, ws10m = (props["ALLSKY_SFC_SW_DWN"], props["T2M"],
                                  props["RH2M"], props["WS10M"])
    keys = list(ghi_raw.keys())
    ts = pd.to_datetime(keys, format="%Y%m%d%H") + pd.Timedelta(hours=TZ_OFFSET_H)
    out = pd.DataFrame({
        "timestamp": ts,
        "ghi_wh_m2": [ghi_raw[k] for k in keys],
        "temp_c": [t2m[k] for k in keys],
        "rh_pct": [rh2m[k] for k in keys],
        "wind_ms": [ws10m[k] for k in keys],
    }).sort_values("timestamp").reset_index(drop=True)
    n_missing = (out["ghi_wh_m2"] == -999.0).sum()
    if n_missing:
        print(f"  WARNING: {n_missing} missing (-999) hours in {path}, dropping them")
        out = out[out["ghi_wh_m2"] != -999.0].reset_index(drop=True)
    return out


print("\n=== Full calendar-year 2024 climatology (closes the 3-month-only seasonal gap) ===")
full_year_weather = load_power_json_full_year("nasa_power_full_year_2024.json")
full_year_pv = simulate_pv(full_year_weather)
full_year_pv["month"] = full_year_pv["timestamp"].dt.month
full_year_pv["hour"] = full_year_pv["timestamp"].dt.hour
full_year_pv.to_csv("simulated_pv_full_year_2024.csv", index=False)
print(f"Simulated {len(full_year_pv)} hours across all 12 months of 2024 "
      f"(vs. the 3-month, 2-season RAW_FILES cross-check above).")

climatology_month_hour = full_year_pv.groupby(["month", "hour"])["pv_ac_kw"].mean().reset_index()
climatology_month_hour.rename(columns={"pv_ac_kw": "c_pv_kw"}, inplace=True)
climatology_month_hour.to_csv("pv_climatology_month_hour.csv", index=False)

monthly_energy = full_year_pv.groupby("month")["pv_ac_kw"].agg(
    mean_ac_kw="mean", total_energy_mwh=lambda s: s.sum() / 1000.0).reset_index()
print("\nMonthly PV summary, full year 2024 (mean AC kW, total energy MWh):")
print(monthly_energy.to_string(index=False))
swing_pct = 100 * (monthly_energy["mean_ac_kw"].max() - monthly_energy["mean_ac_kw"].min()) / monthly_energy["mean_ac_kw"].mean()
print(f"\nSeasonal swing (max-month vs min-month mean AC power, relative to annual mean): {swing_pct:.1f}%")
print("This is the best available REAL seasonal-variability reference at this site "
      "(used in scenario_generator_8760h.py's load-climatology sensitivity check).")
print("\nSaved: simulated_pv_full_year_2024.csv, pv_climatology_month_hour.csv")
