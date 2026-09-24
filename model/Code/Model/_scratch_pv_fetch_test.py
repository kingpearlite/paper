import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.environ['MGPY_PARAMS'] = os.path.abspath('../Inputs/Parameters_9sc_20y_gili_ketapang_MILP.dat')
os.environ['MGPY_RES'] = os.path.abspath('../Inputs/RES_Time_Series_9sc_gili_ketapang.csv')
import RE_calculation as rc

DATE_START = "20150101"
DATE_END = "20241231"
# NASA POWER hourly JSON endpoint caps at 8 years per request (tested: 2015-2022 OK,
# 2015-2023 fails). Split into two chunks that together cover the full 10-year window,
# each safely under the cap.
HOURLY_CHUNKS = [("20150101", "20221231"), ("20230101", "20241231")]

data_import = open(os.environ['MGPY_PARAMS']).readlines()
patched = []
for line in data_import:
    if 'param: date_start' in line:
        patched.append(f"param: date_start := '{DATE_START}';\n")
    elif 'param: date_end' in line:
        patched.append(f"param: date_end := '{DATE_END}';\n")
    else:
        patched.append(line)

(date_start, date_end, lat, lon, lat_ext_1, lon_ext_1, lat_ext_2, lon_ext_2,
 standard_lon, URL_1_d, URL_2_d, periods) = rc.URL_creation_d(patched)
URL_daily = URL_1_d + URL_2_d
print("daily URLs:", len(URL_daily))

# Build hourly URLs once per chunk by patching date_start/date_end for URL_creation_h only.
hourly_urls_by_chunk = []
for (cs, ce) in HOURLY_CHUNKS:
    patched_h = []
    for line in data_import:
        if 'param: date_start' in line:
            patched_h.append(f"param: date_start := '{cs}';\n")
        elif 'param: date_end' in line:
            patched_h.append(f"param: date_end := '{ce}';\n")
        else:
            patched_h.append(line)
    urls_h = rc.URL_creation_h(patched_h)
    hourly_urls_by_chunk.append(urls_h)
    print(f"chunk {cs}-{ce}: {len(urls_h)} hourly URLs")

t0 = time.time()
jsdata_daily = rc.multithread_data_download(URL_daily)
print("daily fetch:", round(time.time() - t0, 1), "s")

t0 = time.time()
jsdata_hourly_chunks = [rc.multithread_data_download(urls) for urls in hourly_urls_by_chunk]
print("hourly fetch (both chunks):", round(time.time() - t0, 1), "s")

# Merge each corner's two chunked hourly responses into one JSON blob spanning the
# full 10-year window, preserving the exact top-level structure data_2D_interpolation
# expects (so it can be fed in exactly like a single, unchunked response).
param_hourly_str = ['WS50M', 'WS2M', 'WD50M', 'T2M']
merged_hourly = []
n_corners = len(hourly_urls_by_chunk[0])
for corner in range(n_corners):
    chunk_a = json.loads(jsdata_hourly_chunks[0][corner])
    chunk_b = json.loads(jsdata_hourly_chunks[1][corner])
    merged = chunk_a  # reuse chunk A's structure (geometry, header, etc.)
    for p in param_hourly_str:
        # Python 3.7+ dicts preserve insertion order; chunk A's keys already end
        # at 2022123123, chunk B's start at 2023010100, so a simple update()
        # appends B's keys in chronological order after A's.
        merged['properties']['parameter'][p].update(chunk_b['properties']['parameter'][p])
    merged_hourly.append(json.dumps(merged))

n_keys = len(json.loads(merged_hourly[0])['properties']['parameter']['T2M'])
print("merged hourly key count (expect 87648 for 10y, 87672 with 2 leap days=8784*2+8760*8):", n_keys)

jsdata = jsdata_daily + merged_hourly
# NOTE: data_2D_interpolation expects the '&start=YYYYMMDD'/'&end=YYYYMMDD'-prefixed
# strings URL_creation_d returns (it slices them at fixed offsets), NOT the plain
# DATE_START/DATE_END constants used to build the hourly chunk URLs above.
param_daily_interp, param_hourly_interp = rc.data_2D_interpolation(
    jsdata, date_start, date_end, lat, lon, lat_ext_1, lon_ext_1, lat_ext_2, lon_ext_2)

print("param_daily_interp: params x years x months =", len(param_daily_interp), len(param_daily_interp[0]), len(param_daily_interp[0][0]))
print("param_hourly_interp: params x years x months =", len(param_hourly_interp), len(param_hourly_interp[0]), len(param_hourly_interp[0][0]))

import numpy as np
for yr in range(len(param_daily_interp[0])):
    irr = [v for month in param_daily_interp[0][yr] for v in month]
    print(f"year {2015+yr}: n_days={len(irr)} mean_irr={np.mean(irr):.3f} min={np.min(irr):.3f} max={np.max(irr):.3f}")

# Sanity check hourly reconstruction: total hours per year
for yr in range(len(param_hourly_interp[0])):
    hrs = sum(len(day) for month in param_hourly_interp[0][yr] for day in month)
    print(f"year {2015+yr}: hourly count = {hrs}")

# --- Step 1: PV physics conversion, per real year instead of one collapsed TMY ---
# Mirrors RE_supply()'s own I_tilt / T_cell / energy_PV block (RE_calculation.py:829-853),
# just run once per real year (param_daily_interp[0][year], param_hourly_interp[3][year])
# instead of once on the single Finkelstein-Schafer-selected typical year.
(nom_power, tilt, azim, ro_ground, k_T, NMOT, T_NMOT, G_NMOT) = rc.solarPV_parameters(data_import)
print(f"\nPV module params: nom_power={nom_power}W tilt={tilt} azim={azim} k_T={k_T}%/C NMOT={NMOT}C")

n_years = len(param_daily_interp[0])
energy_PV_by_year = []  # [year] -> flat 8760(or 8784)-length Wh/module list

for yr in range(n_years):
    irr_daily_yr = param_daily_interp[0][yr]     # [month][day] -> daily irradiance
    T_amb_yr = param_hourly_interp[3][yr]        # [month][day][hour] -> T2M

    I_tilt = [[] for _ in range(12)]
    day_of_year = 1
    for month in range(12):
        for day in range(len(irr_daily_yr[month])):
            I_tilt[month].append(rc.hourly_solar(irr_daily_yr[month][day], lat, lon, standard_lon,
                                                  day_of_year, tilt, azim, ro_ground))
            day_of_year += 1

    energy_PV_yr = [[] for _ in range(12)]
    for month in range(12):
        energy_PV_yr[month] = [[] for _ in range(len(T_amb_yr[month]))]
        for day in range(len(T_amb_yr[month])):
            for hour in range(len(T_amb_yr[month][day])):
                t_cell = T_amb_yr[month][day][hour] + ((NMOT - T_NMOT) / G_NMOT) * I_tilt[month][day][hour] * 1000
                p = I_tilt[month][day][hour] * nom_power * (1 + (k_T / 100) * (t_cell - 25))
                energy_PV_yr[month][day].append(max(p, 0.0))  # PV output can't be physically negative

    flat = [v for month in energy_PV_yr for day in month for v in day]
    energy_PV_by_year.append(flat)
    print(f"year {2015+yr}: n_hours={len(flat)} sum_kWh={sum(flat)/1000:.1f} peak_W={max(flat):.1f} "
          f"capacity_factor={sum(flat)/1000/(nom_power/1000*len(flat)):.4f}")

# Persist the 10 real, independent yearly PV traces for inspection before building the
# Markov-chain shape generator on top of them.
import csv
out_path = os.path.abspath('../Inputs/_scratch_pv_multiyear_raw.csv')
with open(out_path, 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow([str(2015 + yr) for yr in range(n_years)])
    max_len = max(len(c) for c in energy_PV_by_year)
    for row in range(max_len):
        w.writerow([energy_PV_by_year[yr][row] if row < len(energy_PV_by_year[yr]) else '' for yr in range(n_years)])
print(f"\nSaved 10 real yearly PV traces (Wh/module) to {out_path}")
